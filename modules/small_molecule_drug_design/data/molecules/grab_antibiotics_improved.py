#!/usr/bin/env python3
import csv, sys, time, json, urllib.parse, itertools, re
from collections import OrderedDict
import requests
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
# ATC groups that contain antibacterials (systemic + non-systemic + TB):
ATC_PREFIXES = [
    "J01",  # Antibacterials for systemic use
    "J04",  # Antimycobacterials (TB, etc.)
    "A07AA",  # Intestinal antibiotics
    "D06AX",  # Other topical antibiotics
    "S01AA",  # Ophthalmological antibiotics
    "S02AA",  # Otological antibiotics
]

# ChEMBL keyword search to catch investigational/non-ATC items
KEYWORDS = ["antibiotic", "antibacterial"]

OUT_CSV = "antibiotics_all.csv"
REQUESTS_TIMEOUT = 40
SLEEP = 0.1  # gentle throttle


# -----------------------------
# HELPERS
# -----------------------------
def get_json(url, params=None, headers=None):
    for attempt in range(3):
        try:
            r = requests.get(url,
                             params=params,
                             headers=headers,
                             timeout=REQUESTS_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            elif r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
            else:
                sys.stderr.write(f"[WARN] {url} -> HTTP {r.status_code}\n")
                return None
        except requests.RequestException as e:
            sys.stderr.write(f"[WARN] {url} -> {e}\n")
            time.sleep(1.5 * (attempt + 1))
    return None


def paginated_chembl(url, params, desc="Fetching"):
    """Yield combined ChEMBL JSON rows across pages that use page_meta.next."""
    total_fetched = 0
    with tqdm(desc=desc, unit="molecules") as pbar:
        while True:
            obj = get_json(url, params)
            if not obj:
                break

            # molecules endpoint returns 'molecules'; drug endpoint returns 'drugs'; search returns 'page_meta'
            key = next((k
                        for k in ("molecules", "drugs", "mechanisms",
                                  "molecule_synonyms", "activities", "stages")
                        if k in obj), None)
            if key:
                batch_size = len(obj[key])
                total_fetched += batch_size
                pbar.update(batch_size)
                pbar.set_postfix({"total": total_fetched})

                for row in obj[key]:
                    yield row

            nxt = obj.get("page_meta", {}).get("next")
            if not nxt:
                break
            # ChEMBL returns relative next URLs
            url = "https://www.ebi.ac.uk" + nxt
            params = {}  # already encoded in next
            time.sleep(SLEEP)


def safe_get(d, *path, default=None):
    cur = d
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


def add_row(index, key, row):
    # prefer first occurrence; keep set for provenance if you like
    if key not in index:
        index[key] = row


def is_antibiotic_related(molecule):
    """Check if molecule is antibiotic-related based on name, ATC codes, or keywords"""
    name = (molecule.get("pref_name") or "").lower()

    # Check ATC codes - handle different data structures
    atc_codes = []
    atc_classifications = molecule.get("atc_classifications", [])
    if isinstance(atc_classifications, list):
        for c in atc_classifications:
            if isinstance(c, dict):
                code = c.get("code", "")
                if code:
                    atc_codes.append(code)
            elif isinstance(c, str):
                atc_codes.append(c)

    # Check ATC codes
    for atc in atc_codes:
        if any(atc.startswith(prefix) for prefix in ATC_PREFIXES):
            return True

    # Check name for antibiotic keywords
    antibiotic_keywords = [
        "antibiotic", "antibacterial", "antimicrobial", "bactericidal",
        "bacteriostatic", "penicillin", "cephalosporin", "tetracycline",
        "quinolone", "macrolide", "aminoglycoside", "sulfonamide",
        "trimethoprim", "vancomycin", "clindamycin", "erythromycin",
        "azithromycin", "ciprofloxacin", "levofloxacin", "doxycycline",
        "amoxicillin", "ampicillin", "ceftriaxone", "gentamicin",
        "streptomycin", "chloramphenicol", "rifampin", "isoniazid",
        "ethambutol", "pyrazinamide"
    ]
    for keyword in antibiotic_keywords:
        if keyword in name:
            return True

    return False


# -----------------------------
# 1) ChEMBL by therapeutic flag and name filtering
# -----------------------------
def pull_chembl_therapeutic():
    results = OrderedDict()
    print("Fetching ChEMBL therapeutic molecules...")

    base = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
    params = {
        "molecule_type": "Small molecule",
        "therapeutic_flag": True,
        "limit": 200,
        "offset": 0
    }

    for m in paginated_chembl(base, params, "Therapeutic molecules"):
        if not is_antibiotic_related(m):
            continue

        chembl_id = m.get("molecule_chembl_id")
        name = m.get("pref_name")
        struct = m.get("molecule_structures") or {}
        smiles = struct.get("canonical_smiles")
        inchikey = struct.get("standard_inchi_key")
        # Extract ATC codes safely
        atc_codes = []
        for c in m.get("atc_classifications", []):
            if isinstance(c, dict) and c.get("code"):
                atc_codes.append(c.get("code"))
            elif isinstance(c, str):
                atc_codes.append(c)
        atc_codes = ";".join(atc_codes)
        phase = m.get("max_phase")

        if not smiles:  # Skip molecules without SMILES
            continue

        key = inchikey or smiles or chembl_id
        add_row(
            results, key, {
                "source": "ChEMBL_Therapeutic",
                "chembl_id": chembl_id,
                "name": name,
                "smiles": smiles,
                "inchikey": inchikey,
                "atc_codes": atc_codes,
                "max_phase": phase,
            })
    return results


# -----------------------------
# 2) ChEMBL keyword search for investigational/non-ATC molecules
# -----------------------------
def pull_chembl_by_keywords():
    results = OrderedDict()
    print("Fetching ChEMBL molecules by keyword search...")

    for kw in KEYWORDS:
        print(f"  Searching for: {kw}")
        url = "https://www.ebi.ac.uk/chembl/api/data/search.json"
        params = {"q": kw, "limit": 200, "offset": 0}

        for hit in paginated_chembl(url, params, f"Search:{kw}"):
            if hit.get("entity_type") != "Molecule":
                continue

            chembl_id = hit.get("entity_id")
            # fetch molecule detail to get SMILES/IK and ATC (if any)
            m = get_json(
                f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
            )
            if not m or not is_antibiotic_related(m):
                continue

            name = m.get("pref_name")
            struct = m.get("molecule_structures") or {}
            smiles = struct.get("canonical_smiles")
            inchikey = struct.get("standard_inchi_key")
            # Extract ATC codes safely
            atc_codes = []
            for c in m.get("atc_classifications", []):
                if isinstance(c, dict) and c.get("code"):
                    atc_codes.append(c.get("code"))
                elif isinstance(c, str):
                    atc_codes.append(c)
            atc_codes = ";".join(atc_codes)
            phase = m.get("max_phase")

            if not smiles:  # Skip molecules without SMILES
                continue

            key = inchikey or smiles or chembl_id
            add_row(
                results, key, {
                    "source": f"ChEMBL_SEARCH:{kw}",
                    "chembl_id": chembl_id,
                    "name": name,
                    "smiles": smiles,
                    "inchikey": inchikey,
                    "atc_codes": atc_codes,
                    "max_phase": phase,
                })
            time.sleep(0.05)
        time.sleep(SLEEP)
    return results


# -----------------------------
# MAIN
# -----------------------------
def main():
    master = OrderedDict()

    print("=== Antibiotic Data Collection ===")
    print(f"Target ATC prefixes: {', '.join(ATC_PREFIXES)}")
    print(f"Search keywords: {', '.join(KEYWORDS)}")
    print()

    # Step 1: Therapeutic molecules search
    by_therapeutic = pull_chembl_therapeutic()
    master.update(by_therapeutic)
    print(f"Found {len(by_therapeutic)} molecules from therapeutic search")

    # Step 2: Keyword search
    by_kw = pull_chembl_by_keywords()
    for k, v in by_kw.items():
        master.setdefault(k, v)  # keep first source if duplicate
    print(f"Found {len(by_kw)} additional molecules from keyword search")

    # Combine and deduplicate
    rows = list(master.values())

    # Sort by phase (approved drugs first) and name
    def phase_val(x):
        try:
            return int(x.get("max_phase") or 0)
        except:
            return 0

    rows.sort(key=lambda r: (-phase_val(r), (r["name"] or "")))

    # Write CSV
    if not rows:
        print("No antibiotic molecules found. Try adjusting search criteria.",
              file=sys.stderr)
        return

    print(f"\nWriting {len(rows)} antibiotic molecules to {OUT_CSV}...")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f,
                           fieldnames=[
                               "source", "chembl_id", "name", "smiles",
                               "inchikey", "atc_codes", "max_phase"
                           ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Done! Wrote {len(rows)} antibiotic molecules to {OUT_CSV}")

    # Summary stats
    phases = {}
    sources = {}
    for r in rows:
        phase = r.get("max_phase") or "Unknown"
        source = r.get("source", "Unknown")
        phases[phase] = phases.get(phase, 0) + 1
        sources[source] = sources.get(source, 0) + 1

    print("\n=== Summary ===")
    print("By development phase:")
    for phase, count in sorted(phases.items(),
                               key=lambda x: int(x[0])
                               if x[0].isdigit() else 999):
        print(f"  Phase {phase}: {count}")

    print("\nBy source:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
