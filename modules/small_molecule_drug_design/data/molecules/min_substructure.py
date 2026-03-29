# pip install rdkit-pypi pandas
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS, Recap
from rdkit.Chem.rdmolops import GetMolFrags
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.rdmolfiles import MolToSmiles
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN
from collections import Counter, defaultdict
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
import pickle
import os
from tqdm import tqdm


# ------------- 1) Load + STANDARDIZE -------------
def is_oral_druglike(mol, max_mw=500):
    """Filter for drug-like molecular weight."""
    mw = Descriptors.MolWt(mol)
    return mw <= max_mw


def standardize(smiles_list, filter_druglike=True):
    normalizer = rdMolStandardize.Normalizer()
    reionizer = rdMolStandardize.Reionizer()

    out = []
    for smi in tqdm(smiles_list, desc="Standardizing molecules"):
        m = Chem.MolFromSmiles(smi)
        if not m:
            continue
        # remove salts
        parent = rdMolStandardize.FragmentParent(m)
        # normalize + reionize
        parent = normalizer.normalize(parent)
        parent = reionizer.reionize(parent)
        # skip tautomer enumeration (too slow for large datasets)
        Chem.SanitizeMol(parent)

        # Filter for oral drug-like properties
        if filter_druglike and not is_oral_druglike(parent):
            continue

        out.append(parent)
    return out


def standardize_zinc(smiles_list):
    """Simplified standardization for ZINC (no drug-like filtering)."""
    out = []
    for smi in tqdm(smiles_list, desc="Processing ZINC molecules"):
        m = Chem.MolFromSmiles(smi)
        if m:
            out.append(m)
    return out


# ------------- 2) Murcko Scaffolds -------------
def murcko_scaffolds(mols, min_atoms=5, max_atoms=20):
    c = Counter()
    exemplars = {}
    for m in tqdm(mols, desc="Extracting Murcko scaffolds"):
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        n_atoms = scaf.GetNumAtoms()
        if n_atoms < min_atoms or n_atoms > max_atoms:
            continue
        smi = Chem.MolToSmiles(scaf, isomericSmiles=False)
        c[smi] += 1
        exemplars.setdefault(smi, scaf)
    return c, exemplars


def cluster_by_scaffold(mols, min_atoms=5, max_atoms=20):
    """Cluster molecules by their Murcko scaffold (ring patterns)."""
    clusters = defaultdict(list)
    for i, m in enumerate(mols):
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        n_atoms = scaf.GetNumAtoms()
        if n_atoms < min_atoms or n_atoms > max_atoms:
            smi = "OTHER"
        else:
            smi = Chem.MolToSmiles(scaf, isomericSmiles=False)
        clusters[smi].append(i)
    return clusters


# ------------- 3) BRICS and RECAP fragments -------------
def brics_fragments(mols, min_atoms=5, max_atoms=20):
    c = Counter()
    for m in tqdm(mols, desc="Extracting BRICS fragments"):
        # BRICSDecompose returns a set of fragment SMILES (with wildcard attachment points)
        frags = BRICS.BRICSDecompose(m, minFragmentSize=2)
        for frag in frags:
            frag_mol = Chem.MolFromSmiles(frag)
            if frag_mol:
                n_atoms = frag_mol.GetNumAtoms()
                if min_atoms <= n_atoms <= max_atoms:
                    c[frag] += 1
    return c


def recap_fragments(mols, min_atoms=5, max_atoms=20):
    c = Counter()
    for m in tqdm(mols, desc="Extracting RECAP fragments"):
        try:
            tree = Recap.RecapDecompose(m)
        except Exception:
            continue
        if not tree:
            continue
        leaves = tree.GetLeaves()
        for smi in leaves.keys():
            frag_mol = Chem.MolFromSmiles(smi)
            if frag_mol:
                n_atoms = frag_mol.GetNumAtoms()
                if min_atoms <= n_atoms <= max_atoms:
                    c[smi] += 1
    return c


# ------------- 4) Frequent atom-environments (ECFP) → SMARTS -------------
def env_smarts_from_morgan(mols, radius=2, nBits=2048, top_k=100):
    """
    Convert frequent ECFP bits back to representative substructures (SMARTS-like).
    Note: hashed bits collide; we enumerate all atom centers recorded in bitInfo.
    """
    bit_counts = Counter()
    per_bit_examples = defaultdict(list)

    # first pass: count bits + store example environments
    for m in tqdm(mols, desc="Computing Morgan fingerprints"):
        bitInfo = defaultdict(list)
        fp = rdMD.GetMorganFingerprintAsBitVect(m,
                                                radius,
                                                nBits=nBits,
                                                bitInfo=bitInfo)
        onbits = list(fp.GetOnBits())
        bit_counts.update(onbits)
        for b in onbits:
            # keep up to a few examples per bit for later reconstruction
            if len(per_bit_examples[b]) < 5:
                per_bit_examples[b].append((m, bitInfo[b]))

    # second pass: reconstruct submols for top bits
    records = []
    for bit, cnt in tqdm(bit_counts.most_common(top_k),
                         desc="Reconstructing environments"):
        env_smiles = set()
        # build a few example fragments to represent this bit
        for m, centers in per_bit_examples[bit]:
            for (atom_idx, rad) in centers:
                bonds = FindAtomEnvironmentOfRadiusN(m, rad, atom_idx)
                if not bonds:
                    continue
                atoms = set()
                for bidx in bonds:
                    b = m.GetBondWithIdx(bidx)
                    atoms.add(b.GetBeginAtomIdx())
                    atoms.add(b.GetEndAtomIdx())
                try:
                    amap = {}
                    for i, a in enumerate(sorted(atoms)):
                        amap[a] = i
                    sub = Chem.PathToSubmol(m, list(bonds), atomMap=amap)
                    sm = Chem.MolToSmiles(sub, isomericSmiles=False)
                    env_smiles.add(sm)
                except Exception:
                    continue
        records.append({
            "bit": bit,
            "count": cnt,
            "examples": ";".join(sorted(env_smiles))[:500]
        })
    df = pd.DataFrame(records).sort_values("count", ascending=False)
    return df


# ------------- Utility: export & quick grid image -------------
def counter_to_df(counter, label="fragment", total=None):
    total = total or sum(counter.values())
    data = [{
        "fragment": k,
        "count": v,
        "coverage": v / total
    } for k, v in counter.most_common()]
    return pd.DataFrame(data)


def compute_tfidf(target_counter, background_counter, n_target, n_background):
    """
    Compute TF-IDF scores for fragments.
    TF = frequency in target / n_target
    IDF = log((n_target + n_background) / (count_in_target + count_in_background))
    TF-IDF = TF × IDF
    Higher scores = enriched in target vs background
    """
    records = []
    for frag, target_count in target_counter.items():
        bg_count = background_counter.get(frag, 0)
        tf = target_count / n_target
        idf = np.log((n_target + n_background) / (target_count + bg_count + 1))
        tfidf = tf * idf
        records.append({
            "fragment": frag,
            "count_target": target_count,
            "count_background": bg_count,
            "tf": tf,
            "idf": idf,
            "tfidf": tfidf,
            "enrichment": target_count / (bg_count + 1)
        })
    return pd.DataFrame(records).sort_values("tfidf", ascending=False)


def compute_enrichment_ranking(target_counter,
                               background_counter,
                               n_target,
                               n_background,
                               min_target_count=3):
    """
    Better ranking for antibiotic-specific scaffolds:
    1. Enrichment ratio (fold change)
    2. Minimum count threshold to avoid noise
    3. Statistical significance (Fisher's exact test)
    """
    from scipy.stats import fisher_exact
    import numpy as np

    records = []
    for frag, target_count in target_counter.items():
        bg_count = background_counter.get(frag, 0)

        # Skip if too few occurrences in target
        if target_count < min_target_count:
            continue

        # Enrichment ratio
        enrichment = target_count / (bg_count + 1)

        # Fisher's exact test
        # Contingency table: [target_with, target_without, bg_with, bg_without]
        target_with = target_count
        target_without = n_target - target_count
        bg_with = bg_count
        bg_without = n_background - bg_count

        try:
            odds_ratio, p_value = fisher_exact([[target_with, target_without],
                                                [bg_with, bg_without]])
        except:
            odds_ratio, p_value = 1.0, 1.0

        # Combined score: enrichment * -log(p_value) * target_frequency
        target_freq = target_count / n_target
        significance_score = -np.log10(max(p_value, 1e-10))  # Avoid log(0)
        combined_score = enrichment * significance_score * target_freq

        records.append({
            "fragment": frag,
            "count_target": target_count,
            "count_background": bg_count,
            "enrichment": enrichment,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "target_frequency": target_freq,
            "significance_score": significance_score,
            "combined_score": combined_score
        })

    return pd.DataFrame(records).sort_values("combined_score", ascending=False)


def draw_top_smiles(smiles_iterable,
                    n=16,
                    molSize=(500, 500),
                    molsPerRow=4,
                    counts=None,
                    total=None,
                    enrichment_scores=None):
    ms = []
    legends = []
    for i, smi in enumerate(smiles_iterable[:n]):
        m = Chem.MolFromSmiles(smi)
        if m:
            ms.append(m)
            if enrichment_scores is not None:
                # Show enrichment score
                score = enrichment_scores[i]
                cnt = counts[i] if counts else 0
                legends.append(f"n={cnt}\nScore={score:.3f}")
            elif counts and total:
                cnt = counts[i]
                pct = (cnt / total) * 100
                legends.append(f"n={cnt} ({pct:.1f}%)")
            else:
                legends.append("")
    if not ms: return None
    img = MolsToGridImage(ms,
                          molsPerRow=molsPerRow,
                          subImgSize=molSize,
                          legends=legends if legends else None)
    # return PIL Image
    bio = BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return Image.open(bio)


# ------------- Example usage -------------
if __name__ == "__main__":
    import os

    # Read SMILES from existing_antibiotics.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    antibiotics_file = os.path.join(script_dir, "combined_antibiotics.txt")

    print(f"Reading antibiotics from: {antibiotics_file}")

    with open(antibiotics_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(smiles_list)} SMILES strings")

    mols = standardize(smiles_list, filter_druglike=True)
    print(f"Standardized to {len(mols)} molecules (MW≤500)")

    # Create output directory
    out_dir = os.path.join(script_dir, "antibiotic_substructures_mw500")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Murcko (5-20 atoms)
    print("Computing Murcko scaffolds (5-20 atoms)...")
    scaf_counts, scaf_ex = murcko_scaffolds(mols, min_atoms=5, max_atoms=20)
    df_scaf = counter_to_df(scaf_counts, label="murcko")
    out_scaf = os.path.join(out_dir, "murcko_scaffolds.csv")
    df_scaf.to_csv(out_scaf, index=False)
    print(f"  Wrote {out_scaf} ({len(df_scaf)} scaffolds)")

    # BRICS (5-20 atoms)
    print("Computing BRICS fragments (5-20 atoms)...")
    df_brics = counter_to_df(brics_fragments(mols, min_atoms=5, max_atoms=20),
                             label="brics")
    out_brics = os.path.join(out_dir, "brics_fragments.csv")
    df_brics.to_csv(out_brics, index=False)
    print(f"  Wrote {out_brics} ({len(df_brics)} fragments)")

    # RECAP (5-20 atoms)
    print("Computing RECAP fragments (5-20 atoms)...")
    df_recap = counter_to_df(recap_fragments(mols, min_atoms=5, max_atoms=20),
                             label="recap")
    out_recap = os.path.join(out_dir, "recap_fragments.csv")
    df_recap.to_csv(out_recap, index=False)
    print(f"  Wrote {out_recap} ({len(df_recap)} fragments)")

    # ECFP environments → SMARTS-like examples
    print("Computing Morgan fingerprint environments...")
    df_env = env_smarts_from_morgan(mols, radius=2, nBits=2048, top_k=200)
    out_env = os.path.join(out_dir, "morgan_envs_top200.csv")
    df_env.to_csv(out_env, index=False)
    print(f"  Wrote {out_env}")

    # ============ TF-IDF vs ZINC ============
    print("\n[TF-IDF] Loading ZINC reference set...")
    zinc_file = os.path.join(script_dir, "zinc_250k.txt")
    zinc_cache_file = os.path.join(script_dir, "zinc_fragments_cache.pkl")

    if os.path.exists(zinc_file):
        # Check for cached ZINC fragments
        if os.path.exists(zinc_cache_file):
            print("  Loading cached ZINC fragments...")
            with open(zinc_cache_file, "rb") as f:
                zinc_data = pickle.load(f)
            zinc_scaf_counts = zinc_data["scaf_counts"]
            zinc_brics_counts = zinc_data["brics_counts"]
            zinc_recap_counts = zinc_data["recap_counts"]
            print(
                f"  Loaded cached fragments for {zinc_data['n_mols']} ZINC molecules"
            )
        else:
            with open(zinc_file, "r") as f:
                zinc_smiles = [line.strip() for line in f
                               if line.strip()]  # Sample 10k for speed
            print(f"  Loaded {len(zinc_smiles)} ZINC molecules")

            zinc_mols = standardize_zinc(zinc_smiles)
            print(f"  Processed to {len(zinc_mols)} ZINC molecules")

            # Extract ZINC fragments
            print("[TF-IDF] Extracting ZINC fragments...")
            zinc_scaf_counts, _ = murcko_scaffolds(zinc_mols,
                                                   min_atoms=5,
                                                   max_atoms=20)
            zinc_brics_counts = brics_fragments(zinc_mols,
                                                min_atoms=5,
                                                max_atoms=20)
            zinc_recap_counts = recap_fragments(zinc_mols,
                                                min_atoms=5,
                                                max_atoms=20)

            # Cache the results
            print("  Caching ZINC fragments...")
            zinc_data = {
                "scaf_counts": zinc_scaf_counts,
                "brics_counts": zinc_brics_counts,
                "recap_counts": zinc_recap_counts,
                "n_mols": len(zinc_mols)
            }
            with open(zinc_cache_file, "wb") as f:
                pickle.dump(zinc_data, f)
            print(f"  Cached fragments for {len(zinc_mols)} ZINC molecules")

        # Compute enrichment rankings (better than TF-IDF for chemical scaffolds)
        print("[Enrichment] Computing enrichment scores...")
        n_zinc_mols = zinc_data["n_mols"]
        df_scaf_enrich = compute_enrichment_ranking(scaf_counts,
                                                    zinc_scaf_counts,
                                                    len(mols),
                                                    n_zinc_mols,
                                                    min_target_count=3)
        df_brics_enrich = compute_enrichment_ranking(dict(
            df_brics.set_index("fragment")["count"]),
                                                     zinc_brics_counts,
                                                     len(mols),
                                                     n_zinc_mols,
                                                     min_target_count=3)
        df_recap_enrich = compute_enrichment_ranking(dict(
            df_recap.set_index("fragment")["count"]),
                                                     zinc_recap_counts,
                                                     len(mols),
                                                     n_zinc_mols,
                                                     min_target_count=3)

        # Save enrichment results
        out_scaf_enrich = os.path.join(out_dir,
                                       "murcko_scaffolds_enrichment.csv")
        df_scaf_enrich.to_csv(out_scaf_enrich, index=False)
        print(f"  Wrote {out_scaf_enrich}")

        out_brics_enrich = os.path.join(out_dir,
                                        "brics_fragments_enrichment.csv")
        df_brics_enrich.to_csv(out_brics_enrich, index=False)
        print(f"  Wrote {out_brics_enrich}")

        out_recap_enrich = os.path.join(out_dir,
                                        "recap_fragments_enrichment.csv")
        df_recap_enrich.to_csv(out_recap_enrich, index=False)
        print(f"  Wrote {out_recap_enrich}")

        # Generate enrichment-ranked images
        print("[Enrichment] Generating antibiotic-specific motif images...")

        # Murcko scaffolds
        scaf_enrich_records = df_scaf_enrich.to_dict("records")
        img_scaf_enrich = draw_top_smiles(
            [r["fragment"] for r in scaf_enrich_records],
            n=40,
            molsPerRow=8,
            counts=[r["count_target"] for r in scaf_enrich_records[:40]],
            total=len(mols),
            enrichment_scores=[
                r["combined_score"] for r in scaf_enrich_records[:40]
            ])
        if img_scaf_enrich:
            out_img_enrich = os.path.join(out_dir,
                                          "top_scaffolds_enrichment.png")
            img_scaf_enrich.save(out_img_enrich)
            print(f"  Wrote {out_img_enrich}")

        # BRICS fragments
        brics_enrich_records = df_brics_enrich.to_dict("records")
        img_brics_enrich = draw_top_smiles(
            [r["fragment"] for r in brics_enrich_records],
            n=40,
            molsPerRow=8,
            counts=[r["count_target"] for r in brics_enrich_records[:40]],
            total=len(mols),
            enrichment_scores=[
                r["combined_score"] for r in brics_enrich_records[:40]
            ])
        if img_brics_enrich:
            out_img_brics_enrich = os.path.join(out_dir,
                                                "top_brics_enrichment.png")
            img_brics_enrich.save(out_img_brics_enrich)
            print(f"  Wrote {out_img_brics_enrich}")

        # RECAP fragments
        recap_enrich_records = df_recap_enrich.to_dict("records")
        img_recap_enrich = draw_top_smiles(
            [r["fragment"] for r in recap_enrich_records],
            n=40,
            molsPerRow=8,
            counts=[r["count_target"] for r in recap_enrich_records[:40]],
            total=len(mols),
            enrichment_scores=[
                r["combined_score"] for r in recap_enrich_records[:40]
            ])
        if img_recap_enrich:
            out_img_recap_enrich = os.path.join(out_dir,
                                                "top_recap_enrichment.png")
            img_recap_enrich.save(out_img_recap_enrich)
            print(f"  Wrote {out_img_recap_enrich}")
    else:
        print(f"  ZINC file not found at {zinc_file}, skipping TF-IDF")

    # Quick visual sanity-check
    print("Generating images...")
    scaf_records = df_scaf.to_dict("records")
    img = draw_top_smiles([r["fragment"] for r in scaf_records],
                          n=40,
                          molsPerRow=8,
                          counts=[r["count"] for r in scaf_records[:40]],
                          total=len(mols))
    if img:
        out_img = os.path.join(out_dir, "top_scaffolds.png")
        img.save(out_img)
        print(f"  Wrote {out_img}")

    brics_records = df_brics.to_dict("records")
    img2 = draw_top_smiles([r["fragment"] for r in brics_records],
                           n=40,
                           molsPerRow=8,
                           counts=[r["count"] for r in brics_records[:40]],
                           total=len(mols))
    if img2:
        out_img2 = os.path.join(out_dir, "top_brics.png")
        img2.save(out_img2)
        print(f"  Wrote {out_img2}")

    recap_records = df_recap.to_dict("records")
    img3 = draw_top_smiles([r["fragment"] for r in recap_records],
                           n=40,
                           molsPerRow=8,
                           counts=[r["count"] for r in recap_records[:40]],
                           total=len(mols))
    if img3:
        out_img3 = os.path.join(out_dir, "top_recap.png")
        img3.save(out_img3)
        print(f"  Wrote {out_img3}")

    # ============ Scaffold-based Clustering ============
    print("\n[Clustering] Grouping molecules by ring patterns...")
    clusters = cluster_by_scaffold(mols, min_atoms=5, max_atoms=20)

    # Create cluster summary
    cluster_records = []
    for scaffold_smi, mol_indices in sorted(clusters.items(),
                                            key=lambda x: len(x[1]),
                                            reverse=True):
        cluster_records.append({
            "scaffold": scaffold_smi,
            "n_molecules": len(mol_indices),
            "percentage": len(mol_indices) / len(mols) * 100
        })

    df_clusters = pd.DataFrame(cluster_records)
    out_clusters = os.path.join(out_dir, "scaffold_clusters.csv")
    df_clusters.to_csv(out_clusters, index=False)
    print(f"  Found {len(clusters)} unique scaffold clusters")
    print(f"  Wrote {out_clusters}")

    # Visualize top clusters (show representative molecules)
    print("[Clustering] Generating cluster visualization...")
    cluster_mols = []
    cluster_legends = []
    for i, (scaffold_smi, mol_indices) in enumerate(
            sorted(clusters.items(), key=lambda x: len(x[1]),
                   reverse=True)[:40]):
        if scaffold_smi == "OTHER":
            continue
        # Take first molecule from cluster as representative
        if mol_indices:
            cluster_mols.append(mols[mol_indices[0]])
            cluster_legends.append(
                f"n={len(mol_indices)}\n({len(mol_indices)/len(mols)*100:.1f}%)"
            )

    if cluster_mols:
        img_clusters = MolsToGridImage(cluster_mols,
                                       molsPerRow=8,
                                       subImgSize=(500, 500),
                                       legends=cluster_legends)
        bio = BytesIO()
        img_clusters.save(bio, format="PNG")
        bio.seek(0)
        img_clusters_pil = Image.open(bio)
        out_img_clusters = os.path.join(out_dir, "scaffold_clusters.png")
        img_clusters_pil.save(out_img_clusters)
        print(f"  Wrote {out_img_clusters}")

    print(f"\n✓ Complete! Results in: {out_dir}")
