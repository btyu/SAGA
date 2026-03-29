# conda install -c conda-forge rdkit
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from scileo_agent.core.registry import get_scorer
from scileo_agent.core.data_models.candidate import Candidate

# Ensure scorer class is imported so its scorers are registered
import modules.small_molecule_drug_design.scorer.druglikeness_scorer  # noqa: F401

def build_catalogs():
    cats = {}
    for name, kind in [
        ("PAINS_A", FilterCatalogParams.FilterCatalogs.PAINS_A),
        ("PAINS_B", FilterCatalogParams.FilterCatalogs.PAINS_B),
        ("PAINS_C", FilterCatalogParams.FilterCatalogs.PAINS_C),
        ("BRENK",   FilterCatalogParams.FilterCatalogs.BRENK),
    ]:
        p = FilterCatalogParams(); p.AddCatalog(kind)
        cats[name] = FilterCatalog(p)
    return cats

def find_alerts(mol, catalogs):
    hits = []
    for cat_name, cat in catalogs.items():
        for i in range(cat.GetNumEntries()):
            entry = cat.GetEntry(i)
            smarts = getattr(entry, "GetSmarts", lambda: "")()
            if not smarts:
                continue
            q = Chem.MolFromSmarts(smarts)
            if not q:
                continue
            for atom_ids in mol.GetSubstructMatches(q, useChirality=False, useQueryQueryMatches=True):
                frag = Chem.MolFragmentToSmiles(mol, atoms=list(atom_ids), kekuleSmiles=True, isomericSmiles=True)
                hits.append({
                    "catalog": cat_name,
                    "alert": entry.GetDescription() or "unnamed_alert",
                    "smarts": smarts,
                    "atom_ids": list(atom_ids),
                    "frag_smiles": frag,
                })
    return hits

def highlight_svg(mol, atom_ids, path):
    m2 = Chem.Mol(mol); AllChem.Compute2DCoords(m2)
    bonds = {b.GetIdx() for b in m2.GetBonds() if b.GetBeginAtomIdx() in atom_ids and b.GetEndAtomIdx() in atom_ids}
    d = Draw.rdMolDraw2D.MolDraw2DSVG(600, 450)
    d.drawOptions().addAtomIndices = False
    d.DrawMolecule(m2, highlightAtoms=atom_ids, highlightBonds=list(bonds))
    d.FinishDrawing(); Path(path).write_text(d.GetDrawingText())

def analyze_smiles(smi, out_dir="alerts_out", name="molecule"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    mol = Chem.MolFromSmiles(smi)
    if not mol: raise ValueError(f"Bad SMILES: {smi}")
    try: Chem.SanitizeMol(mol)
    except Exception as e: print(f"[warn] sanitize: {e}")

    cats = build_catalogs()
    alerts = find_alerts(mol, cats)

    # Align counts with scorer: use RDKit catalog-level unique-entry matches
    pains_n = sum(len(cats[k].GetMatches(mol)) for k in ("PAINS_A", "PAINS_B", "PAINS_C"))
    brenk_n = len(cats["BRENK"].GetMatches(mol))
    print(f"PAINS: {pains_n}, BRENK: {brenk_n}")
    

if __name__ == "__main__":
    # Example
    smi = "COc1cccc(C(C=O)NCNc2cncc(F)n2)c1"
    analyze_smiles(smi, out_dir="alerts_out", name="example")
