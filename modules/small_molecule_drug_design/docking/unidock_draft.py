""" Code from https://github.com/tsa87/cgflow.
"""

import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol
from unidock_tools.application.unidock_pipeline import UniDock


def run_etkdg(mol: RDMol, sdf_path: Path | str, seed: int = 1) -> bool:
    if mol.GetNumAtoms() == 0:
        return False
    try:
        param = AllChem.srETKDGv3()
        param.randomSeed = seed
        mol = Chem.Mol(mol)
        mol.RemoveAllConformers()
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, param)
        mol = Chem.RemoveHs(mol)
        assert mol.GetNumConformers() > 0
        with Chem.SDWriter(str(sdf_path)) as w:
            w.write(mol)
    except Exception:
        return False
    else:
        return True


def docking(
    rdmols: list[RDMol],
    protein_path: str | Path,
    center: tuple[float, float, float],
    seed: int = 1,
    size: float | tuple[float, float, float] = 20.0,
    search_mode: str = "balance",
) -> list[tuple[None, float] | tuple[RDMol, float]]:
    if isinstance(size, float | int):
        size = (size, size, size)

    protein_path = Path(protein_path)

    out_dir = "/gpfs/radev/home/tl688/pitl688/scileoagent_drug/dockingout_proteins/"
    out_dir = Path(out_dir)
    sdf_list = []
    for i, mol in enumerate(rdmols):
        ligand_file = out_dir / f"{i}.sdf"
        flag = run_etkdg(mol, ligand_file)
        if flag:
            sdf_list.append(ligand_file)
    if len(sdf_list) > 0:
        runner = UniDock(
            receptor=protein_path,
            ligands=sdf_list,
            center_x=round(center[0], 3),
            center_y=round(center[1], 3),
            center_z=round(center[2], 3),
            size_x=round(size[0], 3),
            size_y=round(size[1], 3),
            size_z=round(size[2], 3),
            workdir=out_dir / "workdir",
        )
        runner.docking(
            save_dir=out_dir / "savedir",
            search_mode=search_mode,
            num_modes=1,
            seed=seed,
        )

    res: list[tuple[None, float] | tuple[RDMol, float]] = []
    for i in range(len(rdmols)):
        try:
            docked_file = out_dir / "savedir" / f"{i}.sdf"
            docked_rdmol: Chem.Mol = list(
                Chem.SDMolSupplier(str(docked_file)))[0]
            assert docked_rdmol is not None
            docking_score = float(docked_rdmol.GetProp("docking_score"))
        except Exception:
            docked_rdmol, docking_score = None, 0.0
        res.append((docked_rdmol, docking_score))
    return res


def scoring(
    rdmols: list[RDMol],
    protein_path: str | Path,
    center: tuple[float, float, float],
    size: float = 25.0,
):
    protein_path = Path(protein_path)

    out_dir = "/gpfs/radev/home/tl688/pitl688/scileoagent_drug/dockingout_proteins/"
    out_dir = Path(out_dir)
    sdf_list = []
    for i, mol in enumerate(rdmols):
        ligand_file = out_dir / f"{i}.sdf"
        try:
            lig_pos = np.array(mol.GetConformer().GetPositions())
            min_x, min_y, min_z = lig_pos.min(0)
            assert center[0] - size / 2 < min_x
            assert center[1] - size / 2 < min_y
            assert center[2] - size / 2 < min_z
            max_x, max_y, max_z = lig_pos.max(0)
            assert center[0] + size / 2 > max_x
            assert center[1] + size / 2 > max_y
            assert center[2] + size / 2 > max_z
            with Chem.SDWriter(str(ligand_file)) as w:
                w.write(mol)
        except Exception:
            pass
        else:
            sdf_list.append(ligand_file)
    if len(sdf_list) > 0:
        runner = UniDock(
            receptor=protein_path,
            ligands=sdf_list,
            center_x=round(center[0], 3),
            center_y=round(center[1], 3),
            center_z=round(center[2], 3),
            size_x=round(size, 3),
            size_y=round(size, 3),
            size_z=round(size, 3),
            workdir=out_dir / "workdir",
        )
        try:
            runner.docking(save_dir=out_dir / "savedir", score_only=True)
        except Exception:
            pass

    res: list[tuple[None, float] | tuple[RDMol, float]] = []
    for i in range(len(rdmols)):
        docked_file = str(out_dir / "savedir" / f"{i}.sdf")
        try:
            docked_rdmol = list(
                Chem.SDMolSupplier(str(docked_file), sanitize=False))[0]
            assert docked_rdmol is not None
            docking_score = float(docked_rdmol.GetProp("docking_score"))
        except Exception:
            docked_rdmol, docking_score = None, 0.0
        res.append((docked_rdmol, docking_score))
    return res


if __name__ == "__main__":
    """Dock small molecules to DRD2 receptor."""
    from rdkit import Chem

    # Example SMILES (aspirin, caffeine, ibuprofen)
    smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine  
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # ibuprofen
    ]

    # Convert SMILES to RDMol objects
    rdmols = [Chem.MolFromSmiles(smi) for smi in smiles]
    rdmols = [mol for mol in rdmols if mol is not None]

    # DRD2 protein and pocket parameters
    # protein_path = "/projects/jlab/to.shen/SciLeoAgent/modules/small_molecule_drug_design/data/pdb/DRD2.pdb"
    curr_filepath = Path(__file__).resolve()
    protein_path = str(curr_filepath.parent.parent / "data" / "pdb" / "DRD2.pdb")

    assert Path(protein_path).exists()
    center = (9.925, 5.846, -9.582)

    # Run docking
    results = docking(rdmols, protein_path, center, seed=42)

    # Print results
    for i, (mol, score) in enumerate(results):
        if mol is not None:
            print(f"Molecule {i}: Docking score = {score:.2f}")
        else:
            print(f"Molecule {i}: Docking failed")
