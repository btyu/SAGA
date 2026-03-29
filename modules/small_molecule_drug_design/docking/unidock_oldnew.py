""" Code from https://github.com/tsa87/cgflow.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
from modules.small_molecule_drug_design.docking import plip_analyze
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol
from unidock_tools.application.unidock_pipeline import UniDock
import parmed

module_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(module_dir)  # modules/small_molecule_drug_design


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
    output_path: str | Path | None = None,
) -> list[tuple[None, float] | tuple[RDMol, float]]:
    if isinstance(size, float | int):
        size = (size, size, size)

    protein_path = Path(protein_path)

    with tempfile.TemporaryDirectory() as out_dir:
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
                docked_rdmols = list(Chem.SDMolSupplier(str(docked_file)))
                docked_rdmol: Chem.Mol = docked_rdmols[0]

                assert docked_rdmol is not None
                docking_score = float(docked_rdmol.GetProp("docking_score"))

                # Save docked pose if output path is specified
                if output_path is not None and docked_rdmol is not None:
                    output_file = Path(output_path) / f"docked_{i}.sdf"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with Chem.SDWriter(str(output_file)) as w:
                        for mol in docked_rdmols:
                            w.write(mol)

            except Exception:
                docked_rdmol, docking_score = None, 0.0
            res.append((docked_rdmol, docking_score))
    return res


def scoring(
    rdmols: list[RDMol],
    protein_path: str | Path,
    center: tuple[float, float, float],
    size: float = 25.0,
    output_path: str | Path | None = None,
):
    protein_path = Path(protein_path)

    with tempfile.TemporaryDirectory() as out_dir:
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
                docked_rdmols = list(
                    Chem.SDMolSupplier(str(docked_file), sanitize=False))
                docked_rdmol = docked_rdmols[0]
                assert docked_rdmol is not None
                docking_score = float(docked_rdmol.GetProp("docking_score"))

                # Save scored pose if output path is specified
                if output_path is not None and docked_rdmol is not None:
                    output_file = Path(output_path) / f"scored_{i}.sdf"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with Chem.SDWriter(str(output_file)) as w:
                        for mol in docked_rdmols:
                            w.write(mol)

            except Exception:
                docked_rdmol, docking_score = None, 0.0
            res.append((docked_rdmol, docking_score))
    return res


def mpro_plip_score_computation(sdf_path: str | Path,
                                debug=False) -> dict[str, int]:
    """Analyze an MPRO complex and return the maximum number of interactions present.

    Args:
        sdf_path: Path to a docked ligand SDF file (potentially multiple poses).

    Returns:
        Dict mapping three interaction identifiers to 1 (present) or 0 (absent):
        "hydrogen_bond/HIS/161/A", "hydrogen_bond/GLU/164/A", "pi_stack/HIS/39/A".
    """
    protein_structure = parmed.load_file(base_dir + f'/data/pdb/MPRO.pdb')

    supplier = Chem.SDMolSupplier(str(sdf_path))

    targets = [
        "hydrogen_bond/HIS/161/A",
        "hydrogen_bond/GLU/164/A",
        "pi_stack/HIS/39/A",
    ]

    best_presence: dict[str, int] | None = None
    best_count = -1
    best_report: dict | None = None
    found_any = False

    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        found_any = True

        ligand_pdb_path = str(sdf_path).replace('.sdf', f'_{i}.pdb')
        Chem.MolToPDBFile(mol, ligand_pdb_path)

        ligand_structure = parmed.load_file(ligand_pdb_path)
        complex_structure = protein_structure + ligand_structure
        complex_pdb_path = ligand_pdb_path.replace('.pdb', '_complex.pdb')
        complex_structure.save(complex_pdb_path, overwrite=True)

        report = plip_analyze.plip_analyze_single_frame(complex_pdb_path,
                                                        mol_name='UNL')

        presence = {key: (1 if key in report else 0) for key in targets}
        count = sum(presence.values())

        if debug:
            present = [k for k, v in presence.items() if v]
            print(
                f"[PLIP] mol_index={i} interaction_count={count} present={present} presence={presence}"
            )

        if count > best_count:
            best_count = count
            best_presence = presence
            best_report = report

    if not found_any:
        if debug:
            raise ValueError("No ligand found in SDF file")
        return {
            "hydrogen_bond/HIS/161/A": 0,
            "hydrogen_bond/GLU/164/A": 0,
            "pi_stack/HIS/39/A": 0,
        }

    if debug and best_report is not None:
        print(best_report)

    assert best_presence is not None
    return best_presence


if __name__ == "__main__":
    """Dock small molecules to MPRO receptor and test MPRO integration."""
    from rdkit import Chem

    smiles = [
        "CN(Cc1cccc(F)c1)C(=O)Cn1nnc2ccccc21",
        "O=C(Cn1nnc2ccccc21)N1CC=CCC[C@@H]1c1ccccc1",
        "CCN(C(=O)Cn1nnc2ccccc21)C(c1ccccc1)c1ccccc1"
    ]

    # Convert SMILES to RDMol objects
    rdmols = [Chem.MolFromSmiles(smi) for smi in smiles]
    rdmols = [mol for mol in rdmols if mol is not None]

    # MPRO protein and pocket parameters
    curr_filepath = Path(__file__).resolve()
    protein_path = str(curr_filepath.parent.parent / "data" / "pdb" /
                       "MPRO.pdb")

    assert Path(protein_path).exists()
    center = (9.050, 8.898, -1.508)

    # Run docking with optional output path
    output_dir = "temp/docked_poses_mpro"
    results = docking(rdmols,
                      protein_path,
                      center,
                      seed=42,
                      output_path=output_dir)

    # Print results and test MPRO interaction scoring integration
    for i, (mol, score) in enumerate(results):
        if mol is not None:
            print(f"Molecule {i}: Docking score = {score:.2f}")
            sdf_file = Path(output_dir) / f"docked_{i}.sdf"
            if sdf_file.exists():
                presence = mpro_plip_score_computation(sdf_file, debug=True)
                print(presence)
                print(
                    f"  MPRO interactions -> HIS161_A: {presence['hydrogen_bond/HIS/161/A']}, "
                    f"GLU164_A: {presence['hydrogen_bond/GLU/164/A']}, "
                    f"HIS39_A: {presence['pi_stack/HIS/39/A']}")
            else:
                print(f"  Docked SDF not found for Molecule {i}")
        else:
            print(f"Molecule {i}: Docking failed")
