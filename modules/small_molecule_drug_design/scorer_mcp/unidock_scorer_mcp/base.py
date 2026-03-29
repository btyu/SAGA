import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from unidock_tools.application.unidock_pipeline import UniDock

from .scorer_utils import BaseScorer, scorer
from .docking_utils import mpro_plip_score_computation, generate_residue_map, get_ligand_coords_from_sdf

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ProteinTarget:
    protein_name: str
    pdb_path: str
    pocket_center: Tuple[float, float, float]


class Scorer(BaseScorer):
    """Collection of UniDock scoring functions for protein-ligand docking."""

    def __init__(self):
        # Call parent constructor to set up registry
        super().__init__()

        # Define PDB data directory
        pdb_dir = os.path.join(CURRENT_FILE_DIR, "scorer_data", "pdb")

        # Initialize protein targets
        self.protein_targets = {
            "DRD2": ProteinTarget(
                protein_name="DRD2",
                pdb_path=os.path.join(pdb_dir, "DRD2.pdb"),
                pocket_center=(9.925, 5.846, -9.582),
            ),
            "GSK3B": ProteinTarget(
                protein_name="GSK3B",
                pdb_path=os.path.join(pdb_dir, "GSK3B.pdb"),
                pocket_center=(-14.782, -17.079, -3.559),
            ),
            "JNK3": ProteinTarget(
                protein_name="JNK3",
                pdb_path=os.path.join(pdb_dir, "JNK3.pdb"),
                pocket_center=(23.167, 8.921, 31.848),
            ),
            "BRD4": ProteinTarget(
                protein_name="BRD4",
                pdb_path=os.path.join(pdb_dir, "BRD4.pdb"),
                pocket_center=(28.751, 15.826, -2.335),
            ),
            "MPRO": ProteinTarget(
                protein_name="MPRO",
                pdb_path=os.path.join(pdb_dir, "MPRO.pdb"),
                pocket_center=(9.050, 8.898, -1.508),
            ),
            "mu_opioid_clean": ProteinTarget(
                protein_name="mu_opioid_clean",
                pdb_path=os.path.join(pdb_dir, "mu_opioid_clean.pdb"),
                pocket_center=(1.2857, 16.4479, -59.1143),
            ),
            "ampc_clean": ProteinTarget(
                protein_name="ampc_clean",
                pdb_path=os.path.join(pdb_dir, "ampc_clean.pdb"),
                pocket_center=(23.5567, 5.7351, 14.3991),
            )
        }

        self.seed = 42
        self.pocket_distance_cutoff = 12.0

        # Store interaction scores for MPRO (key: SMILES, value: dict of interactions)
        self._mpro_interactions = {}

    def _normalize_score(self, score: float) -> float:
        """Normalize UniDock score to 0-1 range.

        Args:
            score: UniDock score in kcal/mol (negative values indicate binding)

        Returns:
            Normalized score where 0 = no binding, 1 = strong binding (<= -15 kcal/mol)
        """
        if score is None:
            return 0.0

        # UniDock scores are negative for binding (lower = better binding)
        # Normalize: 0 = no binding, 1 = strong binding (<= -15 kcal/mol)
        normalized = np.clip(-score / 15.0, 0.0, 1.0)
        return normalized

    def _run_etkdg(self, mol, sdf_path: Path, seed: int = 1) -> bool:
        """Generate 3D conformer using ETKDG."""
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

    def _docking(
        self,
        smiles_list: List[str],
        protein_path: str,
        center: Tuple[float, float, float],
        seed: int = 1,
        size: float = 20.0,
        search_mode: str = "balance",
        output_path: Optional[str] = None,
    ) -> List[Tuple[Optional[object], float]]:
        """Perform molecular docking."""
        if isinstance(size, (float, int)):
            size = (size, size, size)

        protein_path = Path(protein_path)

        with tempfile.TemporaryDirectory() as out_dir:
            out_dir = Path(out_dir)
            sdf_list = []
            valid_indices = []

            for i, smiles in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                ligand_file = out_dir / f"{i}.sdf"
                flag = self._run_etkdg(mol, ligand_file, seed)
                if flag:
                    sdf_list.append(ligand_file)
                    valid_indices.append(i)

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

            res: List[Tuple[Optional[object], float]] = []
            for i in range(len(smiles_list)):
                try:
                    docked_file = out_dir / "savedir" / f"{i}.sdf"
                    docked_rdmols = list(Chem.SDMolSupplier(str(docked_file)))
                    docked_rdmol = docked_rdmols[0]

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

    def _score_unidock(
        self,
        samples: List[str],
        protein_target: ProteinTarget,
        compute_mpro_interactions: bool = False
    ) -> List[Optional[float]]:
        """Score molecules using UniDock docking."""
        # Create unique temporary directory for this batch
        docked_poses_dir = os.path.join(
            tempfile.gettempdir(),
            f"unidock_docking_{uuid.uuid4()}"
        )
        os.makedirs(docked_poses_dir, exist_ok=True)

        try:
            # Run docking
            docking_results = self._docking(
                samples,
                protein_target.pdb_path,
                protein_target.pocket_center,
                seed=self.seed,
                output_path=docked_poses_dir
            )

            # Extract normalized scores
            results = []
            for i, (mol, score) in enumerate(docking_results):
                smiles = samples[i]
                normalized = self._normalize_score(score) if mol is not None else None
                results.append(normalized)

                # Compute MPRO interactions if requested
                if compute_mpro_interactions and protein_target.protein_name == "MPRO" and mol is not None:
                    sdf_filepath = os.path.join(docked_poses_dir, f"docked_{i}.sdf")
                    if os.path.exists(sdf_filepath):
                        try:
                            interaction_score = mpro_plip_score_computation(sdf_filepath)
                            self._mpro_interactions[smiles] = {
                                "HIS161_A": interaction_score["hydrogen_bond/HIS/161/A"],
                                "GLU164_A": interaction_score["hydrogen_bond/GLU/164/A"],
                                "HIS39_A": interaction_score["pi_stack/HIS/39/A"]
                            }
                        except Exception:
                            self._mpro_interactions[smiles] = {
                                "HIS161_A": 0,
                                "GLU164_A": 0,
                                "HIS39_A": 0
                            }
                    else:
                        self._mpro_interactions[smiles] = {
                            "HIS161_A": 0,
                            "GLU164_A": 0,
                            "HIS39_A": 0
                        }
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(docked_poses_dir):
                shutil.rmtree(docked_poses_dir)

        return results

    @scorer(
        name="drd2_unidock",
        population_wise=False,
        description=(
            "DRD2 UniDock binding affinity score (value range: 0.0 to 1.0). "
            "This score represents the normalized binding affinity to the dopamine D2 receptor (DRD2) calculated using UniDock molecular docking. "
            "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
            "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for DRD2 antagonist or agonist development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited therapeutic potential. "
            "DRD2 is a key target for antipsychotic drugs and Parkinson's disease therapeutics, making this score valuable for CNS drug discovery."
        ),
    )
    def score_drd2_unidock(self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate the normalized docking score for DRD2 using UniDock.

        Uses UniDock molecular docking to predict binding affinity to the dopamine D2 receptor.
        Raw scores in kcal/mol are normalized to 0-1 range where higher values indicate stronger binding.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._score_unidock(samples, self.protein_targets["DRD2"])

    @scorer(
        name="gsk3b_unidock",
        population_wise=False,
        description=(
            "GSK3B UniDock binding affinity score (value range: 0.0 to 1.0). "
            "This score represents the normalized binding affinity to glycogen synthase kinase-3 beta (GSK3B) calculated using UniDock molecular docking. "
            "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
            "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for GSK3B inhibitor development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited kinase inhibition potential. "
            "GSK3B is a critical target for diabetes, cancer, and neurological disorders, making this score important for multi-therapeutic drug discovery."
        ),
    )
    def score_gsk3b_unidock(self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate the normalized docking score for GSK3B using UniDock.

        Uses UniDock molecular docking to predict binding affinity to glycogen synthase kinase-3 beta.
        Raw scores in kcal/mol are normalized to 0-1 range where higher values indicate stronger binding.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._score_unidock(samples, self.protein_targets["GSK3B"])

    @scorer(
        name="jnk3_unidock",
        population_wise=False,
        description=(
            "JNK3 UniDock binding affinity score (value range: 0.0 to 1.0). "
            "This score represents the normalized binding affinity to c-Jun N-terminal kinase 3 (JNK3) calculated using UniDock molecular docking. "
            "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
            "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for JNK3 inhibitor development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited therapeutic potential. "
            "JNK3 is a key target for neurodegenerative diseases and inflammatory conditions, making this score valuable for neuroprotective drug discovery."
        ),
    )
    def score_jnk3_unidock(self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate the normalized docking score for JNK3 using UniDock.

        Uses UniDock molecular docking to predict binding affinity to c-Jun N-terminal kinase 3.
        Raw scores in kcal/mol are normalized to 0-1 range where higher values indicate stronger binding.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._score_unidock(samples, self.protein_targets["JNK3"])

    @scorer(
        name="brd4_unidock",
        population_wise=False,
        description=(
            "BRD4 UniDock binding affinity score (value range: 0.0 to 1.0). "
            "This score represents the normalized binding affinity to bromodomain-containing protein 4 (BRD4) calculated using UniDock molecular docking. "
            "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
            "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for BRD4 inhibitor development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited epigenetic modulation potential. "
            "BRD4 is a crucial target for cancer therapy and inflammatory diseases, making this score important for epigenetic drug discovery."
        ),
    )
    def score_brd4_unidock(self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate the normalized docking score for BRD4 using UniDock.

        Uses UniDock molecular docking to predict binding affinity to bromodomain-containing protein 4.
        Raw scores in kcal/mol are normalized to 0-1 range where higher values indicate stronger binding.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._score_unidock(samples, self.protein_targets["BRD4"])

    @scorer(
        name="mpro_unidock",
        population_wise=False,
        description=(
            "MPRO UniDock binding affinity score (value range: 0.0 to 1.0). "
            "This score represents the normalized binding affinity to SARS-CoV-2 main protease (MPRO) calculated using UniDock molecular docking. "
            "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
            "High scores (≥0.6) indicate good predicted binding affinity (≤-9.0 kcal/mol) suitable for COVID-19 antiviral development, while low scores (≤0.5) suggest insufficient binding (<-7.5 kcal/mol) with limited protease inhibition potential. "
            "MPRO is an essential target for coronavirus therapeutics, making this score critical for antiviral drug discovery."
        ),
    )
    def score_mpro_unidock(self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate the normalized docking score for MPRO using UniDock.

        Uses UniDock molecular docking to predict binding affinity to SARS-CoV-2 main protease.
        Raw scores in kcal/mol are normalized to 0-1 range where higher values indicate stronger binding.
        Also computes protein-ligand interaction scores for use by interaction-specific scorers.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._score_unidock(samples, self.protein_targets["MPRO"], compute_mpro_interactions=True)

    @scorer(
        name="mpro_his161_a",
        population_wise=False,
        description=(
            "MPRO HIS161/A hydrogen bond interaction score (value range: 0.0 or 1.0). "
            "This binary score indicates whether the docked molecule forms a hydrogen bond with the critical HIS161 residue in chain A of the SARS-CoV-2 main protease active site. "
            "The score is computed using PLIP (Protein-Ligand Interaction Profiler) analysis of the docked pose. "
            "A score of 1.0 indicates the presence of a hydrogen bond interaction with HIS161/A, which is essential for MPRO inhibition, while a score of 0.0 indicates no such interaction. "
            "HIS161 is part of the catalytic dyad and hydrogen bonding with this residue is crucial for effective protease inhibition and antiviral activity. This interaction is very common in known MPRO binders and represents a key pharmacophore feature."
        ),
    )
    def score_mpro_his161_a(self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate MPRO HIS161/A hydrogen bond interaction score.

        Analyzes docked poses using PLIP to identify hydrogen bond interactions with HIS161 residue.
        Returns binary scores indicating presence (1.0) or absence (0.0) of this critical interaction.
        If MPRO docking hasn't been performed yet, it will be computed automatically.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        # Check if we need to compute MPRO docking first
        needs_docking = any(smiles not in self._mpro_interactions for smiles in samples)
        if needs_docking:
            self._score_unidock(samples, self.protein_targets["MPRO"], compute_mpro_interactions=True)

        results = []
        for smiles in samples:
            if smiles in self._mpro_interactions:
                results.append(float(self._mpro_interactions[smiles]["HIS161_A"]))
            else:
                results.append(None)
        return results

    @scorer(
        name="mpro_glu164_a",
        population_wise=False,
        description=(
            "MPRO GLU164/A hydrogen bond interaction score (value range: 0.0 or 1.0). "
            "This binary score indicates whether the docked molecule forms a hydrogen bond with the GLU164 residue in chain A of the SARS-CoV-2 main protease active site. "
            "The score is computed using PLIP (Protein-Ligand Interaction Profiler) analysis of the docked pose. "
            "A score of 1.0 indicates the presence of a hydrogen bond interaction with GLU164/A, which stabilizes inhibitor binding in the active site, while a score of 0.0 indicates no such interaction. "
            "GLU164 is located in the S1 binding pocket and interactions with this residue enhance binding selectivity and inhibitor potency against MPRO. This interaction is very common in known MPRO binders and is considered a critical binding motif."
        ),
    )
    def score_mpro_glu164_a(self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate MPRO GLU164/A hydrogen bond interaction score.

        Analyzes docked poses using PLIP to identify hydrogen bond interactions with GLU164 residue.
        Returns binary scores indicating presence (1.0) or absence (0.0) of this critical interaction.
        If MPRO docking hasn't been performed yet, it will be computed automatically.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        # Check if we need to compute MPRO docking first
        needs_docking = any(smiles not in self._mpro_interactions for smiles in samples)
        if needs_docking:
            self._score_unidock(samples, self.protein_targets["MPRO"], compute_mpro_interactions=True)

        results = []
        for smiles in samples:
            if smiles in self._mpro_interactions:
                results.append(float(self._mpro_interactions[smiles]["GLU164_A"]))
            else:
                results.append(None)
        return results

    @scorer(
        name="mpro_his39_a",
        population_wise=False,
        description=(
            "MPRO HIS39/A pi-stacking interaction score (value range: 0.0 or 1.0). "
            "This binary score indicates whether the docked molecule forms a pi-stacking interaction with the HIS39 residue in chain A of the SARS-CoV-2 main protease active site. "
            "The score is computed using PLIP (Protein-Ligand Interaction Profiler) analysis of the docked pose. "
            "A score of 1.0 indicates the presence of a pi-stacking interaction with HIS39/A, which provides additional binding affinity and selectivity, while a score of 0.0 indicates no such interaction. "
            "HIS39 is positioned in the active site and pi-stacking interactions with aromatic inhibitors can significantly enhance binding affinity and contribute to antiviral potency."
        ),
    )
    def score_mpro_his39_a(self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate MPRO HIS39/A pi-stacking interaction score.

        Analyzes docked poses using PLIP to identify pi-stacking interactions with HIS39 residue.
        Returns binary scores indicating presence (1.0) or absence (0.0) of this critical interaction.
        If MPRO docking hasn't been performed yet, it will be computed automatically.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        # Check if we need to compute MPRO docking first
        needs_docking = any(smiles not in self._mpro_interactions for smiles in samples)
        if needs_docking:
            self._score_unidock(samples, self.protein_targets["MPRO"], compute_mpro_interactions=True)

        results = []
        for smiles in samples:
            if smiles in self._mpro_interactions:
                results.append(float(self._mpro_interactions[smiles]["HIS39_A"]))
            else:
                results.append(None)
        return results

    @scorer(
        name="ampcclean_unidock",
        population_wise=False,
        description=(
            "AMPC UniDock binding affinity score (value range: 0.0 to 1.0). "
            "This score represents the normalized binding affinity to AmpC beta-lactamase calculated using UniDock molecular docking. "
            "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
            "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for beta-lactamase inhibitor development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited antibiotic resistance reversal potential. "
            "AmpC is a key target for combating antibiotic resistance, making this score valuable for developing adjuvant therapies to restore beta-lactam antibiotic efficacy."
        ),
    )
    def score_ampcclean_unidock(self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate the normalized docking score for AMPC using UniDock.

        Uses UniDock molecular docking to predict binding affinity to AmpC beta-lactamase.
        Raw scores in kcal/mol are normalized to 0-1 range where higher values indicate stronger binding.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._score_unidock(samples, self.protein_targets["ampc_clean"])

    @scorer(
        name="muopioidclean_unidock",
        population_wise=False,
        description=(
            "Mu-opioid receptor UniDock binding affinity score (value range: 0.0 to 1.0). "
            "This score represents the normalized binding affinity to the mu-opioid receptor (MOR) calculated using UniDock molecular docking. "
            "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
            "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for opioid analgesic development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited pain management potential. "
            "The mu-opioid receptor is the primary target for opioid analgesics, making this score important for pain management drug discovery while considering addiction potential."
        ),
    )
    def score_muopioidclean_unidock(self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate the normalized docking score for mu-opioid receptor using UniDock.

        Uses UniDock molecular docking to predict binding affinity to the mu-opioid receptor.
        Raw scores in kcal/mol are normalized to 0-1 range where higher values indicate stronger binding.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._score_unidock(samples, self.protein_targets["mu_opioid_clean"])
