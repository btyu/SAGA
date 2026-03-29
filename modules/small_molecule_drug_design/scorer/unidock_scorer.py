# Note: This is just an example of implementation. It may not be the best way to calculate the objective score.
from scileo_agent.core.registry import register_scorer_class, register_scorer
from scileo_agent.core.data_models import Candidate
from typing import List, Optional
from rdkit import Chem
from dataclasses import dataclass
from typing import Tuple
from modules.small_molecule_drug_design.docking import unidock, pose
import numpy as np
import json

# Get important paths
import os
import glob
import uuid
import pandas as pd

module_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(module_dir)  # modules/small_molecule_drug_design
grand_dir = os.path.dirname(base_dir)  # modules/
grandgrand_dir = os.path.dirname(grand_dir)  #SciLeoAgent
data_pdb_dir = os.path.join(base_dir, "data", "pdb")
docked_poses_dir = os.path.join(base_dir, "data", "docked_poses_temp")


@dataclass
class ProteinTarget:
    protein_name: str
    pdb_path: str
    pocket_center: Tuple[float, float, float]


@register_scorer_class  # <- Apply this decorator to your class
class UniDockScorers:
    """Collection of unidock scoring functions."""

    def __init__(
            self):  # <- Note: The init function must not have any arguments
        self.protein_targets = {
            "DRD2":
            ProteinTarget(
                protein_name="DRD2",
                pdb_path=os.path.join(data_pdb_dir, "DRD2.pdb"),
                pocket_center=(9.925, 5.846, -9.582),
            ),
            "GSK3B":
            ProteinTarget(
                protein_name="GSK3B",
                pdb_path=os.path.join(data_pdb_dir, "GSK3B.pdb"),
                pocket_center=(-14.782, -17.079, -3.559),
            ),
            "JNK3":
            ProteinTarget(
                protein_name="JNK3",
                pdb_path=os.path.join(data_pdb_dir, "JNK3.pdb"),
                pocket_center=(23.167, 8.921, 31.848),
            ),
            "BRD4":
            ProteinTarget(
                protein_name="BRD4",
                pdb_path=os.path.join(data_pdb_dir, "BRD4.pdb"),
                pocket_center=(28.751, 15.826, -2.335),
            ),
            "MPRO":
            ProteinTarget(
                protein_name="MPRO",
                pdb_path=os.path.join(data_pdb_dir, "MPRO.pdb"),
                pocket_center=(9.050, 8.898, -1.508),
            ),
            "MARS1":
            ProteinTarget(
                protein_name="sarsmars1",
                pdb_path=base_dir +
                "sars_mers_combined/mers_test_1_protein.pdb",
                pocket_center=(7.813, -0.9809999999999999, 22.566),
            ),
            "mu_opioid_clean":
            ProteinTarget(
                protein_name="mu_opioid_clean",
                pdb_path=os.path.join(data_pdb_dir, "mu_opioid_clean.pdb"),
                pocket_center=(1.2857, 16.4479, -59.1143),
            ),
            "ampc_clean":
            ProteinTarget(
                protein_name="ampc_clean",
                pdb_path=os.path.join(data_pdb_dir, "ampc_clean.pdb"),
                pocket_center=(23.5567, 5.7351, 14.3991),
            )
        }
        self.seed = 42
        # make sure docked_poses_dir has a unique name
        self.docked_poses_dir = os.path.join(docked_poses_dir,
                                             str(uuid.uuid4()))
        self.pocket_distance_cutoff = 12.0

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

    def _score_unidock(self, candidates: List[Candidate],
                       protein_target: ProteinTarget) -> List[float]:
        valid_mols = []
        valid_indices = []

        for i, candidate in enumerate(candidates):
            smiles = candidate.representation  # type: ignore[attr-defined]
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                continue
            valid_mol = Chem.MolFromSmiles(smiles)
            if valid_mol is None:
                continue
            valid_mols.append(valid_mol)
            valid_indices.append(i)

        valid_tuples = unidock.docking(valid_mols,
                                       protein_target.pdb_path,
                                       protein_target.pocket_center,
                                       seed=self.seed,
                                       output_path=self.docked_poses_dir)
        valid_scores = [valid_tuple[1] for valid_tuple in valid_tuples]

        results = [0.0] * len(candidates)
        for valid_idx, valid_score in zip(valid_indices, valid_scores):
            results[valid_idx] = self._normalize_score(valid_score)

        # Add docking residue map to candidate metadata
        # Add interaction score to candidate properties

        # Keep track of the sdf index as only valid indices are docked
        sdf_idx = 0
        for i, candidate in enumerate(candidates):
            if i in valid_indices:
                sdf_filepath = os.path.join(self.docked_poses_dir,
                                            f"docked_{sdf_idx}.sdf")
                sdf_idx += 1

                if not os.path.exists(sdf_filepath):
                    candidate.set_property("HIS161_A", 0)
                    candidate.set_property("GLU164_A", 0)
                    candidate.set_property("HIS39_A", 0)
                    continue


#                     raise FileNotFoundError(f"Docked pose file {sdf_filepath} not found")

                ligand_coords = pose.get_ligand_coords_from_sdf(sdf_filepath)
                docking_residue_map = pose.generate_residue_map(
                    protein_target.pdb_path,
                    ligand_coords,
                    distance_cutoff=self.pocket_distance_cutoff)
                candidate.metadata["docking_residue_map"] = json.dumps(
                    docking_residue_map)

                # Compute the interaction score for MPRO
                if protein_target.protein_name == "MPRO":
                    try:
                        interaction_score = unidock.mpro_plip_score_computation(
                            sdf_filepath)
                        candidate.set_property(
                            "HIS161_A",
                            interaction_score["hydrogen_bond/HIS/161/A"])
                        candidate.set_property(
                            "GLU164_A",
                            interaction_score["hydrogen_bond/GLU/164/A"])
                        candidate.set_property(
                            "HIS39_A", interaction_score["pi_stack/HIS/39/A"])
                    except:
                        candidate.set_property("HIS161_A", 0)
                        candidate.set_property("GLU164_A", 0)
                        candidate.set_property("HIS39_A", 0)
            else:
                candidate.set_property("HIS161_A", 0)
                candidate.set_property("GLU164_A", 0)
                candidate.set_property("HIS39_A", 0)
        os.system(f"rm -r {self.docked_poses_dir}")
        return results

    def _score_unidock_nodelete(self, candidates: List[Candidate],
                       protein_target: ProteinTarget) -> List[float]:
        valid_mols = []
        valid_indices = []

        for i, candidate in enumerate(candidates):
            smiles = candidate.representation  # type: ignore[attr-defined]
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                continue
            valid_mol = Chem.MolFromSmiles(smiles)
            if valid_mol is None:
                continue
            valid_mols.append(valid_mol)
            valid_indices.append(i)

        valid_tuples = unidock.docking(valid_mols,
                                       protein_target.pdb_path,
                                       protein_target.pocket_center,
                                       seed=self.seed,
                                       output_path=self.docked_poses_dir)
        valid_scores = [valid_tuple[1] for valid_tuple in valid_tuples]

        results = [0.0] * len(candidates)
        for valid_idx, valid_score in zip(valid_indices, valid_scores):
            results[valid_idx] = self._normalize_score(valid_score)

        # Add docking residue map to candidate metadata
        # Add interaction score to candidate properties

        # Keep track of the sdf index as only valid indices are docked
        sdf_idx = 0
        for i, candidate in enumerate(candidates):
            if i in valid_indices:
                sdf_filepath = os.path.join(self.docked_poses_dir,
                                            f"docked_{sdf_idx}.sdf")
                sdf_idx += 1

                if not os.path.exists(sdf_filepath):
                    candidate.set_property("HIS161_A", 0)
                    candidate.set_property("GLU164_A", 0)
                    candidate.set_property("HIS39_A", 0)
                    continue


#                     raise FileNotFoundError(f"Docked pose file {sdf_filepath} not found")

                ligand_coords = pose.get_ligand_coords_from_sdf(sdf_filepath)
                docking_residue_map = pose.generate_residue_map(
                    protein_target.pdb_path,
                    ligand_coords,
                    distance_cutoff=self.pocket_distance_cutoff)
                candidate.metadata["docking_residue_map"] = json.dumps(
                    docking_residue_map)

                # Compute the interaction score for MPRO
                if protein_target.protein_name == "MPRO":
                    try:
                        interaction_score = unidock.mpro_plip_score_computation(
                            sdf_filepath)
                        candidate.set_property(
                            "HIS161_A",
                            interaction_score["hydrogen_bond/HIS/161/A"])
                        candidate.set_property(
                            "GLU164_A",
                            interaction_score["hydrogen_bond/GLU/164/A"])
                        candidate.set_property(
                            "HIS39_A", interaction_score["pi_stack/HIS/39/A"])
                    except:
                        candidate.set_property("HIS161_A", 0)
                        candidate.set_property("GLU164_A", 0)
                        candidate.set_property("HIS39_A", 0)
            else:
                candidate.set_property("HIS161_A", 0)
                candidate.set_property("GLU164_A", 0)
                candidate.set_property("HIS39_A", 0)
        return results

    def _score_unidock_unnorm(
            self, candidates: List[Candidate],
            protein_target: ProteinTarget) -> List[float]:
        valid_mols = []
        valid_indices = []

        for i, candidate in enumerate(candidates):
            smiles = candidate.representation  # type: ignore[attr-defined]
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                continue
            valid_mol = Chem.MolFromSmiles(smiles)
            if valid_mol is None:
                continue
            valid_mols.append(valid_mol)
            valid_indices.append(i)

        valid_tuples = unidock.docking(valid_mols,
                                       protein_target.pdb_path,
                                       protein_target.pocket_center,
                                       seed=self.seed)
        valid_scores = [valid_tuple[1] for valid_tuple in valid_tuples]

        results = [0.0] * len(candidates)
        for valid_idx, valid_score in zip(valid_indices, valid_scores):
            results[valid_idx] = valid_score

        return results

    @register_scorer(  # <- Apply this decorator to the scoring functions in this class
        name="drd2_unidock",
        population_wise=False,
        description=
        "DRD2 UniDock binding affinity score (value range: 0.0 to 1.0). "
        "This score represents the normalized binding affinity to the dopamine D2 receptor (DRD2) calculated using UniDock molecular docking. "
        "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
        "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for DRD2 antagonist or agonist development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited therapeutic potential. "
        "DRD2 is a key target for antipsychotic drugs and Parkinson's disease therapeutics, making this score valuable for CNS drug discovery.",
    )
    def score_drd2_unidock(self,
                           candidates: List[Candidate]) -> Optional[float]:
        """Calculate the normalized docking score for DRD2."""
        return self._score_unidock(candidates, self.protein_targets["DRD2"])

    @register_scorer(
        name="gsk3b_unidock",
        population_wise=False,
        description=
        "GSK3B UniDock binding affinity score (value range: 0.0 to 1.0). "
        "This score represents the normalized binding affinity to glycogen synthase kinase-3 beta (GSK3B) calculated using UniDock molecular docking. "
        "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
        "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for GSK3B inhibitor development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited kinase inhibition potential. "
        "GSK3B is a critical target for diabetes, cancer, and neurological disorders, making this score important for multi-therapeutic drug discovery.",
    )
    def score_gsk3b_unidock(self,
                            candidates: List[Candidate]) -> Optional[float]:
        """Calculate the normalized docking score for GSK3B."""
        return self._score_unidock(candidates, self.protein_targets["GSK3B"])

    @register_scorer(
        name="jnk3_unidock",
        population_wise=False,
        description=
        "JNK3 UniDock binding affinity score (value range: 0.0 to 1.0). "
        "This score represents the normalized binding affinity to c-Jun N-terminal kinase 3 (JNK3) calculated using UniDock molecular docking. "
        "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
        "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for JNK3 inhibitor development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited therapeutic potential. "
        "JNK3 is a key target for neurodegenerative diseases and inflammatory conditions, making this score valuable for neuroprotective drug discovery.",
    )
    def score_jnk3_unidock(self,
                           candidates: List[Candidate]) -> Optional[float]:
        """Calculate the normalized docking score for JNK3."""
        return self._score_unidock(candidates, self.protein_targets["JNK3"])

    @register_scorer(
        name="brd4_unidock",
        population_wise=False,
        description=
        "BRD4 UniDock binding affinity score (value range: 0.0 to 1.0). "
        "This score represents the normalized binding affinity to bromodomain-containing protein 4 (BRD4) calculated using UniDock molecular docking. "
        "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
        "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for BRD4 inhibitor development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited epigenetic modulation potential. "
        "BRD4 is a crucial target for cancer therapy and inflammatory diseases, making this score important for epigenetic drug discovery.",
    )
    def score_brd4_unidock(self,
                           candidates: List[Candidate]) -> Optional[float]:
        """Calculate the normalized docking score for BRD4."""
        return self._score_unidock(candidates, self.protein_targets["BRD4"])

    @register_scorer(
        name="mpro_unidock",
        population_wise=False,
        description=
        "MPRO UniDock binding affinity score (value range: 0.0 to 1.0). "
        "This score represents the normalized binding affinity to SARS-CoV-2 main protease (MPRO) calculated using UniDock molecular docking. "
        "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
        "High scores (≥0.6) indicate good predicted binding affinity (≤-9.0 kcal/mol) suitable for COVID-19 antiviral development, while low scores (≤0.5) suggest insufficient binding (<-7.5 kcal/mol) with limited protease inhibition potential. "
        "MPRO is an essential target for coronavirus therapeutics, making this score critical for antiviral drug discovery.",
    )
    def score_mpro_unidock(self,
                           candidates: List[Candidate]) -> Optional[float]:
        return self._score_unidock(candidates, self.protein_targets["MPRO"])

    @register_scorer(
        name="mpro_his161_a",
        population_wise=False,
        description=
        "MPRO HIS161/A hydrogen bond interaction score (value range: 0.0 or 1.0). "
        "This binary score indicates whether the docked molecule forms a hydrogen bond with the critical HIS161 residue in chain A of the SARS-CoV-2 main protease active site. "
        "The score is computed using PLIP (Protein-Ligand Interaction Profiler) analysis of the docked pose. "
        "A score of 1.0 indicates the presence of a hydrogen bond interaction with HIS161/A, which is essential for MPRO inhibition, while a score of 0.0 indicates no such interaction. "
        "HIS161 is part of the catalytic dyad and hydrogen bonding with this residue is crucial for effective protease inhibition and antiviral activity. This interaction is very common in known MPRO binders and represents a key pharmacophore feature.",
    )
    def score_mpro_his161_a(self,
                            candidates: List[Candidate]) -> Optional[float]:
        if "HIS161_A" not in candidates[0].properties:
            # Dock the candidates
            self._score_unidock(candidates, self.protein_targets["MPRO"])
            assert "HIS161_A" in candidates[
                0].properties, "HIS161_A not found in candidates after docking"

        return [candidate.properties["HIS161_A"] for candidate in candidates]

    @register_scorer(
        name="mpro_glu164_a",
        population_wise=False,
        description=
        "MPRO GLU164/A hydrogen bond interaction score (value range: 0.0 or 1.0). "
        "This binary score indicates whether the docked molecule forms a hydrogen bond with the GLU164 residue in chain A of the SARS-CoV-2 main protease active site. "
        "The score is computed using PLIP (Protein-Ligand Interaction Profiler) analysis of the docked pose. "
        "A score of 1.0 indicates the presence of a hydrogen bond interaction with GLU164/A, which stabilizes inhibitor binding in the active site, while a score of 0.0 indicates no such interaction. "
        "GLU164 is located in the S1 binding pocket and interactions with this residue enhance binding selectivity and inhibitor potency against MPRO. This interaction is very common in known MPRO binders and is considered a critical binding motif.",
    )
    def score_mpro_glu164_a(self,
                            candidates: List[Candidate]) -> Optional[float]:
        if "GLU164_A" not in candidates[0].properties:
            # Dock the candidates
            self._score_unidock(candidates, self.protein_targets["MPRO"])
            assert "GLU164_A" in candidates[
                0].properties, "GLU164_A not found in candidates after docking"

        return [candidate.properties["GLU164_A"] for candidate in candidates]

    @register_scorer(
        name="mpro_his39_a",
        population_wise=False,
        description=
        "MPRO HIS39/A pi-stacking interaction score (value range: 0.0 or 1.0). "
        "This binary score indicates whether the docked molecule forms a pi-stacking interaction with the HIS39 residue in chain A of the SARS-CoV-2 main protease active site. "
        "The score is computed using PLIP (Protein-Ligand Interaction Profiler) analysis of the docked pose. "
        "A score of 1.0 indicates the presence of a pi-stacking interaction with HIS39/A, which provides additional binding affinity and selectivity, while a score of 0.0 indicates no such interaction. "
        "HIS39 is positioned in the active site and pi-stacking interactions with aromatic inhibitors can significantly enhance binding affinity and contribute to antiviral potency.",
    )
    def score_mpro_his39_a(self,
                           candidates: List[Candidate]) -> Optional[float]:
        if "HIS39_A" not in candidates[0].properties:
            # Dock the candidates
            self._score_unidock(candidates, self.protein_targets["MPRO"])
            assert "HIS39_A" in candidates[
                0].properties, "HIS39_A not found in candidates after docking"

        return [candidate.properties["HIS39_A"] for candidate in candidates]

    @register_scorer(
        name="mars1_unidock",
        population_wise=False,
        description=
        "MARS1 UniDock binding affinity score (value range: 0.0 to 1.0). "
        "This score represents the normalized binding affinity to MARS1 (SARS/MERS combined target) calculated using UniDock molecular docking. "
        "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
        "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for broad-spectrum coronavirus antiviral development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited therapeutic potential. "
        "MARS1 represents a combined target for developing pan-coronavirus therapeutics against multiple viral strains.",
    )
    def score_mars1_unidock(self,
                            candidates: List[Candidate]) -> Optional[float]:
        return self._score_unidock(candidates, self.protein_targets["MARS1"])

    @register_scorer(
        name="ampcclean_unidock",
        population_wise=False,
        description=
        "AMPC UniDock binding affinity score (value range: 0.0 to 1.0). "
        "This score represents the normalized binding affinity to AmpC beta-lactamase calculated using UniDock molecular docking. "
        "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
        "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for beta-lactamase inhibitor development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited antibiotic resistance reversal potential. "
        "AmpC is a key target for combating antibiotic resistance, making this score valuable for developing adjuvant therapies to restore beta-lactam antibiotic efficacy.",
    )
    def score_ampcclean_unidock(
            self, candidates: List[Candidate]) -> Optional[float]:
        return self._score_unidock(candidates,
                                   self.protein_targets["ampc_clean"])

    @register_scorer(
        name="muopioidclean_unidock",
        population_wise=False,
        description=
        "Mu-opioid receptor UniDock binding affinity score (value range: 0.0 to 1.0). "
        "This score represents the normalized binding affinity to the mu-opioid receptor (MOR) calculated using UniDock molecular docking. "
        "The raw UniDock scores (in kcal/mol, where more negative values indicate stronger binding) are normalized by dividing by -15.0 and clipping to 0-1 range. "
        "High scores (>0.7) indicate strong predicted binding affinity (≤-10.5 kcal/mol) suitable for opioid analgesic development, while low scores (<0.3) suggest weak binding (>-4.5 kcal/mol) with limited pain management potential. "
        "The mu-opioid receptor is the primary target for opioid analgesics, making this score important for pain management drug discovery while considering addiction potential."
    )
    def score_muopioidclean_unidock(
            self, candidates: List[Candidate]) -> Optional[float]:
        return self._score_unidock(candidates,
                                   self.protein_targets["mu_opioid_clean"])

if __name__ == "__main__":

    # Test to see if the score is registered correctly
    from scileo_agent.core.registry import get_scorer, list_scorers
    from scileo_agent.core.data_models import Objective, Candidate

    # List all available scorers
    print("Available scorers:\n", list_scorers())

    # Get specific scorers
    drd2_unidock_scorer = get_scorer("drd2_unidock")
    gsk3b_unidock_scorer = get_scorer("gsk3b_unidock")
    jnk3_unidock_scorer = get_scorer("jnk3_unidock")
    brd4_unidock_scorer = get_scorer("brd4_unidock")
    mpro_unidock_scorer = get_scorer("mpro_unidock")
    mpro_his161_a_scorer = get_scorer("mpro_his161_a")
    mpro_glu164_a_scorer = get_scorer("mpro_glu164_a")
    mpro_his39_a_scorer = get_scorer("mpro_his39_a")

    # Test the scorers
    candidates = [
        Candidate(representation="CN(Cc1cccc(F)c1)C(=O)Cn1nnc2ccccc21"),
        Candidate(representation="O=C(Cn1nnc2ccccc21)N1CC=CCC[C@@H]1c1ccccc1"),
        Candidate(
            representation="CCN(C(=O)Cn1nnc2ccccc21)C(c1ccccc1)c1ccccc1"),
        Candidate(
            representation="C[C@@H](c1ccccc1)N(C(=O)Cn1nnc2ccccc21)C1CC1"),
    ]
    # print(drd2_unidock_scorer(candidates))
    # print(gsk3b_unidock_scorer(candidates))
    # print(jnk3_unidock_scorer(candidates))
    # print(brd4_unidock_scorer(candidates))
    print(mpro_unidock_scorer(candidates))
    print(mpro_his161_a_scorer(candidates))
    print(mpro_glu164_a_scorer(candidates))
    print(mpro_his39_a_scorer(candidates))
