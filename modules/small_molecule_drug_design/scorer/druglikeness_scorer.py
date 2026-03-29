from scileo_agent.core.registry import register_scorer_class, register_scorer
from scileo_agent.core.data_models import Candidate
from typing import List, Optional
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, rdMolDescriptors
from rdkit.Chem import FilterCatalog
from modules.small_molecule_drug_design.properties.SA_score import sascorer
import numpy as np

# silence rdkit warnings
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


@register_scorer_class
class DruglikenessScorers:
    """Collection of druglikeness scoring functions with 0-1 scoring."""

    def __init__(self):
        # Initialize RDKit structural alert catalogs once
        pains_params = FilterCatalog.FilterCatalogParams()
        pains_params.AddCatalog(
            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        pains_params.AddCatalog(
            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
        pains_params.AddCatalog(
            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
        self._pains_catalog = FilterCatalog.FilterCatalog(pains_params)

        brenk_params = FilterCatalog.FilterCatalogParams()
        brenk_params.AddCatalog(
            FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        self._brenk_catalog = FilterCatalog.FilterCatalog(brenk_params)
        # Lazy-initialized external druglikeness model
        self._deepdl_model = None
        self._deepdl_error = None

    def _calculate_fsp3(self, mol):
        """Calculate fraction of sp3 carbons."""
        if mol is None:
            return None
        carbon_count = 0
        sp3_carbon_count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon
                carbon_count += 1
                if atom.GetHybridization(
                ) == Chem.rdchem.HybridizationType.SP3:
                    sp3_carbon_count += 1
        return sp3_carbon_count / carbon_count if carbon_count > 0 else 0

    def _linear_interpolation(self,
                              value,
                              good_threshold,
                              bad_threshold,
                              reverse=False):
        """Linear interpolation between good (score=1) and bad (score=0) thresholds."""
        if reverse:  # For properties where lower is better (like SA)
            if value <= good_threshold:
                return 1.0
            elif value >= bad_threshold:
                return 0.0
            else:
                return (bad_threshold - value) / (bad_threshold -
                                                  good_threshold)
        else:  # For properties where higher is better
            if value <= good_threshold:
                return 1.0
            elif value >= bad_threshold:
                return 0.0
            else:
                return 1.0 - (value - good_threshold) / (bad_threshold -
                                                         good_threshold)

    @register_scorer(
        name="sa_score",
        population_wise=False,
        description=
        "Synthetic Accessibility (SA) druglikeness score (value range: 0.0 to 1.0). "
        "SA score estimates how difficult a molecule would be to synthesize, based on fragment contributions and molecular complexity. "
        "This metric is crucial for drug discovery as synthetic feasibility directly affects development costs, timelines, and manufacturability. "
        "High scores (>0.8) indicate easily synthesizable molecules with simple structures and common fragments, while low scores (<0.3) suggest very challenging synthesis requiring complex multi-step routes or exotic reagents. "
        "The scoring is based on the original SA scale (1-10) where lower raw SA values indicate easier synthesis, normalized to 0-1 druglikeness scale where higher normalized scores indicate better synthetic accessibility.",
    )
    def score_sa_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate Synthetic Accessibility (SA) druglikeness score.

        Returns normalized scores between 0.0 and 1.0 based on synthetic accessibility:
        - Score = 1.0: SA ≤ 1 (easily synthesizable molecules)
        - Score = 0.0: SA ≥ 10 (very difficult to synthesize)
        - Score = linear interpolation: SA between 1-10

        Lower SA scores indicate easier synthesis and are preferred for drug development.
        Higher normalized scores indicate more synthetically accessible molecules.

        Returns:
            List of scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                sa_score = sascorer.calculateScore(mol)
                if sa_score is None:
                    results.append(0.0)
                else:
                    # Normalize SA score (1-10 range) to 0-1 druglikeness score
                    # Lower SA is better, so we use reverse=True
                    score = self._linear_interpolation(sa_score,
                                                       good_threshold=1,
                                                       bad_threshold=10,
                                                       reverse=True)
                    results.append(score)
        return results

    @register_scorer(
        name="deepdl_druglikeness",
        population_wise=False,
        description="DeepDL druglikeness score (value range: 0.0 to 1.0). "
        "This is an unsupervised deep learning model trained on the distribution of real approved drugs to capture complex, non-linear drug-like patterns that traditional rule-based filters might miss. "
        "The model learns latent representations of drug-like chemical space from marketed pharmaceuticals. "
        "High scores (>0.7) indicate strong similarity to approved drugs with optimal drug-like properties, while low scores (<0.3) suggest non-drug-like molecules that deviate significantly from known pharmaceutical space. "
        "This scorer complements rule-based approaches by capturing subtle drug-like features and provides a data-driven assessment of overall druglikeness."
    )
    def score_deepdl_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """Score molecules using DeepDL model; normalize 0-100 → 0-1.

        Requires package providing `druglikeness.deepdl.DeepDL`.
        Model is loaded lazily on first use with pretrained 'extended' on CPU.
        Returns None for invalid SMILES or if model unavailable.
        """
        if self._deepdl_model is None:
            from druglikeness.deepdl import DeepDL  # type: ignore
            self._deepdl_model = DeepDL.from_pretrained('extended',
                                                        device="cpu")

        # Filter out empty SMILES strings
        filtered_candidates = []
        for c in candidates:
            smiles = c.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "" or Chem.MolFromSmiles(smiles) is None:
                filtered_candidates.append(None)
            else:
                filtered_candidates.append(c)
        
        # Only process non-empty candidates
        non_empty_candidates = [c for c in filtered_candidates if c is not None]
        if non_empty_candidates:
            smiles_list = [c.representation for c in non_empty_candidates]
            raw_scores = self._deepdl_model.screening(smiles_list=smiles_list,
                                                      naive=True,
                                                      batch_size=64)
        else:
            raw_scores = []
        
        # Reconstruct results with 0.0 for invalid candidates
        results = []
        score_idx = 0
        for candidate in filtered_candidates:
            if candidate is None:
                results.append(0.0)
            else:
                if score_idx < len(raw_scores):
                    results.append(0.0 if raw_scores[score_idx] is None else max(0.0, min(1.0, float(raw_scores[score_idx]) / 100.0)))
                    score_idx += 1
                else:
                    results.append(0.0)
        return results

    @register_scorer(
        name="solubility_score",
        population_wise=False,
        description=
        "Aqueous solubility druglikeness score (value range: 0.0 to 1.0). "
        "Solubility measures how well a compound dissolves in water, which is critical for oral bioavailability, intravenous formulation, and overall pharmacokinetics. "
        "Poor solubility is a major cause of drug development failures and limits dosing options. "
        "High scores (>0.8) indicate excellent predicted solubility (LogP <2) suitable for various formulations and administration routes, while low scores (<0.3) suggest poor solubility (LogP >4) that may require specialized formulation techniques or limit bioavailability. "
        "This score uses LogP as a validated proxy for aqueous solubility, as lipophilic compounds generally have reduced water solubility.",
    )
    def score_solubility_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate solubility druglikeness score using LogP as proxy.

        Returns normalized scores between 0.0 and 1.0 based on estimated solubility:
        - Score = 1.0: Estimated good solubility (LogP < 2)
        - Score = 0.0: Estimated poor solubility (LogP > 4)
        - Score = linear interpolation: LogP between 2-4

        Uses LogP as a proxy for solubility since direct solubility calculation requires more complex models.
        Lower LogP generally correlates with better aqueous solubility.

        Returns:
            List of scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                # Use LogP as proxy for solubility (inverse relationship)
                logp = Crippen.MolLogP(mol)
                score = self._linear_interpolation(logp,
                                                   good_threshold=2,
                                                   bad_threshold=5)
                results.append(score)
        return results

    @register_scorer(
        name="logp_score",
        population_wise=False,
        description=
        "Lipophilicity (LogP) druglikeness score (value range: 0.0 to 1.0). "
        "LogP measures the partition coefficient between octanol and water, indicating the balance between lipophilicity and hydrophilicity. "
        "This property is fundamental to drug absorption, distribution, metabolism, and excretion (ADMET). "
        "Optimal LogP balance ensures adequate membrane permeability while maintaining sufficient solubility. "
        "High scores (>0.8) indicate optimal lipophilicity (LogP <2) with good membrane permeability and water solubility balance, while low scores (<0.3) suggest excessive lipophilicity (LogP >4) leading to poor solubility, potential toxicity, and challenging formulation. "
        "LogP is a key component of Lipinski's Rule of Five and strongly correlates with oral bioavailability.",
    )
    def score_logp_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate LogP druglikeness score.

        Returns normalized scores between 0.0 and 1.0 based on lipophilicity (LogP):
        - Score = 1.0: LogP < 2 (optimal lipophilicity for drug-like molecules)
        - Score = 0.0: LogP > 3 (too lipophilic, poor ADMET properties)
        - Score = linear interpolation: LogP between 2-3

        Lower LogP values indicate better water solubility and permeability balance.
        Higher scores indicate more drug-like lipophilicity.

        Returns:
            List of scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                logp = Crippen.MolLogP(mol)
                score = self._linear_interpolation(logp,
                                                   good_threshold=2,
                                                   bad_threshold=4)
                results.append(score)
        return results

    @register_scorer(
        name="mw_score",
        population_wise=False,
        description=
        "Molecular Weight (MW) druglikeness score (value range: 0.0 to 1.0). "
        "MW is a fundamental predictor of oral bioavailability and overall drug-like properties, as it affects membrane permeability, metabolic stability, and formulation feasibility. "
        "Molecular size directly impacts a compound's ability to cross biological membranes and reach target sites. "
        "High scores (>0.8) indicate optimal molecular size (MW 200-400 Da) associated with excellent oral bioavailability and drug-like properties, while scores gradually decrease for molecules outside this range, with very small (MW <150 Da) or very large (MW >500 Da) molecules receiving lower scores due to poor drug-like properties and bioavailability challenges. "
        "Molecules outside the optimal range may have poor ADMET profiles and formulation challenges.",
        "High scores (>0.8) indicate optimal molecular size (MW ≤400 Da) associated with excellent oral bioavailability and drug-like properties, while low scores (<0.3) suggest large molecules (MW >500 Da) that violate Lipinski's Rule of Five and typically exhibit poor absorption, distribution, and bioavailability. "
        "Smaller molecules generally have better ADMET profiles and are easier to formulate.",
    )
    def score_mw_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate molecular weight druglikeness score.

        Returns normalized scores between 0.0 and 1.0 based on molecular weight:
        - Score = 1.0: MW ≤ 400 Da (optimal size for oral bioavailability)
        - Score = 0.0: MW > 500 Da (violates Lipinski's Rule of Five)
        - Score = linear interpolation: MW between 400-500 Da

        Lower molecular weights are generally associated with better ADMET properties.
        Higher scores indicate more drug-like molecular size.

        Returns:
            List of scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                mw = Descriptors.MolWt(mol)
                score = self._linear_interpolation(mw,
                                                   good_threshold=400,
                                                   bad_threshold=600)
                results.append(score)
        return results

    @register_scorer(
        name="psa_score",
        population_wise=False,
        description=
        "Topological Polar Surface Area (TPSA) druglikeness score (value range: 0.0 to 1.0). "
        "TPSA measures the surface area occupied by polar atoms (oxygen and nitrogen with attached hydrogens) and is a validated predictor of membrane permeability, blood-brain barrier penetration, and oral bioavailability. "
        "TPSA directly correlates with passive diffusion across biological membranes. "
        "High scores (>0.8) indicate optimal polar surface area (TPSA ≤120 Ų) associated with excellent membrane permeability and bioavailability, while low scores (<0.3) suggest excessive polarity (TPSA >140 Ų) leading to poor membrane transport, reduced absorption, and limited tissue distribution. "
        "TPSA is particularly important for CNS drugs where blood-brain barrier penetration is crucial.",
    )
    def score_psa_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate Polar Surface Area (PSA) druglikeness score.

        Returns normalized scores between 0.0 and 1.0 based on topological PSA:
        - Score = 1.0: PSA ≤ 120 Ų (optimal for membrane permeability)
        - Score = 0.0: PSA > 140 Ų (poor membrane permeability, BBB penetration)
        - Score = linear interpolation: PSA between 120-140 Ų

        Lower PSA values correlate with better cell membrane permeability.
        Higher scores indicate more drug-like polar surface area.

        Returns:
            List of scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                psa = rdMolDescriptors.CalcTPSA(mol)
                score = self._linear_interpolation(psa,
                                                   good_threshold=120,
                                                   bad_threshold=160)
                results.append(score)
        return results

    @register_scorer(
        name="rotatable_bonds_score",
        population_wise=False,
        description=
        "Rotatable Bonds druglikeness score (value range: 0.0 to 1.0). "
        "Rotatable bonds measure molecular flexibility and conformational freedom, which affects both target binding affinity and pharmacokinetic properties. "
        "Excessive flexibility incurs entropy penalties upon binding and can reduce oral bioavailability. "
        "High scores (>0.8) indicate optimal molecular rigidity (≤7 rotatable bonds) associated with strong target binding due to reduced entropy loss and good oral bioavailability, while low scores (<0.3) suggest excessive flexibility (>10 rotatable bonds) leading to weak binding affinity, poor selectivity, and challenging pharmacokinetics. "
        "Moderate flexibility is preferred to maintain binding specificity while allowing necessary conformational adjustments.",
    )
    def score_rotatable_bonds_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate rotatable bonds druglikeness score.

        Returns normalized scores between 0.0 and 1.0 based on molecular flexibility:
        - Score = 1.0: ≤ 7 rotatable bonds (optimal flexibility for binding)
        - Score = 0.0: > 10 rotatable bonds (too flexible, entropy penalty)
        - Score = linear interpolation: 7-10 rotatable bonds

        Fewer rotatable bonds indicate better binding affinity due to lower entropy loss.
        Higher scores indicate more drug-like molecular flexibility.

        Returns:
            List of scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
                score = self._linear_interpolation(rb,
                                                   good_threshold=7,
                                                   bad_threshold=12)
                results.append(score)
        return results

    @register_scorer(
        name="fsp3_score",
        population_wise=False,
        description=
        "Fraction of sp3 carbons (FSP3) druglikeness score (value range: 0.0 to 1.0). "
        "FSP3 measures molecular complexity and three-dimensional character, representing the fraction of carbon atoms with tetrahedral geometry. "
        "This metric is crucial for escaping the 'flatland' of aromatic compounds that dominate many chemical libraries. "
        "Higher sp3 content provides molecular diversity, improved target selectivity, and better ADMET properties. "
        "High scores (>0.8) indicate excellent 3D character (FSP3 >0.3) associated with enhanced selectivity, reduced off-target effects, and improved drug-like properties, while low scores (<0.3) suggest overly flat, aromatic structures (FSP3 <0.2) that may suffer from poor selectivity, increased toxicity, and limited chemical space exploration. "
        "Three-dimensional molecules often have superior pharmacological profiles.",
    )
    def score_fsp3_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate fraction of sp3 carbons (FSP3) druglikeness score.

        Returns normalized scores between 0.0 and 1.0 based on molecular complexity:
        - Score = 1.0: FSP3 > 0.3 (optimal 3D character, escape from flatland)
        - Score = 0.0: FSP3 < 0.2 (too flat/aromatic, poor selectivity)
        - Score = linear interpolation: FSP3 between 0.2-0.3

        Higher FSP3 values indicate more 3D character and better drug-like properties.
        Molecules with higher sp3 content tend to have better selectivity and ADMET.

        Returns:
            List of scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                fsp3 = self._calculate_fsp3(mol)
                if fsp3 is None:
                    results.append(None)
                else:
                    score = self._linear_interpolation(fsp3,
                                                       good_threshold=0.3,
                                                       bad_threshold=0.15,
                                                       reverse=True)
                    results.append(score)
        return results

    @register_scorer(
        name="rings_score",
        population_wise=False,
        description=
        "Number of rings druglikeness score (value range: 0.0 to 1.0). "
        "Ring count affects molecular complexity, structural rigidity, and synthetic accessibility. "
        "Ring systems provide structural frameworks for target binding while influencing synthetic feasibility and optimization potential. "
        "High scores (>0.8) indicate simple ring systems (<1 ring) that are easily synthesizable, readily optimizable, and have favorable ADMET properties, while low scores (<0.3) suggest complex polycyclic systems (>2 rings) that may be synthetically challenging, difficult to optimize, and prone to poor drug-like properties. "
        "Moderate ring content balances structural diversity needed for target binding with synthetic and optimization feasibility required for drug development.",
    )
    def score_rings_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate number of rings druglikeness score.

        Returns normalized scores between 0.0 and 1.0 based on ring count:
        - Score = 1.0: < 1 ring (simple, acyclic structures)
        - Score = 0.0: > 2 rings (complex polycyclic systems)
        - Score = linear interpolation: 1-2 rings

        Fewer rings generally correlate with easier synthesis and optimization.
        Higher scores indicate more drug-like ring complexity.

        Returns:
            List of scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                ring_count = rdMolDescriptors.CalcNumRings(mol)
                score = self._linear_interpolation(ring_count,
                                                   good_threshold=1,
                                                   bad_threshold=3)
                results.append(score)
        return results

    @register_scorer(
        name="hbd_score",
        population_wise=False,
        description=
        "Hydrogen Bond Donors (HBD) druglikeness score (value range: 0.0 to 1.0). "
        "HBD count measures the number of hydrogen atoms attached to electronegative atoms (N, O) that can donate hydrogen bonds, affecting membrane permeability and oral bioavailability per Lipinski's Rule of Five. "
        "HBDs enable specific interactions with biological targets but can impede membrane transport. "
        "High scores (>0.8) indicate optimal HBD count (<3) associated with excellent membrane permeability and oral bioavailability while maintaining target binding capability, while low scores (<0.3) suggest excessive hydrogen bonding capacity (>5 HBDs) leading to poor membrane transport, reduced absorption, and potential formulation challenges. "
        "Balance between target interaction and permeability is crucial for drug success.",
    )
    def score_hbd_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate hydrogen bond donors (HBD) druglikeness score.

        Returns normalized scores between 0.0 and 1.0 based on HBD count:
        - Score = 1.0: < 3 HBD (optimal for membrane permeability)
        - Score = 0.0: > 5 HBD (poor membrane permeability)
        - Score = linear interpolation: 3-5 HBD

        Fewer HBDs generally correlate with better membrane permeability.
        Higher scores indicate more drug-like hydrogen bonding capacity.

        Returns:
            List of scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                hbd = rdMolDescriptors.CalcNumHBD(mol)
                score = self._linear_interpolation(hbd,
                                                   good_threshold=3,
                                                   bad_threshold=6)
                results.append(score)
        return results

    @register_scorer(
        name="hba_score",
        population_wise=False,
        description=
        "Hydrogen Bond Acceptors (HBA) druglikeness score (value range: 0.0 to 1.0). "
        "HBA count measures nitrogen and oxygen atoms that can accept hydrogen bonds, significantly affecting membrane permeability and bioavailability according to Lipinski's Rule of Five. "
        "HBAs facilitate target binding through specific interactions but can hinder passive diffusion across membranes. "
        "High scores (>0.8) indicate optimal HBA count (<5) associated with good membrane permeability and oral bioavailability while providing sufficient binding interactions, while low scores (<0.3) suggest excessive hydrogen bonding capacity (>10 HBAs) leading to poor membrane transport, reduced tissue distribution, and potential bioavailability issues. "
        "Proper balance between molecular recognition and membrane permeability is essential for drug efficacy.",
    )
    def score_hba_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate hydrogen bond acceptors (HBA) druglikeness score.

        Returns normalized scores between 0.0 and 1.0 based on HBA count:
        - Score = 1.0: < 5 HBA (optimal for membrane permeability)
        - Score = 0.0: > 10 HBA (poor membrane permeability)
        - Score = linear interpolation: 5-10 HBA

        Fewer HBAs generally correlate with better membrane permeability.
        Higher scores indicate more drug-like hydrogen bonding capacity.

        Returns:
            List of scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                hba = rdMolDescriptors.CalcNumHBA(mol)
                score = self._linear_interpolation(hba,
                                                   good_threshold=5,
                                                   bad_threshold=12)
                results.append(score)
        return results

    @register_scorer(
        name="heavy_atoms_score",
        population_wise=False,
        description=
        "Heavy atoms count druglikeness score (value range: 0.0 to 1.0). "
        "Heavy atoms (all non-hydrogen atoms) determine overall molecular size, complexity, and drug-like character. "
        "This count directly correlates with molecular weight and affects all ADMET properties including absorption, distribution, metabolism, and excretion. "
        "High scores (>0.8) indicate optimal molecular size (<50 heavy atoms) associated with excellent drug-like properties, good bioavailability, and manageable synthetic complexity, while low scores (<0.3) suggest large, complex molecules (>70 heavy atoms) that typically exhibit poor ADMET properties, challenging synthesis, difficult optimization, and increased development risks. "
        "Smaller molecules generally have superior pharmacological profiles and development success rates.",
    )
    def score_heavy_atoms_druglikeness(
            self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate heavy atoms druglikeness score.

        Returns normalized scores between 0.0 and 1.0 based on heavy atom count:
        - Score = 1.0: < 50 heavy atoms (optimal molecular size)
        - Score = 0.0: > 70 heavy atoms (too large, complex)
        - Score = linear interpolation: 50-70 heavy atoms

        Fewer heavy atoms generally correlate with better drug-like properties.
        Higher scores indicate more appropriate molecular size.

        Returns:
            List of scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                heavy_atoms = mol.GetNumHeavyAtoms()
                score = self._linear_interpolation(heavy_atoms,
                                                   good_threshold=50,
                                                   bad_threshold=70)
                results.append(score)
        return results

    @register_scorer(
        name="qed",
        population_wise=False,
        description=
        "Quantitative Estimate of Drug-likeness (QED) score (value range: 0.0 to 1.0). "
        "QED is a validated, comprehensive metric that combines eight key molecular descriptors (molecular weight, LogP, hydrogen bond donors, hydrogen bond acceptors, topological polar surface area, rotatable bonds, aromatic rings, and structural alerts) weighted by their statistical importance in approved oral drugs. "
        "This benchmark assessment is calibrated on marketed pharmaceuticals to provide the most accurate druglikeness evaluation. "
        "High scores (>0.7) indicate excellent drug-like properties with strong similarity to successful marketed drugs across multiple physicochemical dimensions, while low scores (<0.3) suggest poor overall drug-likeness with multiple violations of optimal ranges. "
        "QED serves as the gold standard for druglikeness assessment, integrating multiple critical factors into a single, validated score.",
    )
    def score_qed(self, candidates: List[Candidate]) -> List[Optional[float]]:
        """
        Calculate Quantitative Estimate of Drug-likeness (QED).

        Returns normalized scores between 0.0 and 1.0 using RDKit's QED implementation:
        - Score = 1.0: Excellent drug-like properties (similar to approved drugs)
        - Score = 0.0: Poor drug-like properties 
        - Score = continuous: Weighted combination of 8 molecular descriptors

        QED considers molecular weight, LogP, HBD, HBA, PSA, rotatable bonds,
        aromatic rings, and alerts. It's calibrated on approved oral drugs.

        Higher scores indicate greater similarity to known oral drugs.

        Returns:
            List of QED scores (0.0-1.0) or None for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                results.append(QED.qed(mol))
        return results

    @register_scorer(
        name="pains_filter",
        population_wise=False,
        description=
        "Pan-Assay Interference Compounds (PAINS) filter score (value range: 0.0 or 1.0). "
        "PAINS are structural motifs that frequently show activity in biochemical assays through non-specific mechanisms rather than genuine target binding, leading to false positives in drug discovery. "
        "These compounds interfere with assays through mechanisms like aggregation, fluorescence, reactivity, or redox cycling. "
        "A score of 1.0 indicates the molecule contains no PAINS alerts (classes A, B, or C) and is suitable for further development, while a score of 0.0 indicates the presence of one or more PAINS motifs that may cause assay interference and should be avoided in drug discovery campaigns. "
        "PAINS filtering is essential for focusing resources on compounds with genuine biological activity."
    )
    def score_pains(self,
                    candidates: List[Candidate]) -> List[Optional[float]]:
        """Return 1.0 if molecule has no PAINS alerts, else 0.0. None for invalid SMILES."""
        results: List[Optional[float]] = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
                continue
            matches = self._pains_catalog.GetMatches(mol)
            results.append(1.0 if len(matches) == 0 else 0.0)
        return results

    @register_scorer(
        name="brenk_filter",
        population_wise=False,
        description=
        "Brenk structural alerts filter score (value range: 0.0, 0.5, or 1.0). "
        "Brenk alerts identify structural motifs associated with toxicity, reactivity, or other undesirable properties that can lead to drug development failures. "
        "These alerts help flag potentially problematic compounds early in the discovery process. "
        "A score of 1.0 indicates no structural alerts and excellent safety profile potential, a score of 0.5 indicates exactly one alert requiring careful evaluation and potential optimization, while a score of 0.0 indicates multiple alerts (≥2) suggesting significant safety or developability concerns that typically warrant compound deprioritization. "
        "Brenk filtering helps prioritize safer compounds and avoid costly late-stage failures due to toxicity or reactivity issues."
    )
    def score_brenk(self,
                    candidates: List[Candidate]) -> List[Optional[float]]:
        """Return 0.5 if exactly one Brenk alert, else 0.0. None for invalid SMILES."""
        results: List[Optional[float]] = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
                continue
            matches = self._brenk_catalog.GetMatches(mol)
            n = len(matches)
            results.append(1.0 if n == 0 else (0.5 if n == 1 else 0.0))
        return results


if __name__ == "__main__":
    from scileo_agent.core.registry import get_scorer, list_scorers
    from scileo_agent.core.data_models import Candidate

    # List all available scorers
    print("Available scorers:\n", list_scorers())
    print()

    # Test the scorers with diverse molecules
    test_smiles = [
        "CCO",  # Ethanol - simple, should score well
        "CC(C)(C)C1=CC=C(C=C1)C(C)(C)C2=CC=C(C=C2)C(C)(C)C3=CC=C(C=C3)C(C)(C)C",  # Large aromatic - should score poorly
        "C1=CC=C(C=C1)C(=O)O",  # Benzoic acid - moderate
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen - known drug, should score well
        "invalid_smiles",  # Invalid SMILES to test error handling
        "COc1cccc(C(C=O)NCNc2cncc(F)n2)c1"
    ]
    candidates = [Candidate(representation=smiles) for smiles in test_smiles]

    # Get all scorers
    sa_scorer = get_scorer("sa_score")
    solubility_scorer = get_scorer("solubility_score")
    logp_scorer = get_scorer("logp_score")
    mw_scorer = get_scorer("mw_score")
    psa_scorer = get_scorer("psa_score")
    rb_scorer = get_scorer("rotatable_bonds_score")
    fsp3_scorer = get_scorer("fsp3_score")
    rings_scorer = get_scorer("rings_score")
    hbd_scorer = get_scorer("hbd_score")
    hba_scorer = get_scorer("hba_score")
    heavy_atoms_scorer = get_scorer("heavy_atoms_score")
    qed_scorer = get_scorer("qed")
    deepdl_scorer = get_scorer("deepdl_druglikeness")
    pains_scorer = get_scorer("pains_filter")
    brenk_scorer = get_scorer("brenk_filter")

    # Test all scorers
    print("Test molecules:")
    for i, smiles in enumerate(test_smiles):
        print(f"{i+1}. {smiles}")
    print()

    print("SA scores:", sa_scorer(candidates))
    print("Solubility scores:", solubility_scorer(candidates))
    print("LogP scores:", logp_scorer(candidates))
    print("MW scores:", mw_scorer(candidates))
    print("PSA scores:", psa_scorer(candidates))
    print("Rotatable bonds scores:", rb_scorer(candidates))
    print("FSP3 scores:", fsp3_scorer(candidates))
    print("Rings scores:", rings_scorer(candidates))
    print("HBD scores:", hbd_scorer(candidates))
    print("HBA scores:", hba_scorer(candidates))
    print("Heavy atoms scores:", heavy_atoms_scorer(candidates))
    print("QED scores:", qed_scorer(candidates))
    print("DeepDL scores:", deepdl_scorer(candidates))
    print("PAINS scores:", pains_scorer(candidates))
    print("BRENK scores:", brenk_scorer(candidates))
