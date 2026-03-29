"""
RDKit utilities for similarity and diversity calculations.

Notes:
- RDKit exposes dynamic members that can confuse static analyzers.
  We disable related Pylint rules at the module level.
"""

# pylint: disable=import-error,no-name-in-module,no-member

from typing import Dict, List, Tuple
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator as rdFPGen

# Pre-initialize Morgan fingerprint generator to avoid deprecated API
_MORGAN_GEN = rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)


def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate Tanimoto similarity between two SMILES strings.

    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string

    Returns:
        Tanimoto similarity score between 0 and 1
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return 0.0

    fp1 = _MORGAN_GEN.GetFingerprint(mol1)
    fp2 = _MORGAN_GEN.GetFingerprint(mol2)

    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def calculate_population_diversity(smiles_list: List[str]) -> float:
    """
    Compute population-level diversity as 1 - average pairwise Tanimoto similarity
    using Morgan fingerprints (radius=2, nBits=2048). Invalid SMILES are ignored.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Diversity score in [0, 1]. Returns 0.0 if fewer than two valid molecules.
    """
    valid_mols = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            valid_mols.append(m)

    if len(valid_mols) < 2:
        return 0.0

    fps = [_MORGAN_GEN.GetFingerprint(m) for m in valid_mols]

    total_similarity = 0.0
    num_pairs = 0
    for i in range(len(fps) - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total_similarity += float(sum(sims))
        num_pairs += len(sims)

    if num_pairs == 0:
        return 0.0

    mean_similarity = total_similarity / num_pairs
    return 1.0 - mean_similarity


def select_top_diverse_modes(smiles_list: List[str],
                             scores: List[float],
                             tanimoto_threshold: float,
                             k: int,
                             leniency: int = 0) -> List[int]:
    """
    Select indices of the top-k diverse molecules based on scores with a
    Tanimoto similarity threshold.

    A candidate is added greedily in descending score order if its Tanimoto
    similarity to every already-selected molecule is strictly less than
    ``tanimoto_threshold``.

    Args:
        smiles_list: List of SMILES strings.
        scores: List of numeric scores aligned with ``smiles_list``.
        tanimoto_threshold: Maximum allowed pairwise similarity between any two
            selected molecules. A pair is allowed only if similarity < threshold.
        k: Number of molecules to select. If k <= 0, returns an empty list.
        leniency: the number molecules from the top k that are allowed to be similar to the already selected molecules.

    Returns:
        List of selected indices into the original lists, in the order they were
        selected (highest-score-first, while respecting diversity).

    Notes:
        - Invalid SMILES are ignored and cannot be selected.
        - If fewer than k valid and mutually diverse molecules exist, returns as
          many as possible.
    """
    if len(smiles_list) != len(scores):
        raise ValueError("smiles_list and scores must have the same length")

    if k <= 0:
        return []

    # Build fingerprints for valid molecules only
    valid_entries = []  # (index, score, fingerprint)
    for idx, (smiles, score) in enumerate(zip(smiles_list, scores)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = _MORGAN_GEN.GetFingerprint(mol)
        valid_entries.append((idx, float(score), fp))

    if not valid_entries:
        return []

    # Sort by score descending, then by index for deterministic behavior
    valid_entries.sort(key=lambda t: (t[1], -t[0]), reverse=True)

    selected_indices: List[int] = []
    selected_fps = []

    for idx, _score, fp in valid_entries:
        if len(selected_indices) >= k:
            break

        if not selected_fps:
            selected_indices.append(idx)
            selected_fps.append(fp)
            continue

        # Compute similarity to already selected set efficiently
        sims = DataStructs.BulkTanimotoSimilarity(fp, selected_fps)
        # Enforce strict inequality as requested (< threshold)
        number_of_similar_molecules = sum(1 for sim in sims
                                          if float(sim) >= tanimoto_threshold)
        if number_of_similar_molecules <= leniency:
            selected_indices.append(idx)
            selected_fps.append(fp)

    return selected_indices


def structure_filter(frag, compound):

    # Example: Compound and fragment as SMILES
    compound_smiles = frag  # e.g., toluene
    fragment_smarts = compound  # benzene ring

    # Convert to RDKit Mol objects
    compound = Chem.MolFromSmiles(compound_smiles)
    fragment = Chem.MolFromSmarts(fragment_smarts)
    
    has_fragment = False
    
    # Check for substructure match
    if compound is not None and fragment is not None:
        try:
            has_fragment = compound.HasSubstructMatch(fragment)
        except Exception as e:
            print(f"Error in substructure matching: {e}")

    return has_fragment


def visualize_substructure_match(smiles: str,
                                 smarts: str,
                                 pattern_name: str,
                                 output_path: str = None):
    """
    Visualize a molecule with highlighted substructure matches.
    
    Args:
        smiles: SMILES string of the molecule
        smarts: SMARTS pattern to highlight
        pattern_name: Name of the pattern for the title
        output_path: Optional path to save the image
        
    Returns:
        RDKit drawing object
    """
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts(smarts)

    if mol is None or pattern is None:
        print(f"Failed to parse molecule or pattern: {smiles}, {smarts}")
        return None


def filter_smiles_preserves_existing_hits(
    smiles_list: List[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Placeholder implementation that preserves all valid SMILES strings.

    This keeps compatibility with downstream consumers that expect the helper
    without re-introducing the historical SMARTS-based filtering logic.

    Args:
        smiles_list: SMILES strings to evaluate.

    Returns:
        A tuple of (kept_smiles, dropped_reasons) where dropped_reasons maps the
        dropped SMILES to a list of textual reasons. Invalid SMILES are dropped.
    """
    kept: List[str] = []
    dropped_reasons: Dict[str, List[str]] = {}

    for smi in smiles_list:
        if not smi:
            dropped_reasons.setdefault("<empty>", []).append("empty_string")
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            dropped_reasons.setdefault(smi, []).append("invalid_smiles")
            continue

        kept.append(smi)

    return kept, dropped_reasons

    # Get all substructure matches
    matches = mol.GetSubstructMatches(pattern)

    if not matches:
        print(f"No matches found for pattern {pattern_name}")
        return None

    # Highlight all matches
    highlight_atoms = set()
    highlight_bonds = set()

    for match in matches:
        highlight_atoms.update(match)
        # Find bonds between highlighted atoms
        for i in range(len(match)):
            for j in range(i + 1, len(match)):
                bond = mol.GetBondBetweenAtoms(match[i], match[j])
                if bond is not None:
                    highlight_bonds.add(bond.GetIdx())

    # Set highlight colors
    highlight_colors = {
        atom: (1.0, 0.0, 0.0)
        for atom in highlight_atoms
    }  # Red
    bond_colors = {bond: (1.0, 0.0, 0.0) for bond in highlight_bonds}  # Red

    # Create drawing object
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)

    # Draw the molecule
    drawer.DrawMolecule(mol,
                        highlightAtoms=list(highlight_atoms),
                        highlightAtomColors=highlight_colors,
                        highlightBonds=list(highlight_bonds),
                        highlightBondColors=bond_colors)
    drawer.FinishDrawing()

    # Save if output path provided
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        print(f"Saved visualization to {output_path}")

    return drawer


def test_smarts_with_visualization(smiles: str, smarts: str,
                                   pattern_name: str):
    """
    Test a SMARTS pattern against a molecule and visualize the match.
    
    Args:
        smiles: SMILES string of the molecule
        smarts: SMARTS pattern to test
        pattern_name: Name of the pattern
        
    Returns:
        bool: True if match found, False otherwise
    """
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts(smarts)

    if mol is None or pattern is None:
        print(f"Failed to parse: {smiles} or {smarts}")
        return False

    has_match = mol.HasSubstructMatch(pattern)

    if has_match:
        print(f"\n✓ MATCH FOUND for {pattern_name}")
        print(f"  Molecule: {smiles}")
        print(f"  Pattern: {smarts}")

        # Visualize the match
        import re
        safe_smiles = re.sub(r'[/\\]', '_', smiles)
        output_path = f"match_{pattern_name}_{safe_smiles}.png"
        visualize_substructure_match(smiles, smarts, pattern_name, output_path)
    else:
        print(f"\n✗ NO MATCH for {pattern_name}")
        print(f"  Molecule: {smiles}")
        print(f"  Pattern: {smarts}")

    return has_match


if __name__ == "__main__":
    # Minimal runnable test cases for each SMARTS filter name using filter_smiles
    # Each entry: name -> {"positives": [...], "negatives": [...]} where
    #  - positives should be removed by filter_smiles when that SMARTS is active
    #  - negatives should pass through unchanged
    test_cases = {
        # Sulfonamides
        "sulfone_general": {
            "positives": ["CS(=O)(=O)C"],
            "negatives": ["c1ccccc1"],
        },
        "sulfonate_ester": {
            "positives": ["CO[S](=O)(=O)c1ccc(C)cc1"],
            "negatives": ["c1ccccc1"],
        },
        "sulfonamide_h1": {
            "positives": [
                "Nc1ccc(S(=O)(=O)Nc2ncccn2)cc1",
                "CC1=NN=C(S1)NS(=O)(=O)c2ccc(N)cc2"
            ],
            "negatives": ["c1ccccc1"],
        },

        # Tetracyclic skeletons
        "tetracyclic_core": {
            "positives": [
                "C[C@]1(c2cccc(c2C(=O)C3=C([C@]4([C@@H](C[C@@H]31)[C@@H](C(=C(C4=O)C(=O)N)O)N(C)C)O)O)O)O",  # Tetracycline
                "CC1C2C(=O)C3=C(C(=C(C(=O)C3=C(C2=C(C(=O)C4=C1C(=C(C=C4O)O)O)O)O)O)O)N(C)C",  # Doxycycline  
                "CN(C)c1ccc(c2c1C[C@H]3C[C@H]4[C@@H](C(=C(C(=O)[C@]4(C(=C3C2=O)O)O)C(=O)N)O)N(C)C)O",  # Minocycline
            ],
            "negatives": [
                "O=C2c1ccccc1C(=O)c3ccccc23",  # Anthraquinone (3 rings only)
                "c1ccc(O)cc1",  # Phenol (single ring)
            ],
        },
        "beta_lactams": {
            "postivies": [
                "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc3ccccc3)C(=O)O)C",
                "CC1(C)[C@H](C(=O)O)N2C(=O)[C@@H](NC(=O)CO)C(C)(C)S[C@H]12c1ccc(O)cc1O",
                "CC1(C)[C@@H](C(=O)O)N2C(=O)[C@@H](NC(=O)Cc3ccccc3)C(C)(C)S[C@H]12"
            ],
            "negatives": []
        },
        "quinolone": {
            "positives": [
                    # ciprofloxacin
                    "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",  # PubChem canonical. :contentReference[oaicite:0]{index=0}
                    # levofloxacin (S-ofloxacin) — your string was already correct
                    "C[C@H]1COc2c3n1cc(c(=O)c3cc(c2N4CCN(CC4)C)F)C(=O)O",  # Wikipedia SMILES. :contentReference[oaicite:1]{index=1}
                    # norfloxacin — add explicit E/Z bonds per Wikipedia
                    "O=C(O)\\C2=C\\N(c1cc(c(F)cc1C2=O)N3CCNCC3)CC",  # Wikipedia SMILES. :contentReference[oaicite:2]{index=2}
                    # moxifloxacin — your string was already correct
                    "COc1c2c(cc(c1N3C[C@@H]4CCCN[C@@H]4C3)F)c(=O)c(cn2C5CC5)C(=O)O",  # Wikipedia SMILES. :contentReference[oaicite:3]{index=3}
                    # enoxacin (swapping your last one for a valid quinolone)
                    "Fc1c(nc2c(c1)C(=O)C(\\C(=O)O)=C/N2CC)N3CCNCC3",  # Wikipedia SMILES. :contentReference[oaicite:4]{index=4}      
            ],
            "negatives": [
                "c1ccccc1",
            ],
        },
        # "pyridone_or_2hp": {
        #     "positives": [
        #         "Oc1ccccn1",  # 2-hydroxypyridine
        #         "O=c1cccc[nH]1",  # 2-pyridone
        #     ],
        #     "negatives": ["c1ccncc1", "CCO"],
        # },
        # "pyridone_like_lactam": {
        #     "positives": [
        #         "O=c1cccc[nH]1",  # 2-pyridone tautomer captures ring amide [nH]
        #     ],
        #     "negatives": ["c1ccncc1"],
        # },
        # "pyridone_like_enol": {
        #     "positives": ["Oc1ncccc1"],
        #     "negatives": ["c1ccncc1"],
        # },
        # "beta_lactam": {
        #     "positives": ["O=C1NCC1"],  # 2-azetidinone core
        #     "negatives": ["C1CCCCC1"],
        # },

        # # Quinone family
        # "p_benzoquinone": {
        #     "positives": ["O=C1C=CC(=O)C=C1"],
        #     "negatives": ["c1ccccc1"],
        # },
        # "p_quinone_imine": {
        #     "positives": ["O=C1C=CC(=N)C=C1"],
        #     "negatives": ["c1ccccc1N"],
        # },
        # "p_hydroquinone": {
        #     "positives": ["Oc1ccc(O)cc1"],
        #     "negatives": ["c1ccccc1O"],
        # },
        # # Ring-constrained variants use same scaffolds and should also match
        # "p_benzoquinone_ring": {
        #     "positives": ["O=C1C=CC(=O)C=C1"],
        #     "negatives": ["c1ccccc1"],
        # },
        # "p_quinone_imine_ring": {
        #     "positives": ["O=C1C=CC(=N)C=C1"],
        #     "negatives": ["c1ccccc1N"],
        # },

        # # Phenol/Aniline cross patterns
        # "aniline_pheno": {
        #     "positives": ["c1ccc(N)cc1O"],
        #     "negatives": ["c1ccccc1O"],
        # },
        # "pheno_aniline": {
        #     "positives": ["c1ccc(O)cc1N"],
        #     "negatives": ["c1ccccc1N"],
        # },
        # # Neutral O variants (phenoxide form)
        # "aniline_pheno_neutralO": {
        #     "positives": ["c1ccc(N)cc1OC"],
        #     "negatives": ["c1ccc(N)cc1O"],
        # },
        # "pheno_aniline_neutralO": {
        #     "positives": ["c1ccc(OC)cc1N"],
        #     "negatives": ["c1ccc(O)cc1N"],
        # },

        # Ring filters
        "small_strained": {
            "positives": ["C1CC1", "C1CCC1"],
            "negatives": ["C1CCCCC1"],
        },
        "macrocycle": {
            "positives": ["C1CCCCCCCCCCC1"],
            "negatives": ["C1CCCCC1"],
        },
        # Refined ring filters - only problematic hetero-hetero adjacencies
        "OO_SS_adjacency": {
            "positives": ["O1OCCCC1"],  # O-O adjacency (rare and problematic)
            "negatives": ["c1ccccc1", "N1CCNCC1",
                          "n1occc1"],  # Normal rings should pass
        },
        "hetero_halogen_adjacency": {
            "positives": ["c1c(Cl)ccc1",
                          "c1c(Br)ccc1"],  # hetero-halogen adjacency
            "negatives": ["c1ccccc1", "N1CCNCC1",
                          "n1occc1"],  # Normal rings should pass
        },
        # Three fused, non-aromatic rings. Use a simple polycyclic cage-like hydrocarbon
        # If this is too strict for the chosen scaffold, the test will be reported.
        "polyfused_hetero": {
            "positives": [],
            "negatives": ["C1CCCCC1"],
        },

        # # Pyrimidine derivatives
        "diaminopyrimidine": {
            "positives": [
                "CN(CC1CCCCC1C(Cc2c(N)nc(N)nc2)=O)Cl",  # Complex molecule with diaminopyrimidine
                "Nc1nc(N)ccn1",  # Simple diaminopyrimidine
            ],
            "negatives":
            ["c1ccccc1", "Nc1ncccc1"
             ],  # Benzene and simple pyrimidine without amino groups
        },
        "monoaminopyrimidine": {
            "positives": ["NC1=NC=CC=N1"],
            "negatives": ["c1ccccc1"],
        },

        # # Existing hits - specific molecules that should be filtered
        "thiazole_5Cl_4carboxylic_acid": {
            "positives": ["ClC1=C(C(O)=O)N=CS1"],
            "negatives": ["c1ccccc1", "n1occc1"],
        },
        "thiazole_4Cl_5carboxylic_acid": {
            "positives": ["Clc1nc(sc1)C(=O)O"],
            "negatives": ["c1ccccc1", "n1occc1"],
        },
        "pyridine_2cyano_4F_5Cl": {
            "positives": ["N#CC1=C(Cl)C=C(F)C=N1"],
            "negatives": ["c1ccccc1", "c1ccncc1"],
        },
        "thiazole_like_N1S4_SMARTS": {
            "positives": ["N#Cc1c(Cl)sc(N(CCN2CCOCC2)O)n1"],
            "negatives": ["c1ccccc1", "c1ccncc1"],
        },
    }

    total = 0
    passed = 0
    failed = 0
    failures = []

    def run_case(name: str, positives, negatives):
        patt = BAD_SMARTS_MAP.get(name)
        if patt is None:
            return False, {"error": "pattern_not_found"}

        # Filter with only this SMARTS active
        filtered_pos = filter_smiles(positives, smarts_list=[patt])
        filtered_neg = filter_smiles(negatives, smarts_list=[patt])

        ok_pos = len(filtered_pos) == 0
        ok_neg = filtered_neg == negatives

        # Visualize matches for positive molecules
        if positives:
            print(f"\n{'='*60}")
            print(f"TESTING PATTERN: {name}")
            print(f"{'='*60}")

            # Get the SMARTS pattern string for visualization
            smarts_str = None
            for group_patterns in SMARTS_GROUPS.values():
                for pattern_name, pattern_mol in group_patterns:
                    if pattern_name == name:
                        smarts_str = Chem.MolToSmarts(pattern_mol)
                        break
                if smarts_str:
                    break

            if smarts_str:
                print(f"SMARTS Pattern: {smarts_str}")
                print(f"Expected to match: {positives}")
                print(f"Expected to NOT match: {negatives}")

                # Test each positive molecule
                for smiles in positives:
                    test_smarts_with_visualization(smiles, smarts_str, name)

                # Test a few negative molecules to show they don't match
                print("\nTesting negative examples (should NOT match):")
                for smiles in negatives[:2]:  # Test first 2 negatives
                    test_smarts_with_visualization(smiles, smarts_str, name)

        if ok_pos and ok_neg:
            return True, {}
        return False, {
            "remaining_pos":
            filtered_pos,
            "neg_diff": [s for s in negatives if s not in filtered_neg] +
            [s for s in filtered_neg if s not in negatives],
        }

    # Execute tests for all defined cases
    for name, tc in test_cases.items():
        total += 1
        ok, info = run_case(name, tc.get("positives", []),
                            tc.get("negatives", []))
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((name, info))

    # Basic sanity check for default filter_smiles behavior with BAD_SMARTS_LIST and PAINS
    sanity_input = [
        "O=C1C=CC(=O)C=C1",  # benzoquinone -> should be removed
        "CCO",  # ethanol -> should remain
    ]
    sanity_output = filter_smiles(sanity_input)
    total += 1
    if sanity_output == ["CCO"]:
        passed += 1
    else:
        failed += 1
        failures.append(("default_filter_smiles", {"out": sanity_output}))

    print(f"SMARTS filter tests: passed {passed}/{total}, failed {failed}")
    if failures:
        for name, info in failures:
            print(f" - FAIL: {name}: {info}")
        raise SystemExit(1)
    raise SystemExit(0)
