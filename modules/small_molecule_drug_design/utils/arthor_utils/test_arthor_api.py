#!/usr/bin/env python3
"""
Test script for Arthor API integration.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.small_molecule_drug_design.utils.arthor_utils import search_similar_compounds


def test_single_smiles():
    """Test with a single SMILES string."""
    print("=" * 60)
    print("Test 1: Single SMILES string")
    print("=" * 60)
    
    smiles = "c1ccccc1"  # benzene
    print(f"Input SMILES: {smiles}")
    
    try:
        results = search_similar_compounds(smiles, max_results=5)
        print(f"\nResults: {len(results)} entry(ies)")
        
        for input_smiles, similar_list in results.items():
            print(f"\nInput: {input_smiles}")
            print(f"Found {len(similar_list)} similar compounds:")
            for i, similar_smiles in enumerate(similar_list, 1):
                print(f"  {i}. {similar_smiles}")
        
        return len(similar_list) > 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_smiles():
    """Test with a list of SMILES strings."""
    print("\n" + "=" * 60)
    print("Test 2: Multiple SMILES strings")
    print("=" * 60)
    
    smiles_list = ["c1ccccc1", "CCO"]  # benzene and ethanol
    print(f"Input SMILES: {smiles_list}")
    
    try:
        results = search_similar_compounds(smiles_list, max_results=3)
        print(f"\nResults: {len(results)} entry(ies)")
        
        for input_smiles, similar_list in results.items():
            print(f"\nInput: {input_smiles}")
            print(f"Found {len(similar_list)} similar compounds:")
            for i, similar_smiles in enumerate(similar_list, 1):
                print(f"  {i}. {similar_smiles}")
        
        return all(len(similar_list) > 0 for similar_list in results.values())
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_empty_input():
    """Test with empty input."""
    print("\n" + "=" * 60)
    print("Test 3: Empty input")
    print("=" * 60)
    
    try:
        results = search_similar_compounds([])
        print(f"Results: {results}")
        print("Expected: empty dict")
        return results == {}
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Arthor API Integration")
    print("=" * 60)
    
    results = []
    results.append(("Single SMILES", test_single_smiles()))
    results.append(("Multiple SMILES", test_multiple_smiles()))
    results.append(("Empty input", test_empty_input()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
