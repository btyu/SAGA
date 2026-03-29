"""
GB-GA crossover and mutation utilities ported from
https://github.com/jensengroup/GB_GA (MIT License).

These helpers operate on SMILES strings and return canonical SMILES outputs.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

rdBase.DisableLog("rdApp.error")


def _to_mol(smiles: str) -> Optional[Chem.Mol]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        # Keep the molecule even if kekulization fails
        pass
    return mol


def _cut(mol: Chem.Mol) -> Optional[Tuple[Chem.Mol, ...]]:
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[*]-;!@[*]")):
        return None

    bond_indices = random.choice(
        mol.GetSubstructMatches(Chem.MolFromSmarts("[*]-;!@[*]")))
    bonds = [mol.GetBondBetweenAtoms(bond_indices[0], bond_indices[1]).GetIdx()]
    fragments_mol = Chem.FragmentOnBonds(
        mol, bonds, addDummies=True, dummyLabels=[(1, 1)])
    try:
        return Chem.GetMolFrags(
            fragments_mol, asMols=True, sanitizeFrags=True)
    except ValueError:
        return None


def _cut_ring(mol: Chem.Mol) -> Optional[Tuple[Chem.Mol, ...]]:
    for _ in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(
                    Chem.MolFromSmarts("[R]@[R]@[R]@[R]")):
                return None
            bond_indices = random.choice(
                mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")))
            bond_pairs = ((bond_indices[0], bond_indices[1]),
                          (bond_indices[2], bond_indices[3]))
        else:
            if not mol.HasSubstructMatch(
                    Chem.MolFromSmarts("[R]@[R;!D2]@[R]")):
                return None
            bond_indices = random.choice(
                mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")))
            bond_pairs = ((bond_indices[0], bond_indices[1]),
                          (bond_indices[1], bond_indices[2]))

        bonds = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bond_pairs]
        fragments_mol = Chem.FragmentOnBonds(
            mol,
            bonds,
            addDummies=True,
            dummyLabels=[(1, 1), (1, 1)],
        )

        try:
            fragments = Chem.GetMolFrags(
                fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) == 2:
                return fragments
        except ValueError:
            return None
    return None


def _ring_ok(mol: Chem.Mol) -> bool:
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True

    if mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]")):
        return False

    cycle_list = mol.GetRingInfo().AtomRings()
    if not cycle_list:
        return True
    max_cycle_length = max(len(ring) for ring in cycle_list)
    if max_cycle_length > 6:
        return False

    if mol.HasSubstructMatch(Chem.MolFromSmarts("[r3,r4]=[r3,r4]")):
        return False

    return True


# Parameters from GB_GA (data set statistics)
_AVERAGE_SIZE = 39.15
_SIZE_STDEV = 3.50


def _mol_ok(mol: Chem.Mol) -> bool:
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return False

    target_size = _SIZE_STDEV * np.random.randn() + _AVERAGE_SIZE
    if mol.GetNumAtoms() <= 5:
        return False
    if mol.GetNumAtoms() >= target_size:
        return False
    return True


def _crossover_internal(
        parent_a: Chem.Mol, parent_b: Chem.Mol) -> Optional[Chem.Mol]:
    parent_smiles = {
        Chem.MolToSmiles(parent_a, canonical=True),
        Chem.MolToSmiles(parent_b, canonical=True),
    }

    for _ in range(10):
        if random.random() <= 0.5:
            fragments_a = _cut(parent_a)
            fragments_b = _cut(parent_b)
            if fragments_a is None or fragments_b is None:
                continue

            rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
            trial_mols = []
            for frag_a in fragments_a:
                for frag_b in fragments_b:
                    trial_mols.append(rxn.RunReactants((frag_a, frag_b))[0])

            candidates = []
            for mol in trial_mols:
                child = mol[0]
                if _mol_ok(child):
                    candidates.append(child)
            if not candidates:
                continue

            new_mol = random.choice(candidates)
            smiles = Chem.MolToSmiles(new_mol, canonical=True)
            if smiles and smiles not in parent_smiles:
                return new_mol

        else:
            fragments_a = _cut_ring(parent_a)
            fragments_b = _cut_ring(parent_b)
            if fragments_a is None or fragments_b is None:
                continue

            rxn_smarts1 = [
                "[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]",
                "[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]",
            ]
            rxn_smarts2 = [
                "([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]",
                "([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]",
            ]

            trial = []
            for smarts in rxn_smarts1:
                rxn1 = AllChem.ReactionFromSmarts(smarts)
                for frag_a in fragments_a:
                    for frag_b in fragments_b:
                        trial.append(rxn1.RunReactants((frag_a, frag_b))[0])

            intermediates = []
            for smarts in rxn_smarts2:
                rxn2 = AllChem.ReactionFromSmarts(smarts)
                for mol in trial:
                    intermediate = mol[0]
                    if _mol_ok(intermediate):
                        intermediates.extend(rxn2.RunReactants((intermediate,)))

            candidates = []
            for mol in intermediates:
                child = mol[0]
                if _mol_ok(child) and _ring_ok(child):
                    candidates.append(child)

            if not candidates:
                continue

            new_mol = random.choice(candidates)
            smiles = Chem.MolToSmiles(new_mol, canonical=True)
            if smiles and smiles not in parent_smiles:
                return new_mol

    return None


def gbga_crossover(parent_a_smiles: str,
                   parent_b_smiles: str) -> Optional[str]:
    """Perform GB-GA crossover on two SMILES strings."""
    parent_a = _to_mol(parent_a_smiles)
    parent_b = _to_mol(parent_b_smiles)
    if parent_a is None or parent_b is None:
        return None

    child = _crossover_internal(parent_a, parent_b)
    if child is None:
        return None
    return Chem.MolToSmiles(child, canonical=True)


def _mutation_smarts_insert_atom() -> str:
    choices = [
        ["single", ["C", "N", "O", "S"], [1.0 / 4.0] * 4],
        ["double", ["C", "N"], [1.0 / 2.0] * 2],
        ["triple", ["C"], [1.0]],
    ]
    p_bo = [0.60, 0.35, 0.05]
    index = int(np.random.choice(list(range(3)), p=p_bo))
    bond_order, atom_list, probs = choices[index]
    new_atom = str(np.random.choice(atom_list, p=probs))
    if bond_order == "single":
        return "[*:1]~[*:2]>>[*:1]X[*:2]".replace("X", new_atom)
    if bond_order == "double":
        return "[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]".replace("X", new_atom)
    return "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]".replace("X", new_atom)


def _mutation_smarts_change_bond_order() -> str:
    choices = [
        "[*:1]!-[*:2]>>[*:1]-[*:2]",
        "[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]",
        "[*:1]#[*:2]>>[*:1]=[*:2]",
        "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]",
    ]
    probs = [0.45, 0.45, 0.05, 0.05]
    return str(np.random.choice(choices, p=probs))


def _mutation_smarts_delete_cyclic_bond() -> str:
    return "[*:1]@[*:2]>>([*:1].[*:2])"


def _mutation_smarts_add_ring() -> str:
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    probs = [0.05, 0.05, 0.45, 0.45]
    return str(np.random.choice(choices, p=probs))


def _mutation_smarts_delete_atom() -> str:
    choices = [
        "[*:1]~[D1:2]>>[*:1]",
        "[*:1]~[D2:2]~[*:3]>>[*:1]-[*:3]",
        "[*:1]~[D3:2](~[*;!H0:3])~[*:4]>>[*:1]-[*:3]-[*:4]",
        "[*:1]~[D4:2](~[*;!H0:3])(~[*;!H0:4])~[*:5]>>[*:1]-[*:3]-[*:4]-[*:5]",
        "[*:1]~[D4:2](~[*;!H0;!H1:3])(~[*:4])~[*:5]>>[*:1]-[*:3](-[*:4])-[*:5]",
    ]
    probs = [0.25, 0.25, 0.25, 0.1875, 0.0625]
    return str(np.random.choice(choices, p=probs))


def _mutation_smarts_append_atom() -> str:
    choices = [
        ["single", ["C", "N", "O", "F", "S", "Cl", "Br"], [1.0 / 7.0] * 7],
        ["double", ["C", "N", "O"], [1.0 / 3.0] * 3],
        ["triple", ["C", "N"], [1.0 / 2.0] * 2],
    ]
    p_bo = [0.60, 0.35, 0.05]
    index = int(np.random.choice(list(range(3)), p=p_bo))
    bond_order, atom_list, probs = choices[index]
    new_atom = str(np.random.choice(atom_list, p=probs))
    if bond_order == "single":
        return "[*;!H0:1]>>[*:1]X".replace("X", "-" + new_atom)
    if bond_order == "double":
        return "[*;!H0;!H1:1]>>[*:1]X".replace("X", "=" + new_atom)
    return "[*;H3:1]>>[*:1]X".replace("X", "#" + new_atom)


def _mutation_smarts_change_atom(mol: Chem.Mol) -> str:
    choices = ["#6", "#7", "#8", "#9", "#16", "#17", "#35"]
    probs = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]
    element_a = str(np.random.choice(choices, p=probs))
    try:
        while not mol.HasSubstructMatch(Chem.MolFromSmarts(f"[{element_a}]")):
            element_a = str(np.random.choice(choices, p=probs))
    except Exception:
        element_a = "#6"

    element_b = str(np.random.choice(choices, p=probs))
    while element_b == element_a:
        element_b = str(np.random.choice(choices, p=probs))

    return "[X:1]>>[Y:1]".replace("X", element_a).replace("Y", element_b)


def _mutate_internal(mol: Chem.Mol, mutation_rate: float) -> Optional[Chem.Mol]:
    if random.random() > mutation_rate:
        return mol

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        return mol

    probs = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for _ in range(10):
        rxn_smarts_options = [
            _mutation_smarts_insert_atom(),
            _mutation_smarts_change_bond_order(),
            _mutation_smarts_delete_cyclic_bond(),
            _mutation_smarts_add_ring(),
            _mutation_smarts_delete_atom(),
            _mutation_smarts_change_atom(mol),
            _mutation_smarts_append_atom(),
        ]
        rxn_smarts = str(np.random.choice(rxn_smarts_options, p=probs))
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        trial_products = rxn.RunReactants((mol,))

        candidates = []
        for prod in trial_products:
            candidate = prod[0]
            if _mol_ok(candidate) and _ring_ok(candidate):
                candidates.append(candidate)
        if candidates:
            return random.choice(candidates)

    return None


def gbga_mutate(smiles: str, mutation_rate: float = 1.0) -> Optional[str]:
    """Mutate a SMILES string using GB-GA mutation rules."""
    mol = _to_mol(smiles)
    if mol is None:
        return None

    mutant = _mutate_internal(mol, mutation_rate)
    if mutant is None:
        return None
    return Chem.MolToSmiles(mutant, canonical=True)







