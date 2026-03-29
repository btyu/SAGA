"""
Written by Jan H. Jensen

"""

import logging
import random
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def _delete_atom() -> str:
    choices = [
        "[*:1]~[D1]>>[*:1]",
        "[*:1]~[D2]~[*:2]>>[*:1]-[*:2]",
        "[*:1]~[D3](~[*;!H0:2])~[*:3]>>[*:1]-[*:2]-[*:3]",
        "[*:1]~[D4](~[*;!H0:2])(~[*;!H0:3])~[*:4]>>[*:1]-[*:2]-[*:3]-[*:4]",
        "[*:1]~[D4](~[*;!H0;!H1:2])(~[*:3])~[*:4]>>[*:1]-[*:2](-[*:3])-[*:4]",
    ]
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]
    return str(np.random.choice(choices, p=p))


def _append_atom() -> str:
    choices = [
        ["single", ["C", "N", "O", "F", "S", "Cl", "Br"], [1.0 / 7.0] * 7],
        ["double", ["C", "N", "O"], [1.0 / 3.0] * 3],
        ["triple", ["C", "N"], [1.0 / 2.0] * 2],
    ]
    p_BO = [0.60, 0.35, 0.05]
    index = int(np.random.choice(list(range(3)), p=p_BO))
    BO, atom_list, probs = choices[index]
    new_atom = str(np.random.choice(atom_list, p=probs))
    if BO == "single":
        return "[*;!H0:1]>>[*:1]X".replace("X", "-" + new_atom)
    if BO == "double":
        return "[*;!H0;!H1:1]>>[*:1]X".replace("X", "=" + new_atom)
    return "[*;H3:1]>>[*:1]X".replace("X", "#" + new_atom)


def _insert_atom() -> str:
    choices = [
        ["single", ["C", "N", "O", "S"], [1.0 / 4.0] * 4],
        ["double", ["C", "N"], [1.0 / 2.0] * 2],
        ["triple", ["C"], [1.0]],
    ]
    p_BO = [0.60, 0.35, 0.05]
    index = int(np.random.choice(list(range(3)), p=p_BO))
    BO, atom_list, probs = choices[index]
    new_atom = str(np.random.choice(atom_list, p=probs))
    if BO == "single":
        return "[*:1]~[*:2]>>[*:1]X[*:2]".replace("X", new_atom)
    if BO == "double":
        return "[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]".replace("X", new_atom)
    return "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]".replace("X", new_atom)


def _change_bond_order() -> str:
    choices = [
        "[*:1]!- [*:2]>>[*:1]-[*:2]".replace("!- ", "!-"),
        "[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]",
        "[*:1]#[*:2]>>[*:1]=[*:2]",
        "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]",
    ]
    p = [0.45, 0.45, 0.05, 0.05]
    return str(np.random.choice(choices, p=p))


def _delete_cyclic_bond() -> str:
    return "[*:1]@[*:2]>>([*:1].[*:2])"


def _add_ring() -> str:
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    p = [0.05, 0.05, 0.45, 0.45]
    return str(np.random.choice(choices, p=p))


def _change_atom(mol: "Chem.Mol") -> str:
    choices = ["#6", "#7", "#8", "#9", "#16", "#17", "#35"]
    p = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]
    X = str(np.random.choice(choices, p=p))
    try:
        while not mol.HasSubstructMatch(Chem.MolFromSmarts("[" + X + "]")):
            X = str(np.random.choice(choices, p=p))
    except Exception:
        X = "#6"
    Y = str(np.random.choice(choices, p=p))
    while Y == X:
        Y = str(np.random.choice(choices, p=p))
    return "[X:1]>>[Y:1]".replace("X", X).replace("Y", Y)


def _is_valid_product(mol: "Chem.Mol") -> bool:
    try:
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        if mol.GetNumAtoms() == 0:
            return False
        return True
    except Exception:
        return False


def mutate_smiles(smiles: str, mutation_rate: float = 1.0) -> str:
    """
    Apply a single non-LLM mutation to a SMILES string using RDKit reaction SMARTS.
    If no valid mutation is produced after several tries, returns the original SMILES.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        if random.random() > mutation_rate:
            return smiles
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        p = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
        for _ in range(10):
            ops: List[str] = [
                _insert_atom(),
                _change_bond_order(),
                _delete_cyclic_bond(),
                _add_ring(),
                _delete_atom(),
                _change_atom(mol),
                _append_atom(),
            ]
            rxn_smarts = str(np.random.choice(ops, p=p))
            try:
                rxn = AllChem.ReactionFromSmarts(rxn_smarts)
                if rxn is None:
                    continue
                trials = rxn.RunReactants((mol,))
            except Exception:
                continue
            valid_products: List[str] = []
            for prod_tuple in trials:
                try:
                    prod = prod_tuple[0]
                except Exception:
                    continue
                if _is_valid_product(prod):
                    try:
                        new_smiles = Chem.MolToSmiles(prod, canonical=True)
                    except Exception:
                        continue
                    if new_smiles:
                        valid_products.append(new_smiles)
            if valid_products:
                return str(random.choice(valid_products))
        return smiles
    except Exception as e:
        logging.error("Non-LLM mutation failed for %s: %s", smiles, e, exc_info=True)
        return smiles


