from rdkit import Chem

# SMARTS for quinolone core (working pattern from rdkit_utils.py)
quinolone = Chem.MolFromSmarts("[n,N]~[c,C]~[c,C]~[c,C](=O)~[c,C]~[c,C]")


def screen_smiles(file_path):
    with open(file_path) as f:
        for line in f:
            smi = line.strip()
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol and mol.HasSubstructMatch(quinolone):
                print(f"Quinolone match: {smi}")
            else:
                pass


if __name__ == "__main__":
    screen_smiles("combined_antibiotics.txt")
