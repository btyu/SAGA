import csv
import os

from pathlib import Path

def extract_smiles(input_csv, output_txt):
    with open(input_csv, 'r') as infile, open(output_txt, 'w') as outfile:
        reader = csv.DictReader(infile)
        for row in reader:
            smiles = row.get('smiles')
            if smiles:
                outfile.write(smiles.strip() + '\n')

if __name__ == '__main__':
    current_dir = Path(__file__).parent
    smi_file = str(current_dir.parent / "data/molecules/zinc_250k.csv")
    output_file = str(current_dir.parent / "data/molecules/zinc_250k.txt")
    extract_smiles(smi_file, output_file)