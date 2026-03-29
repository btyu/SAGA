## Download Enamine dataset first (4.3M compounds)
https://enamine.net/compound-collections/screening-collection

# Process Enamine dataset
gdown 1KAqN4mNyXGVMMO1BGUwywVEV8T4lMKKS -O enamine_screening_collection.zip
unzip enamine_screening_collection.zip

# Convert to SMILES
obabel  Enamine_screening_collection_202508.sdf  -O enamine.smi -osmi

# 1. Clean the smiles file 
python clean_smiles.py enamine.smi enamine_clean.smi

# 2. Compute the druglikeness score for enamine_clean.smi
