# Ring score scorer MCP module

# --- auto-generated scorers start ---
scorers: dict = {
    "ring_score": {
        "function_name": "score_ring_score",
        "type": "candidate-wise",
        "description": "Ring system frequency score (value range: 0.0 to 1.0). This score identifies unusual or 'weird' ring systems by comparing them to ChEMBL database frequencies. Ring systems are extracted from molecules by cleaving linker bonds, and each ring system's frequency in ChEMBL is looked up. The score uses the minimum frequency (most unusual ring) to assess overall ring system quality. High scores (>0.8) indicate molecules with no rings (no penalty) or reasonably common ring systems (frequency ≥100) that are frequently found in known drugs and drug-like compounds, while low scores (<0.3) suggest molecules containing very rare or novel ring systems (frequency ≤10) that may have unknown properties or synthetic challenges. A score of 0.0 indicates the presence of ring systems not found in the ChEMBL database, which may represent highly novel or problematic structures. This metric helps identify compounds with unusual ring systems that may require additional evaluation or optimization.",
        "tool_description": "Calculate ring system frequency score.\n\nReturns normalized scores between 0.0 and 1.0 based on ring system frequencies:\n- Score = 1.0: No rings found (no penalty) or all ring systems have frequency ≥100 (reasonably common)\n- Score = 0.0: Contains ring systems with frequency ≤10 or not in database (very rare/weird)\n- Score = linear interpolation: Frequency between 10-100 on log scale (only penalize reasonably rare rings)\n\nThe score identifies unusual ring systems by extracting all ring systems from the molecule\nand looking up their frequency in ChEMBL. The minimum frequency (most unusual ring) is used\nto assess overall ring system quality.\n\nArgs:\n    samples: List of input samples, where each sample is a SMILES string of a molecule\n\nReturns:\n    List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations",
    },
}
# --- auto-generated scorers end ---
