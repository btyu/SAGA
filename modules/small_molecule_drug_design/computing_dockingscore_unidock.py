import sys
sys.path.insert(0, "/home/tl688/pitl688/scileoagent_drug/SciLeoAgent/")
sys.path.insert(0, "/home/tl688/pitl688/scileoagent_drug/SciLeoAgent/scileo_agent")
from scileo_agent.core.registry import register_scorer_class, register_scorer
from scileo_agent.core.data_models import Candidate
from typing import List, Optional
from rdkit import Chem
from dataclasses import dataclass
from typing import Tuple
from modules.small_molecule_drug_design.docking import unidock_draft
import numpy as np

from scorer.unidock_scorer import *
scorer = UniDockScorers()


from scileo_agent.core.data_models import Population, Objective, Candidate, ObjectiveIndex

from llm_sbdd_optimizer import LLMSBDDOptimizer
from scileo_agent.core.config import LLMConfig
"""Create and configure the optimizer."""
default_config = {
    "population_size": 120,
    "offspring_size": 70,
    "mutation_size": 30,
    "oracle_budget": 10000,
    "seed": 42
}

# LLM configuration
llm_config = LLMConfig(provider="openai", model="gpt-4o-mini")
dds_sample =LLMSBDDOptimizer(module_id="llm_sbdd_optimizer",
                        config=default_config,
                        llm_config=llm_config)

import pandas as pd
mol_list = pd.read_csv("../../logs/mpro_enamine_top120_seed6_20250918_091259/mpro_enamine_top120_seed6_20250918_091259_selected_top10000_diverse.csv")
smile_list = list(mol_list['smiles'].values)

print(len(smile_list))

candidates = [
    Candidate(representation=dds_sample._sanitize_smiles_value(smiles))
    for smiles in smile_list
]
test_mol = Population(candidates=candidates)



protein_file = "./data/pdb/MPRO.pdb"
prot_tar = ProteinTarget(  protein_name="MPRO",
                pdb_path=protein_file,
                pocket_center=(9.050, 8.898, -1.508),)
score_out = scorer._score_unidock_nodelete(test_mol,prot_tar)

