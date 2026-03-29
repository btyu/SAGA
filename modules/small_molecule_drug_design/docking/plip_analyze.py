import os, sys
from pathlib import Path
from io import StringIO
import xml.etree.ElementTree as ET
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import logging
from functools import partial

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from plip.structure.preparation import PDBComplex, logger as PLIP_LOGGER
from plip.exchange.report import StructureReport
from plip.basic import config as PLIP_CONFIG

PLIP_LOGGER.setLevel(logging.ERROR)
PLIP_LOGGER.propagate = False


def plip_analyze_single_frame(
    pdbpath: os.PathLike, 
    mol_name: str ="MOL",
    add_hydrogen: bool = False,
    resnr_renum: Optional[Dict[int, int]] = None
) -> Dict[str, int]:
    """
    Analyze a single ligand-complex structure

    Parameters
    ----------
    pdbpath: os.PathLike
        Path to the pdbfile to be analyzed
    add_hydrogen: bool
        Whether to add hydrogen to the pdb structure. Default is False
    
    Return
    ------
    interact_count_frame: Dict[str, int]
        A dict with interaction type as the key, and the number of interaction as value
        The key is in the format "{name}/{restype}/{resnr}/{chain}". For example,
        'hydrophobic_interaction/ALA/123/A'
    """
    
    if add_hydrogen:
        PLIP_CONFIG.NOHYDRO = False
    else:
        PLIP_CONFIG.NOHYDRO = True
        
    pdb = PDBComplex()
    pdb.load_pdb(str(pdbpath))
    pdb.analyze()
    report = StructureReport(pdb)

    tmp = sys.stdout
    xmlstr = StringIO()
    sys.stdout = xmlstr
    report.write_xml(True)
    sys.stdout = tmp
    xmlstr.seek(0)

    xmlobj = ET.fromstring(xmlstr.read())

    binding_sites = xmlobj.findall("./bindingsite")
    try:
        bs = [bs for bs in binding_sites if bs.findall("identifiers/longname")[0].text == mol_name][0]
        itypes = bs.findall("interactions/")
    except IndexError: 
        itypes = []
        bs = [bs for bs in binding_sites]
        for site in bs:
            itypes += list(site.findall("interactions/"))
            
    interact_count_frame = {}
    for itype in itypes:
        for item in itype:
            name = item.tag
            restype = item.find("restype").text
            resnr = item.find("resnr").text
            if resnr_renum is not None:
                resnr = resnr_renum[int(resnr)]
            chain = item.find("reschain").text
            sig = f"{name}/{restype}/{resnr}/{chain}"
            cnt = interact_count_frame.get(sig, 0)
            if cnt == 0:
                interact_count_frame.update({sig: cnt+1})

    return interact_count_frame