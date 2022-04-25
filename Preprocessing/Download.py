from Bio.PDB import PDBList
import os
import wget
import sys
sys.path.append('./../')
from config import Config

config= Config()

def download_prot_structure(code, struct_type='pdb', verbose=True):
    """
    Download PDB structure from PDB or AF
    :param code: 4 character PDB accession code if struct_type ='pdb'
                 uniprot code if struct_type='af'
    :type pdb_code: str:
    :struct_type: 'pdb' or 'af'
    :return: returns filepath to downloaded structure
    :rtype: str
    """
        
    if struct_type =='pdb':
        if not os.path.exists(config.pdb_dir):
            os.makedirs(config.pdb_dir)
        pdb_code = code
        pdbl = PDBList(verbose=verbose)
        pdbl.retrieve_pdb_file(pdb_code, pdir=config.pdb_dir, overwrite=True,file_format="pdb")
        # Rename file to .pdb from .ent
        os.rename(
            f'{config.pdb_dir}pdb{pdb_code.lower()}.ent',
            f'{config.pdb_dir}{pdb_code.lower()}.pdb',
        )
        # Assert file has been downloaded
        assert any(pdb_code.lower() in s for s in os.listdir(config.pdb_dir))
        if verbose:
            print(f"Downloaded PDB file for: {pdb_code}")
        return f'{config.pdb_dir}{pdb_code.lower()}.pdb'
    
    elif struct_type=='af':
        if not os.path.exists(config.af_dir):
            os.makedirs(config.af_dir)
        uniprot_id = code
        config.base_url = "https://alphafold.ebi.ac.uk/files/"
        query_url = config.base_url + "AF-" + uniprot_id + "-F1-model_v2.pdb"
        try:
            structure_filename = wget.download(query_url, out=config.af_dir)
        except Exception as e:
            query_url = config.base_url + "AF-" + uniprot_id + "-F1-model_v1.pdb"
            try:
                structure_filename = wget.download(query_url, out=config.af_dir)
            except Exception as e:
                print('unavailable uniprot ',uniprot_id)
                return
        return structure_filename
