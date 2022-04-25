import requests
import re

def pdb_to_uniprot(pdbid):
    url_template = "http://www.rcsb.org/pdb/files/{}.pdb"
    protein = pdbid
    url = url_template.format(protein)
    response = requests.get(url)
    pdb = response.text

    m = re.search('UNP\ +(\w+)', pdb)
    return m.group(1)
