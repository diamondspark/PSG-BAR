#!/usr/bin/env python
# coding: utf-8

# In[3]:


# from config import Config
from Utils import pdb_to_uniprot
from Preprocessing.Download import download_prot_structure
from Preprocessing.ESM import get_esm


# In[4]:


pdb_id = '1BTE'


# # Download PDB Structure for given pdb_id

# In[6]:


pdb_path = download_prot_structure(pdb_id,struct_type='pdb')
print(pdb_path) 


# # Download Alphafold Structure for given pdb_id

# In[7]:


af_path = download_prot_structure(code = pdb_to_uniprot(pdb_id),struct_type='af')
print(af_path)


# # Calculate ESM for proteins in dataset

# In[11]:


from Bio import SeqIO
import torch
esm=[]
for record in SeqIO.parse(pdb_path, "pdb-atom"):
    protein_seq = str(record.seq).replace('X','')
    esm.append(get_esm(protein_seq,len(protein_seq)>1022, representation_layer=6).detach().cpu())
esm_embed = torch.mean(torch.stack(esm),dim=0)
print(pdb_id, esm_embed.shape)


# In[ ]:




