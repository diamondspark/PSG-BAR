import sys
sys.path.append('./../')
from config import Config                
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
import esm
import torch
from tqdm import tqdm
from Bio import SeqIO


config= Config()
esm_model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
device = config.device

esm_model = esm_model.to(device)
       
print(f"Transferred model to {config.device}")
        
esm_model.eval()
batch_converter = alphabet.get_batch_converter()


def get_esm(data,long_seq, representation_layer):
    '''data: protein seq: str
       long_seq: Boolean >1022
    '''
    if long_seq:
        b = [data[i:i+1022] for i in range(len(data)-1021)]  #sliding window to get 1022 amino acids sequences
        data = [subprot for subprot in enumerate(b)]  
        chunks = [data[x:x+1] for x in range(0, len(data), 1)]  #chunks is creating batches of size 50
        sequence_representations = []
        for data in chunks:
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            with torch.no_grad():
                results = esm_model(batch_tokens.to(device), repr_layers=[representation_layer], return_contacts=True)
                token_representations = results["representations"][representation_layer].cpu().detach()
            for i, (_, seq) in enumerate(data):
                sequence_representations.extend([token_representations[i, 1 : len(seq) + 1].mean(0)])


    else:
        data = [(0,data)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = esm_model(batch_tokens.to(device), repr_layers=[representation_layer], return_contacts=True)
            token_representations = results["representations"][representation_layer]

        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
            
    mean = torch.mean(torch.stack(sequence_representations),dim = 0)
    return mean

def get_protein_esm(prot_path):
    '''Get ESM embedding from a pdb path or AF path.
       Average ESM for all the chains in a given PDB/AF structure
    '''
    esm=[]
    for record in SeqIO.parse(prot_path, "pdb-atom"):
        protein_seq = str(record.seq).replace('X','')
        esm.append(get_esm(protein_seq,len(protein_seq)>1022, representation_layer=6).detach().cpu())
    esm_embed = torch.mean(torch.stack(esm),dim=0)
    return esm_embed