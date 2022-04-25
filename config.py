import torch
import os 
import shutil

class Config():
    def __init__(self):
        self.root = './'
        self.project_name = 'psgbar_expt_pdbbind'
        if not os.path.exists(f'{self.root}/{self.project_name}'):
            os.makedirs(f'{self.root}/{self.project_name}')
            os.makedirs(f'{self.root}/{self.project_name}/raw')
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
        self.pdb_dir = f'{self.root}/{self.project_name}/data/prot/PDB/'
        self.af_dir = f'{self.root}/{self.project_name}/data/prot/AF/'
        self.base_url = "https://alphafold.ebi.ac.uk/files/"
        self.esm_file = f'{self.root}/{self.project_name}/esm_file.pkl'
#         self.esm_file = '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Drug_Protein_Interaction_Project1/ER_AR_project/data/PDBBind/esm_small_pdbbind_averaged_chain.pkl'
        self.wandb = True
        self.download_structures=False
        self.create_dataset = False

        self.model_args = {   'seed':42,
                              'savepath':f'{self.root}/{self.project_name}/',
                              'node_n_feat':33,
                              'batch_size':128,
                               'scheduler_gamma': 0.8,
                              'smiles_node_embed_dim':500,
                              'smiles_graph_encoder_dim':500,
                              'protein_encoder_dim':500,
                              'concat_linear_layer':512,
                              'patience':5,
                              'epochs':5000
                          }