import sys
sys.path.append('./../')
from config import Config
from functools import partial
from torch_geometric.data import Data,Dataset
import pandas as pd
import torch
import torch_geometric
import numpy as np 
import os
from tqdm import tqdm
import deepchem as dc
import esm
import pickle
import torch.nn.functional as F
from graphein.protein.edges.distance import (add_peptide_bonds,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,
                                             add_ionic_interactions,
                                             add_aromatic_interactions,
                                             add_aromatic_sulphur_interactions,
                                             add_hydrophobic_interactions,
                                             add_cation_pi_interactions,
                                             add_k_nn_edges
                                            )
from graphein.protein.config import ProteinGraphConfig, DSSPConfig
from graphein.protein.graphs import construct_graph
from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.features.nodes.amino_acid import expasy_protein_scale,meiler_embedding
from graphein.protein.features.nodes import aaindex
#TODO: write your own add_dssp_feature
from Preprocessing.DSSPFeaturizer import add_dssp_feature
from ast import literal_eval
# from graphein.protein.features.nodes.dssp import add_dssp_df, add_dssp_feature


print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

config = Config()



def identifier_nodes_dssp(df,pdb_df):
    '''df: unprocessed Biopython/graphein DSSP df
       graph_pdb_df: graphein g.graph['pdb_df']
    '''
#     print('identifdiwe nodes dssp')
    df.index.names = ['Index']
    node_id=[]
    for aa in df.index:
        splitted = aa.split(":")
        one_letter = splitted[1]
        node_id.append(splitted[0]+':'+splitted[2])
    df['chain_node_id']= node_id
    
    pdb_df['chain_node_id']=''
    for i in range(len(pdb_df)):
        splitted = pdb_df.iat[i,-2].split(":")
        pdb_df.iat[i,-1] = splitted[0]+':'+splitted[2]
        
    if len(df)!= len(pdb_df):
        m = pd.merge(pdb_df, df, how='outer', suffixes=('','_y'), indicator=True)
        rows_in_pdb_not_in_dssp = m[m['_merge']=='left_only'][pdb_df.columns]
        
        for i in range(len(rows_in_pdb_not_in_dssp)):
            df = df.append(pd.Series(), ignore_index=True)
            df.iat[-1+i,-1]= rows_in_pdb_not_in_dssp.iat[i,-1]

    if len(df)>=len(pdb_df):
        df_merged = pdb_df.merge(df, on ='chain_node_id',how='inner')
        
    else:
        df_merged = pdb_df.merge(df,on='chain_node_id',how='outer')
        df_merged.fillna(df_merged.mean(),inplace=True)
        
    return df_merged.loc[:,['asa','phi','psi','NH_O_1_relidx','NH_O_1_energy','O_NH_1_relidx',
                                                              'O_NH_1_energy','NH_O_2_relidx','NH_O_2_energy',
                                                              'O_NH_2_relidx','O_NH_2_energy']], df
     
from torch_geometric.data import Data
class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None,edge_attr_s=None, edge_index_t=None, x_t=None,
                 edge_attr_t=None,interaction_id=None,y=None,dist_mat_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_attr_s = edge_attr_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.edge_attr_t = edge_attr_t
#         if edge_index_t is None:
#             print(interaction_id)
#         idx_edge = edge_index_t.transpose(0,1).numpy()
#         self.edge_attr_t= dist_mat_t[(idx_edge[:,0], idx_edge[:,1])] 
        self.interaction_id = interaction_id
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
#             print(f'x_tsize {self.x_t.size()}')
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

#Pair Molecule dataset


class PairMoleculeDataset(Dataset):
    def __init__(self, root, filename, labelcol, subsample=None,transform=None, pre_transform=None, dssp= False,preprocessed_protgraph=False):
        self.filename = filename
        self.labelcol = labelcol
        self.device = config.device
        self.subsample = subsample
        self.dssp = dssp
        self.preprocessed_protgraph=preprocessed_protgraph
        self.failed_interactions=[]
        self.amino_acid_vocab = ['ALA','ARG','ASN','ASP','ASX','CYS','GLU','GLN','GLX','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER',
                    'THR','TRP','TYR','VAL']


        with open(config.esm_file,'rb') as f:
                self.esm_embedding = pickle.load(f)

        print('esm loaded')

        super(PairMoleculeDataset, self).__init__(root, transform, pre_transform)
        
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        if self.subsample:
            self.data = pd.read_csv(self.raw_paths[0]).head(self.subsample).reset_index()
        else:
            self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        
#         self.data.positions = self.data.positions.apply(literal_eval)
        print('csv read and filtered', len(self.data))
        datalist =   [f'data_{i}.pt' for i in list(self.data.Interaction_ID)]
        return datalist



    def download(self):
        pass
    
    def get_unique_prot_graphs(self,config):
        #calculate protein graphs for all unique pdbs in the data
        
        self.protgraph_dict = dict()
        self.failed_protgraph =[]
        
        configs = {
        "granularity": "CA",
        "keep_hets": False,
        "insertions": False,
        "verbose": False,
        "dssp_config": DSSPConfig(),
        "pdb_dir":f'{config.root}/{config.project_name}/data/prot/PDB/',# './data/prot/PDB/',
        "node_metadata_functions": [meiler_embedding,expasy_protein_scale],
        "edge_construction_functions": [add_peptide_bonds,
                                                  add_hydrogen_bond_interactions,
                                                  add_ionic_interactions,
                                                  add_aromatic_sulphur_interactions,
                                                  add_hydrophobic_interactions,
                                                  add_cation_pi_interactions,
                                                  partial(add_k_nn_edges, k=3, long_interaction_threshold=0)]
        }
        config = ProteinGraphConfig(**configs)
        format_convertor = GraphFormatConvertor('nx', 'pyg', 
                                verbose = 'all_info', 
                                columns = ['edge_index','meiler','coords','expasy','node_id','name','dist_mat','num_nodes'])
        
        
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0], desc='Calculating Protein Graphs'):
            protein_path = mol['path']
            pdbid = mol['pdbid']
            if (pdbid in self.protgraph_dict) or (pdbid in self.failed_protgraph):
                continue
                
            print(f'processing {pdbid} already processed {len(self.protgraph_dict)+len(self.failed_protgraph)} structures. Index {index}' )
            try:
                g = construct_graph(config=config, pdb_path=protein_path)

                protdata = format_convertor(g)
                aa_list=[]
                for aa in protdata.node_id:
                    aa_list.append(self.amino_acid_vocab.index(aa.split(':')[1]))

                aa_tensor = torch.Tensor(aa_list).to(torch.int64)
                aa_one_hot = F.one_hot(aa_tensor, num_classes=len(self.amino_acid_vocab))
                aa_coords = torch.Tensor(protdata.coords[0])

    #             print(f'aa_list {len(aa_list)}, g nodes {protdata.num_nodes}, aa_coords {aa_coords.shape}, aa_one_hot {aa_one_hot.shape}')
                concatenated_3d_onehot_amino_acid= torch.cat((aa_coords,aa_one_hot),dim=-1)

                #prot node features provided with graphein -expasy, meiler and dssp
                ex_df = pd.DataFrame(protdata.expasy)
                meiler_df = pd.DataFrame(protdata.meiler)
                if self.dssp:
                    add_dssp_feature(g,feature=['asa','phi'])
                    dssp_df =  g.graph["dssp_df"].loc[:,['asa','phi','psi','NH_O_1_relidx','NH_O_1_energy','O_NH_1_relidx',
                                                     'O_NH_1_energy','NH_O_2_relidx','NH_O_2_energy',
                                                     'O_NH_2_relidx','O_NH_2_energy']]
                    dssp_df, dssp_unprocessed = identifier_nodes_dssp(dssp_df,g.graph['pdb_df'])
                    print('dssp_df',dssp_df.shape)
                    prot_node_feats_graphein = np.concatenate([ex_df.values,meiler_df.values,dssp_df.values],axis=1)

                else:
                    prot_node_feats_graphein = np.concatenate([ex_df.values,meiler_df.values],axis=1)

                prot_node_feats_graphein = torch.Tensor(prot_node_feats_graphein)                
                prot_node_feats = torch.cat((concatenated_3d_onehot_amino_acid,prot_node_feats_graphein),dim=-1)

                edge_index_t,x_t= protdata.edge_index, prot_node_feats

                idx_edge = edge_index_t.transpose(0,1).numpy()
                dist_mat_t= protdata.dist_mat[0].to_numpy()
                edge_attr_t= dist_mat_t[(idx_edge[:,0], idx_edge[:,1])] 
                self.protgraph_dict[pdbid]= {'x_t':x_t,   #x_t[:,-11:] are the DSSP features
                                       'edge_index_t':edge_index_t,
                                       'edge_attr_t':edge_attr_t,
                                       }
            except Exception as e:
                print(e,pdbid)
                self.failed_protgraph.append(pdbid)
                continue
            
        return self.protgraph_dict

    def process(self):
        #Note to self: Would be nice to have target name in data.target_name.
        #For this create the input csv such that there is a target name corresponding to each prot seq. 
        #Also have unique identifiers for each protein sequence

        #self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        print('processessing....')

        if self.preprocessed_protgraph:
            with open(f'{config.root}/{config.project_name}/protgraph_dict.pkl','rb') as f:
                protgraph_dict = pickle.load(f)
        else:
            protgraph_dict = self.get_unique_prot_graphs(config)
            with open(f'{config.root}/{config.project_name}/protgraph_dict.pkl','wb') as f:
                pickle.dump(protgraph_dict,f)
        
        protgraph_fileterd_df = self.data[self.data.pdbid.isin(protgraph_dict.keys())]


        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True,use_chirality=True,use_partial_charge=True)
        
            
            
        for index, mol in tqdm(protgraph_fileterd_df.iterrows(), total=protgraph_fileterd_df.shape[0]):
            # Featurize molecule
            try:
#                 supplier = Chem.SDMolSupplier(mol["smile_path"], removeHs=True)
#                 rdkit_mol = supplier[0]
                f = featurizer.featurize(mol['SMILES'])
                interaction_id = mol['Interaction_ID']

#                 pos_tensor = torch.Tensor(mol["positions"])

                smilesdata = f[0].to_pyg_graph()
                edge_index_s = smilesdata.edge_index
                x_s = smilesdata.x #torch.cat((pos_tensor,smilesdata.x),dim=-1)
                edge_attr_s = smilesdata.edge_attr
                protein_path = mol['path']
                pdb_id = mol['pdbid']
                protgraph= protgraph_dict[pdb_id]

                x_t,edge_index_t,edge_attr_t= protgraph['x_t'],protgraph['edge_index_t'],protgraph['edge_attr_t']

                y = self._get_label(mol[self.labelcol])
                data = PairData(edge_index_s,x_s,edge_attr_s,edge_index_t,x_t,edge_attr_t,interaction_id, y)


                data.prot_esm = self.esm_embedding[mol["pdbid"]]

    #                 print(f'saving data {self.processed_dir}')
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{data.interaction_id}.pt'))
    #                 print('save finished')

            except Exception as e:
                print(e,interaction_id)
                continue



    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        interaction_id = self.data.Interaction_ID[idx]
        data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{interaction_id}.pt'))     

        return data



#############################################


# import sys
# sys.path.append('./../')
# from config import Config
# from torch_geometric.data import Data,Dataset
# import pandas as pd
# import torch
# import torch_geometric
# import numpy as np 
# import os
# from tqdm import tqdm
# import deepchem as dc
# import esm
# import pickle
# import torch.nn.functional as F
# from graphein.protein.edges.distance import (add_peptide_bonds,
#                                              add_hydrogen_bond_interactions,
#                                              add_disulfide_interactions,
#                                              add_ionic_interactions,
#                                              add_aromatic_interactions,
#                                              add_aromatic_sulphur_interactions,
#                                              add_hydrophobic_interactions,
#                                              add_cation_pi_interactions
#                                             )
# from graphein.protein.config import ProteinGraphConfig, DSSPConfig
# from graphein.protein.graphs import construct_graph
# from graphein.ml.conversion import GraphFormatConvertor
# from graphein.protein.features.nodes.amino_acid import expasy_protein_scale,meiler_embedding
# from graphein.protein.features.nodes import aaindex
# #TODO: write your own add_dssp_feature
# from Preprocessing.DSSPFeaturizer import add_dssp_feature
# # from graphein.protein.features.nodes.dssp import add_dssp_df, add_dssp_feature


# print(f"Torch version: {torch.__version__}")
# print(f"Cuda available: {torch.cuda.is_available()}")
# print(f"Torch geometric version: {torch_geometric.__version__}")

# config = Config()

        
# from torch_geometric.data import Data
# class PairData(Data):
#     def __init__(self, edge_index_s=None, x_s=None,edge_attr_s=None, edge_index_t=None, x_t=None,
#                  edge_attr_t=None,interaction_id=None,y=None,dist_mat_t=None):
#         super().__init__()
#         self.edge_index_s = edge_index_s
#         self.x_s = x_s
#         self.edge_attr_s = edge_attr_s
#         self.edge_index_t = edge_index_t
#         self.x_t = x_t
#         self.edge_attr_t = edge_attr_t
# #         if edge_index_t is None:
# #             print(interaction_id)
# #         idx_edge = edge_index_t.transpose(0,1).numpy()
# #         self.edge_attr_t= dist_mat_t[(idx_edge[:,0], idx_edge[:,1])] 
#         self.interaction_id = interaction_id
#         self.y = y

#     def __inc__(self, key, value, *args, **kwargs):
#         if key == 'edge_index_s':
#             return self.x_s.size(0)
#         if key == 'edge_index_t':
# #             print(f'x_tsize {self.x_t.size()}')
#             return self.x_t.size(0)
#         else:
#             return super().__inc__(key, value, *args, **kwargs)

# #Pair Molecule dataset


# class PairMoleculeDataset(Dataset):
#     def __init__(self, root, filename, labelcol, subsample=None,transform=None, pre_transform=None, dssp=  True,preprocessed_protgraph=False):
#         self.filename = filename
#         self.labelcol = labelcol
#         self.device = config.device
#         self.subsample = subsample
#         self.dssp = dssp
#         self.preprocessed_protgraph=preprocessed_protgraph
#         self.failed_interactions=[]
#         self.amino_acid_vocab = ['ALA','ARG','ASN','ASP','ASX','CYS','GLU','GLN','GLX','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER',
#                     'THR','TRP','TYR','VAL']


#         with open(config.esm_file,'rb') as f:
#                 self.esm_embedding = pickle.load(f)

#         print('esm loaded')

#         super(PairMoleculeDataset, self).__init__(root, transform, pre_transform)
        
        
#     @property
#     def raw_file_names(self):
#         """ If this file exists in raw_dir, the download is not triggered.
#             (The download func. is not implemented here)  
#         """
#         return self.filename

#     @property
#     def processed_file_names(self):
#         """ If these files are found in raw_dir, processing is skipped"""
#         if self.subsample:
#             self.data = pd.read_csv(self.raw_paths[0]).head(self.subsample).reset_index()
#         else:
#             self.data = pd.read_csv(self.raw_paths[0]).reset_index()

#         print('csv read and filtered', len(self.data))
#         datalist =   [f'data_{i}.pt' for i in list(self.data.Interaction_ID)]
#         return datalist



#     def download(self):
#         pass
    
#     def get_unique_prot_graphs(self,config):
#         #calculate protein graphs for all unique pdbs in the data
        
#         self.protgraph_dict = dict()
#         self.failed_protgraph =[]
        
#         configs = {
#         "granularity": "CA",
#         "keep_hets": False,
#         "insertions": False,
#         "verbose": False,
#         "dssp_config": DSSPConfig(),
#         "pdb_dir":f'{config.root}/{config.project_name}/data/prot/PDB/',# './data/prot/PDB/',
#         "node_metadata_functions": [meiler_embedding,expasy_protein_scale],
#         "edge_construction_functions": [add_peptide_bonds,
#                                                   add_hydrogen_bond_interactions,
#                                                   add_ionic_interactions,
#                                                   add_aromatic_sulphur_interactions,
#                                                   add_hydrophobic_interactions,
#                                                   add_cation_pi_interactions]
#         }
#         config = ProteinGraphConfig(**configs)
#         format_convertor = GraphFormatConvertor('nx', 'pyg', 
#                                 verbose = 'all_info', 
#                                 columns = ['edge_index','meiler','coords','expasy','node_id','name','dist_mat','num_nodes'])
        
        
#         for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0], desc='Calculating Protein Graphs'):
#             protein_path = mol['path']
#             pdbid = mol['pdbid']
#             if (pdbid in self.protgraph_dict) or (pdbid in self.failed_protgraph):
#                 continue
                
#             print(f'processing {pdbid} already processed {len(self.protgraph_dict)+len(self.failed_protgraph)} structures. Index {index}' )
#             try:
#                 g = construct_graph(config=config, pdb_path=protein_path)

#                 protdata = format_convertor(g)
#                 aa_list=[]
#                 for aa in protdata.node_id:
#                     aa_list.append(self.amino_acid_vocab.index(aa.split(':')[1]))

#                 aa_tensor = torch.Tensor(aa_list).to(torch.int64)
#                 aa_one_hot = F.one_hot(aa_tensor, num_classes=len(self.amino_acid_vocab))
#                 aa_coords = torch.Tensor(protdata.coords[0])

#     #             print(f'aa_list {len(aa_list)}, g nodes {protdata.num_nodes}, aa_coords {aa_coords.shape}, aa_one_hot {aa_one_hot.shape}')
#                 concatenated_3d_onehot_amino_acid= torch.cat((aa_coords,aa_one_hot),dim=-1)

#                 #prot node features provided with graphein -expasy, meiler and dssp
#                 ex_df = pd.DataFrame(protdata.expasy)
#                 meiler_df = pd.DataFrame(protdata.meiler)
#                 if self.dssp:
#                     add_dssp_feature(g,feature=['asa','phi'])
#                     dssp_df =  g.graph["dssp_df"].loc[:,['asa','phi','psi','NH_O_1_relidx','NH_O_1_energy','O_NH_1_relidx',
#                                                      'O_NH_1_energy','NH_O_2_relidx','NH_O_2_energy',
#                                                      'O_NH_2_relidx','O_NH_2_energy']]
#                     print('dssp_df',dssp_df.shape)
#                     prot_node_feats_graphein = np.concatenate([ex_df.values,meiler_df.values,dssp_df.values],axis=1)

#                 else:
#                     prot_node_feats_graphein = np.concatenate([ex_df.values,meiler_df.values],axis=1)

#                 prot_node_feats_graphein = torch.Tensor(prot_node_feats_graphein)                
#                 prot_node_feats = torch.cat((concatenated_3d_onehot_amino_acid,prot_node_feats_graphein),dim=-1)

#                 edge_index_t,x_t= protdata.edge_index, prot_node_feats

#                 idx_edge = edge_index_t.transpose(0,1).numpy()
#                 dist_mat_t= protdata.dist_mat[0].to_numpy()
#                 edge_attr_t= dist_mat_t[(idx_edge[:,0], idx_edge[:,1])] 
#                 self.protgraph_dict[pdbid]= {'x_t':x_t,
#                                        'edge_index_t':edge_index_t,
#                                        'edge_attr_t':edge_attr_t,
#                                        }
#             except Exception as e:
#                 print(e,pdbid)
#                 self.failed_protgraph.append(pdbid)
#                 continue
            
#         return self.protgraph_dict

#     def process(self):
#         #Note to self: Would be nice to have target name in data.target_name.
#         #For this create the input csv such that there is a target name corresponding to each prot seq. 
#         #Also have unique identifiers for each protein sequence

#         #self.data = pd.read_csv(self.raw_paths[0]).reset_index()
#         print('processessing....')

#         if self.preprocessed_protgraph:
#             with open(f'{config.root}/{config.project_name}/protgraph_dict.pkl','rb') as f:
#                 protgraph_dict = pickle.load(f)
#         else:
#             protgraph_dict = self.get_unique_prot_graphs(config)
#             with open(f'{config.root}/{config.project_name}/protgraph_dict.pkl','wb') as f:
#                 pickle.dump(protgraph_dict,f)
        
#         protgraph_fileterd_df = self.data[self.data.pdbid.isin(protgraph_dict.keys())]


#         featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True,use_chirality=True,use_partial_charge=True)
        
            
            
#         for index, mol in tqdm(protgraph_fileterd_df.iterrows(), total=protgraph_fileterd_df.shape[0]):
#             # Featurize molecule
#             try:
#                 f = featurizer.featurize(mol["SMILES"])
#                 interaction_id = mol['Interaction_ID']
#                 smilesdata = f[0].to_pyg_graph()
#                 edge_index_s = smilesdata.edge_index
#                 x_s = smilesdata.x
#                 edge_attr_s = smilesdata.edge_attr
#                 protein_path = mol['path']
#                 pdb_id = mol['pdbid']
#                 protgraph= protgraph_dict[pdb_id]

#                 x_t,edge_index_t,edge_attr_t= protgraph['x_t'],protgraph['edge_index_t'],protgraph['edge_attr_t']
                
#                 y = self._get_label(mol[self.labelcol])
#                 data = PairData(edge_index_s,x_s,edge_attr_s,edge_index_t,x_t,edge_attr_t,interaction_id, y)


#                 data.prot_esm = self.esm_embedding[mol["pdbid"]]

# #                 print(f'saving data {self.processed_dir}')
#                 torch.save(data, 
#                     os.path.join(self.processed_dir, 
#                                  f'data_{data.interaction_id}.pt'))
#     #                 print('save finished')

#             except Exception as e:
#                 print(f'there is an exception in {protein_path, interaction_id, e}')
#                 if not (str(e) =="'numpy.ndarray' object has no attribute 'to_pyg_graph'"):
#                     try:
#                         g = construct_graph(pdb_path=protein_path, config=config)
#                     except Exception as e:
#                         print(e,mol["Interaction_ID"],pdb_id)
#                         self.failed_interactions.append(mol['Interaction_ID'])
#                         continue
#                 continue


#     def _get_label(self, label):
#         label = np.asarray([label])
#         return torch.tensor(label, dtype=torch.float64)

#     def len(self):
#         return self.data.shape[0]

#     def get(self, idx):
#         """ - Equivalent to __getitem__ in pytorch
#             - Is not needed for PyG's InMemoryDataset
#         """
#         interaction_id = self.data.Interaction_ID[idx]
#         data = torch.load(os.path.join(self.processed_dir, 
#                                  f'data_{interaction_id}.pt'))     

#         return data
