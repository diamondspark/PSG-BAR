import torch.nn as nn
import torch
from Layers.GraphAttention_w_residualconnection import GATConv


class MMProtGraph(nn.Module):
    def __init__(self,args):
        super(MMProtGraph,self).__init__()
        torch.manual_seed(args['seed'])
        
        self.smilesEncoder = SmilesEncoder(args)
        #Interaction attention
        self.interaction_attn = Attention_vis(xt_features=93)
        self.proteinEncoder = ProteinEncoder(args)
        
        self.linear1 = nn.Linear(args['smiles_graph_encoder_dim']+args['protein_encoder_dim']+768,
                                 1024)
        self.linear2 = nn.Linear(1024,1024)
        self.linear3 = nn.Linear(1024,512)
#         self.bn1 = nn.BatchNorm1d(args['concat_linear_layer'])
        self.output = nn.Linear(512,1)
        
    def forward(self,data):
        x_smile = self.smilesEncoder(data)
        alpha, attn_score_per_graph = self.interaction_attn(data,x_smile)
        x_prot= self.proteinEncoder(data,alpha)
      
        batch_size = len(data.interaction_id)
        x = data.prot_esm.reshape(batch_size,-1).double()
        x_concat = torch.cat([x_prot,x_smile,x],dim = -1)
#         print('x_concat',x_concat.shape)
        hidden1 = self.linear1(x_concat)
        hidden1 = nn.functional.leaky_relu(hidden1)
        hidden2 = nn.functional.leaky_relu(self.linear2(hidden1))
        hidden3 = nn.functional.leaky_relu(self.linear3(hidden2))
#         hidden1 = self.bn1(hidden1)
        
        return self.output(hidden3), attn_score_per_graph
    

from torch_geometric import nn as pygnn
from torch import nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
    
class SmilesEncoder(nn.Module):
    def __init__(self,args):
        super(SmilesEncoder,self).__init__()
        torch.manual_seed(args['seed'])
        
        embedding_size = args['smiles_node_embed_dim']
        self.initial_conv = GATConv(args['node_n_feat'],embedding_size,edge_dim = 11, residual= not False)
        self.conv1 = GATConv(embedding_size,embedding_size,dropout = 0.0,edge_dim = 11, residual=not False)
        self.conv2 = GATConv(embedding_size,embedding_size,dropout = 0.0,edge_dim = 11, residual= not False)
        self.conv3 = GATConv(embedding_size,embedding_size,dropout = 0.0,edge_dim = 11, residual= not False)

        # Output layer
        self.out = nn.Linear(embedding_size*2, args['smiles_graph_encoder_dim'])
        
    def forward(self,data):
        #GCNConv can't use edge features. See here https://github.com/FluxML/GeometricFlux.jl/issues/77
        x, edge_index, batch_index, edge_attr = data.x_s.double(), data.edge_index_s, data.x_s_batch, data.edge_attr_s.double()
#         print('after',batch_index.dtype,batch_index.device)
#         print(x.shape,edge_index.shape,edge_attr.shape)
        hidden = self.initial_conv(x,edge_index,edge_attr=edge_attr)
        hidden = nn.functional.leaky_relu(hidden)
        
        hidden = self.conv1(hidden, edge_index,edge_attr=edge_attr)
        hidden = nn.functional.leaky_relu(hidden)
        hidden = self.conv2(hidden, edge_index,edge_attr=edge_attr)
        hidden = nn.functional.leaky_relu(hidden)
        hidden = self.conv3(hidden, edge_index,edge_attr=edge_attr)
        
        
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)
        hidden = nn.functional.leaky_relu(hidden)
        
        return self.out(hidden)
    

from torch_geometric.utils import softmax    
    
class Attention_vis(nn.Module):
    def __init__(self,xt_features=93):
        super(Attention_vis,self).__init__()
        self.linearW1 = nn.Linear(150*2,150)
        self.linearW2 = nn.Linear(150,1)
        self.linearU1 = nn.Linear(500, 150)
        self.linearU2 = nn.Linear(xt_features,150)
        
    def forward(self, data, smile_latent):
        U1_xs = self.linearU1(smile_latent)
        U2_xt = self.linearU2(data.x_t.double())
     
        _, counts=torch.unique_consecutive(data.x_t_batch,return_counts=True)
        U1_xs = torch.repeat_interleave(U1_xs, counts, dim=0)
#         print('NEW U1_xs.shape,U2_xt.shape',U1_xs.shape,U2_xt.shape)
        
#         V = nn.functional.leaky_relu(U1_xs+U2_xt)
        V = torch.cat([U1_xs,U2_xt],dim=-1)
#         print('V',V.shape)
        W1V = torch.tanh(self.linearW1(V))
#         print('W1V',W1V.shape)
        ei = self.linearW2(W1V)
#         print('ei',ei.shape)
        
        #return ei for each node in the graph for all graphs in the batch 
        #Taking softmax is an issue: how do we take softmax over a tuple which has variable sized entries (each graph in tuple has different number of nodes)
        #Since this attention calculation is only for visualization, we skip taking softmax.
        #update: can take graph level softmax by using torch_geometric.utils.softmax, so using it now
        alpha = softmax(ei,data.x_t_batch)
#         print('softmax alpha',alpha.shape,torch.sum(alpha))
        
        attn_score_per_graph = torch.split(alpha,counts.tolist()) 
#         for i,graph in enumerate(alpha_graphs_per_batch):
#             print(i,graph.shape)
#         print(alpha)
        return alpha, attn_score_per_graph
            

class ProteinEncoder(nn.Module):
    def __init__(self, args):
        super(ProteinEncoder,self).__init__()
        torch.manual_seed(args['seed'])
        
        embedding_size = args['smiles_node_embed_dim']
        xt_features=93#93+11(dssp) #TODO if args.dssp: 104, else 93
        self.initial_conv = GATConv(xt_features,embedding_size, residual= not False)
        self.conv1 = GATConv(embedding_size,embedding_size,dropout = 0.1, residual=not False)
        self.conv2 = GATConv(embedding_size,embedding_size,dropout = 0.1, residual= not False)
        self.conv3 = GATConv(embedding_size,embedding_size,dropout = 0.1, residual= not False)
        
#         #Interaction attention
#         self.interaction_attn = Attention_vis(xt_features)

        # Output layer
        self.out = nn.Linear(embedding_size*1, args['protein_encoder_dim'])
        
    def forward(self,data,alpha):#latent_smile_repr):
        #GCNConv can't use edge features. See here https://github.com/FluxML/GeometricFlux.jl/issues/77
        x, edge_index, batch_index,attentions = data.x_t.double(), data.edge_index_t, data.x_t_batch,[]
#         p_coords = [torch.from_numpy(coords[0]).double() for coords in protein_graph_batch.coords]
#         x,edge_index,batch_index = torch.cat(p_coords).to(device),protein_graph_batch.edge_index.to(device),protein_graph_batch.batch

#         print('after',batch_index.dtype,batch_index.device)
#         print(x.shape,edge_index.shape,edge_attr.shape)
        hidden,attn1 = self.initial_conv(x,edge_index,return_attention_weights=True)
        attentions.append(attn1)
        hidden = nn.functional.leaky_relu(hidden)
        
        hidden,attn2 = self.conv1(hidden, edge_index,return_attention_weights=True)
        attentions.append(attn2)
        hidden = nn.functional.leaky_relu(hidden)
        
        hidden,attn3 = self.conv2(hidden, edge_index,return_attention_weights=True)
        attentions.append(attn3)
        hidden = nn.functional.leaky_relu(hidden)
        
        hidden,attn4 = self.conv3(hidden, edge_index,return_attention_weights=True)
        attentions.append(attn4)
        
        #Interaction Attention scores
#         alpha, attn_score_per_graph = self.interaction_attn(data,latent_smile_repr)
        #attention weighted node embedding for protein graph
#         print('conv3,alpha', hidden.shape,alpha.shape)
        attn_scored_hidden = torch.mul(alpha,hidden)
        
        # Global Pooling (stack different aggregations)
        hidden = gmp(attn_scored_hidden, batch_index)
        
#         # Global Pooling (stack different aggregations)
#         hidden = torch.cat([gmp(attn_scored_hidden, batch_index), 
#                             gap(attn_scored_hidden, batch_index)], dim=1)
        hidden = nn.functional.leaky_relu(hidden)
        
        return self.out(hidden)