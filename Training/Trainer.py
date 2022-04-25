from Training.TrainUtils import train_one_epoch, test_one_epoch
from Layers.GraphModel import MMProtGraph
from config import Config
import time
import wandb
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr
import math
import os
import datetime

#Needs train_dataset, test_dataset

# config = Config()

# if config.wandb:
#     wandb.login()

# NUM_GRAPHS_PER_BATCH = config.model_args['batch_size']
# savepath = config.model_args['savepath']
# model = MMProtGraph(config.model_args).to(torch.double)
# device = config.device

# model.to(device)

# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0007,amsgrad=True)  
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.model_args['patience'])

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train_model(train_dataset,test_dataset):
    
    config = Config()

    if config.wandb:
        wandb.login()

    NUM_GRAPHS_PER_BATCH = config.model_args['batch_size']
    savepath = config.model_args['savepath']
    model = MMProtGraph(config.model_args).to(torch.double)
    model.apply(reset_weights)
    
    device = config.device

    model.to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007,amsgrad=True)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.model_args['patience'])

    
    
    t0 = time.time()
    train_loader = DataLoader(train_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, follow_batch=['x_s', 'x_t'])
    print(f'train dataloader time {time.time()-t0}')
    test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle= True,  follow_batch=['x_s', 'x_t'])
    if not os.path.exists(f'{savepath}/Model/'):
            os.makedirs(f'{savepath}/Model/')

    with wandb.init(project=config.project_name, config=config.model_args): #,mode="disabled"
        print(wandb.run.name)
        wandb_config = wandb.config
        #training
        best_loss = math.inf
        early_stopping_counter = 0
        wandb.log({'architecture':model})
        
        for epoch in range(config.model_args['epochs']):
            per_batch_loss = train_one_epoch(train_loader,model,loss_fn,optimizer,device)
            print(f'TRAINING: epoch {epoch} loss {per_batch_loss} time {datetime.datetime.now()}')
            wandb.log({"average loss train":per_batch_loss,"epoch":epoch})

            val_loss,all_pred, all_true_labels,_ = test_one_epoch(test_loader,model,loss_fn,device)
            print(f'VAL: Epoch {epoch}| Average Val Loss {val_loss} time {datetime.datetime.now()}')
            
            flat_pred = np.squeeze(np.array(all_pred))
            flat_label = np.squeeze(np.array(all_true_labels))
            print({"average loss test":val_loss,
                      "pearson correlation":pearsonr(flat_label,flat_pred)[0]})
            wandb.log({"average loss test":val_loss,
                      "pearson correlation":pearsonr(flat_label,flat_pred)[0]})
            scheduler.step(val_loss)    

            if float(val_loss)< best_loss:
                best_loss = val_loss

                torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_loss,
                }, f'{savepath}/Model/{wandb.run.name}_epoch_{epoch}_bestloss_{best_loss}.pt')
                early_stopping_counter=0
            else:
                early_stopping_counter+=1



            if early_stopping_counter>10:
                print("Early stopping due to no improvement.")
                print(f'Best Loss {best_loss}') 
                break
        print(model)
        return model
