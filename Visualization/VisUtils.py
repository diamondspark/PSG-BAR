#Visualization1 : correlation plot
from Training.TrainUtils import test_one_epoch
from torch_geometric.loader import DataLoader
import torch
from Layers.GraphModel import MMProtGraph
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def correlation_plot(dataset,config,**kwargs):
    '''kwargs= {model:preloaded model, model_file}
    '''
    test_loader = DataLoader(dataset, batch_size=config.model_args['batch_size'], shuffle= False,  follow_batch=['x_s', 'x_t'])
    device =  config.device
    if 'model' in kwargs:
        model = kwargs['model']
    else:
        model = MMProtGraph(config.model_args).to(torch.double)
        checkpoint = torch.load(config.model_args['savepath']+'Model/'+kwargs['model_file'],map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    loss_fn = torch.nn.MSELoss()
    val_loss,all_pred, all_true_labels, _ = test_one_epoch(test_loader,model,loss_fn,device,return_attn=False)
    print(f'val loss {val_loss}')
    all_pred_list = list(np.stack( all_pred, axis=0 ).squeeze())
    all_true_labels_list = list(np.stack(all_true_labels,axis=0).squeeze())
    df_pred = pd.DataFrame(list(zip(all_true_labels_list,all_pred_list)), columns=['y_real','y_pred'])
    plt = sns.scatterplot(data=df_pred, x="y_real", y="y_pred")
    plt.set(xlim=(df_pred.y_real.min(0),df_pred.y_real.max(0)))
    plt.set(ylim=(df_pred.y_real.min(0), df_pred.y_real.max(0)))
    plt.plot([df_pred.y_real.min(0),df_pred.y_real.max(0)], [df_pred.y_real.min(0),df_pred.y_real.max(0)], 'b-', linewidth = 2)
    return plt, all_pred_list, all_true_labels_list, pearsonr(all_true_labels_list,all_pred_list)[0]