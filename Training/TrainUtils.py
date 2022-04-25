from tqdm import tqdm
import torch

def train_one_epoch(train_loader, model, loss_fn, optimizer,device):
    all_pred = []
    step,total_loss = 0,0
    model.train()
    for _, data in tqdm(enumerate(train_loader),total = len(train_loader)):
        optimizer.zero_grad() #Reset gradients 
        pred,_ = model(data.to(device))
        
        y = data.y.unsqueeze(dim=-1).to(device)
#         print(f'pred {pred.dtype}, y {y.dtype}, datalist {data_list[0].x.dtype}')
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        step +=1
        pred = pred.cpu().detach()
#         all_pred.append(pred)
        y = y.cpu().detach()
        if step%1==0:
            torch.cuda.empty_cache()
            
#             print(f'step {step}| loss per interaction {loss.item()/args["batch_size"]}')
        
    
    average_loss_per_batch = total_loss/step
    return average_loss_per_batch


def test_one_epoch(test_loader, model, loss_fn,device, return_attn=False):
    all_pred, all_true_labels = [],[]
    step, total_loss = 0,0
    model.eval()
    for _, data in tqdm(enumerate(test_loader),total = len(test_loader)):
        with torch.no_grad():
            pred,attentions = model(data.to(device))
            y = data.y.unsqueeze(dim=-1).to(device)
            loss = loss_fn(pred,y)
        
            total_loss += loss.item()
            step+=1

            pred = pred.cpu().detach().numpy()
            true_label = y.cpu().detach().numpy()
            all_pred.extend(pred)
            all_true_labels.extend(true_label)
    
    if return_attn:
        return total_loss/step, all_pred, all_true_labels, attentions 
    else:
        return total_loss/step, all_pred, all_true_labels, None
    
    