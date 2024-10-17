import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat_gcn_transformer_meth_ge_mut_multiple import GAT_GCN_Transformer_meth_ge_mut_multiple
from utils import *
import datetime
import argparse
import random

seed = 2024
torch.manual_seed(seed)

random.seed(seed)

np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval, model_st):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    loss_ae = nn.MSELoss()
    avg_loss = []
    weight_fn = 0.01
    weight_ae = 2
    alpha = 1
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        if 'VAE' in model_st:
        #For variation autoencoder
            output, _, decode, log_var, mu = model(data)
            loss = weight_fn*loss_fn(output, data.y.view(-1, 1).float().to(device)) + loss_ae(decode, data.target_mut[:,None,:].float().to(device)) + torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        elif 'AE' in model_st:
            output, _, decode = model(data)
            loss = weight_fn*loss_fn(output, data.y.view(-1, 1).float().to(device)) + loss_ae(decode, data.target_mut[:,None,:].float().to(device)) 
        #For non-variational autoencoder
        else:
            output1, output2 = model(data)
            output = torch.add(output1, output2) /2.0
            # loss1 = loss_fn(output1, data.y.view(-1, 1).float().to(device))
            # loss2 = loss_fn(output2, data.y.view(-1, 1).float().to(device))
            # loss_combined = loss_fn(output, data.y.view(-1, 1).float().to(device))
            loss1 = loss_fn(output1, output2)
            loss2 = loss_fn(data.y.view(-1, 1).float().to(device), output2)
            train_loss = alpha * loss1 + loss2
        train_loss.backward()
        optimizer.step()
        avg_loss.append(train_loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss_combined: {:.6f}, loss_soft: {}, loss_hard: {}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           train_loss.item(), loss1, loss2))
    return sum(avg_loss)/len(avg_loss)

def predicting(model, device, loader, model_st):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            #Non-variational autoencoder
            if 'VAE' in model_st:
                #For variation autoencoder
                    output, _, decode, log_var, mu = model(data)
            elif 'AE' in model_st:
                output, _, decode = model(data)
            #For non-variational autoencoder
            else:
                output1, output2 = model(data)
                output = torch.add(output1, output2) /2.0
            total_preds_others = torch.cat((total_preds, output1.cpu()), 0)
            total_preds_rna = torch.cat((total_preds, output2.cpu()), 0)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_preds_others.numpy().flatten(),total_preds_rna.numpy().flatten()

def main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name):

    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
 
    model_st = 'continue_combine_pretrain_kd'
    dataset = 'GDSC'
    train_losses = []
    val_losses = []
    val_pearsons = []
    patience = 0
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train_mix_continue'+'.pt'
    processed_data_file_val = 'data/processed/' + dataset + '_val_mix_continue'+'.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test_mix_continue'+'.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train_mix_continue')
        val_data = TestbedDataset(root='data', dataset=dataset+'_val_mix_continue')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix_continue')

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
        print("CPU/GPU: ", torch.cuda.is_available())
                
        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        print(device)
        model = model.to(device)
        for name, param in model.named_parameters():
            if 'xt_ge' not in name and 'reg3' not in name and 'reg4' not in name and 'out2' not in name:
                param.requires_grad = False
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        best_mse = 1000
        best_pearson = 1
        best_epoch = -1
        # model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        # result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        if not os.path.exists('./result/' + current_time_str):
            os.makedirs('./result/' + current_time_str)
            print("Folder '{}' created successfully.".format(current_time_str))
        else:
            print("Folder '{}' already exists.".format(current_time_str))
        model_file_name = 'result/' + current_time_str + '/' + 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result/' + current_time_str + '/' + 'result_' + model_st + '_' + dataset +  '.csv'
        for epoch in range(num_epoch):
            train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval, model_st)
            G,P,P1,P2 = predicting(model, device, val_loader, model_st)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
                        
            G_test, P_test, P_test1, P_test2 = predicting(model, device, test_loader, model_st)
            ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]
            ret_test1 = [rmse(G_test,P_test1),mse(G_test,P_test1),pearson(G_test,P_test1),spearman(G_test,P_test1)]
            ret_test2 = [rmse(G_test,P_test2),mse(G_test,P_test2),pearson(G_test,P_test2),spearman(G_test,P_test2)]
            train_losses.append(train_loss)
            val_losses.append(ret[1])
            val_pearsons.append(ret[2])
            
            #Reduce Learning rate on Plateau for the validation loss
            scheduler.step(ret[1])

            if ret[1]<best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str,ret_test)) + '\n')
                    f.write(','.join(map(str,ret_test1)) + '\n')
                    f.write(','.join(map(str,ret_test2)) + '\n')
                best_epoch = epoch+1
                best_mse = ret[1]
                best_pearson = ret[2]
                print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,model_st,dataset)
                patience = 0
            else:
                patience += 1
                print(' no improvement since epoch ', best_epoch, '; best_loss, best pearson:', best_mse, best_pearson, model_st, dataset)
            if patience == 20:
                break
            # draw_loss(train_losses, val_losses, loss_fig_name)
            # draw_pearson(val_pearsons, pearson_fig_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0, help='0: Transformer_ge_mut_meth, 1: Transformer_ge_mut, 2: Transformer_meth_mut, 3: Transformer_meth_ge, 4: Transformer_ge, 5: Transformer_mut, 6: Transformer_meth')
    parser.add_argument('--train_batch', type=int, required=False, default=512,  help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=512, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=512, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=300, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')

    args = parser.parse_args()

    # modeling = [GAT_GCN_Transformer_meth_ge_mut_multiple(multiple_ge = True)][args.model]
    model = GAT_GCN_Transformer_meth_ge_mut_multiple(multiple_ge = True)
    train_batch = args.train_batch
    val_batch = args.val_batch
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    model.load_state_dict(torch.load('model_continue_combine_pretrain_relu_GDSC.model'), strict = True)

    main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name)