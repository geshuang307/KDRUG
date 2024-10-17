import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn

from models.gat_gcn_transformer_ge_only import GAT_GCN_Transformer_ge_only, Embed
from models.gat_gcn_transformer_mut_only import GAT_GCN_Transformer_mut_only
# from models.gat_gcn_transformer_mut_ge import GAT_GCN_Transformer_mut_ge
from models.gat_gcn_transformer_meth_only import GAT_GCN_Transformer_meth_only
from models.gat_gcn_transformer_meth_ge import GAT_GCN_Transformer_meth_ge
from models.gat_gcn_transformer_mut_meth import GAT_GCN_Transformer_mut_meth
from utils import *
import datetime
import argparse
import random
import datetime
# from loss import RnCLoss
# from loss_rank_contrastive import SupConLoss
import math

def add_gaussian_noise(matrix, mean=0, std=0.1):

    # 生成高斯噪声
    noise = torch.from_numpy(np.random.normal(mean, std, size=matrix.shape)).float().to(device)
    # 将噪声添加到输入矩阵中
    noisy_matrix = matrix + noise
    return noisy_matrix

def adjust_learning_rate(args, optimizer, epoch):
    lr = 0.01
    eta_min = lr * (0.1 ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / 300)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def train(model, device, train_loader, optimizer, epoch, log_interval, model_st):
    regressor = Embed()
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    loss_ae = nn.MSELoss()
    # loss_rnc = RnCLoss()
    # loss_contrastive = SupConLoss(device = device)
    avg_loss = []
    weight_fn = 0.01
    weight_ae = 2
    # adjust_learning_rate(args, optimizer, epoch)
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
            data1 = data
            # data1.target_ge = add_negative_binomial_noise_to_gene_expression(data.target_ge, 300, 0.5)
            # data1.target_ge = add_gaussian_noise(data.target_ge)
            output, feature = model(data)
            # output1, feature1 = model(data1)
            # feature2 = torch.cat((feature.unsqueeze(1), feature1.unsqueeze(1)), 1)
            loss1 = loss_fn(output, data.y.view(-1, 1).float().to(device))
            # loss2 = loss_rnc(feature2, data.y.view(-1, 1).float().to(device))
            # loss2 = loss_contrastive(feature, data.y_rank.long().to(device))
            # loss = loss1 + 0.001*loss2                   
            loss = loss1
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, loss1:{}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item(), loss1))
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
                output, _ = model(data)

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name):

    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    # for model in modeling:
        # model_st = model.__name__ +  'BN'     # '_' + 'no_fc'
    # model_st = 'GAT_GCN_Transformer_meth_ge_mut_fusion(add_noise=True)'   # pretrain
    # model_st = 'GAT_GCN_Transformer_meth_ge_mut_fusion(three_self_attn)'    # pretrain
    # model_st = 'GAT_GCN_Transformer_ge_only_feature(xc)_rncloss_0.0001'
    # model_st = 'GAT_GCN_Transformer_ge_only_feature(xc_first)_rankloss_gassain_0.001'
    # model_st = 'GAT_GCN_Transformer_meth_ge_mut_seed_' + str(args.seed)
    model_st = 'GAT_GCN_Transformer_mut_meth_seed_' + str(args.seed)
    dataset = 'GDSC'
    train_losses = []
    val_losses = []
    val_pearsons = []
    patience = 0
    print('\nrunning on ', model_st + '_' + dataset )
    # processed_data_file_train = 'data/processed/' + dataset + '_train_mix_continue_rank'+'.pt'
    # processed_data_file_val = 'data/processed/' + dataset + '_val_mix_continue_rank'+'.pt'
    # processed_data_file_test = 'data/processed/' + dataset + '_test_mix_continue_rank'+'.pt'
    processed_data_file_train = 'data/processed/' + dataset + '_train_mix_continue'+'.pt'
    processed_data_file_val = 'data/processed/' + dataset + '_val_mix_continue'+'.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test_mix_continue'+'.pt'
    # processed_data_file_train = 'data/processed/' + dataset + '_train_16906_ge_nonorm_continue'+'.pt'
    # processed_data_file_val = 'data/processed/' + dataset + '_val_16906_ge_nonorm_continue'+'.pt'
    # processed_data_file_test = 'data/processed/' + dataset + '_test_16906_ge_nonorm_continue'+'.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        # train_data = TestbedDataset(root='data', dataset=dataset+'_train_mix_continue_rank')
        # val_data = TestbedDataset(root='data', dataset=dataset+'_val_mix_continue_rank')
        # test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix_continue_rank')
        train_data = TestbedDataset(root='data', dataset=dataset+'_train_mix_continue')
        val_data = TestbedDataset(root='data', dataset=dataset+'_val_mix_continue')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix_continue')
        # train_data = TestbedDataset(root='data', dataset=dataset+'_train_16906_ge_nonorm_continue')
        # val_data = TestbedDataset(root='data', dataset=dataset+'_val_16906_ge_nonorm_continue')
        # test_data = TestbedDataset(root='data', dataset=dataset+'_test_16906_ge_nonorm_continue')
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
        print("CPU/GPU: ", torch.cuda.is_available())
                
        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        print(device)
        model = model.to(device)            # GAT_GCN_Transformer_meth_ge_mut_fusion 的时候 改
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        best_mse = 1000
        best_pearson = 1
        best_epoch = -1
        # 获取当前时间
        current_time = datetime.datetime.now()

        # 将当前时间格式化为字符串
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        if not os.path.exists('./result/' + current_time_str):
            os.makedirs('./result/' + current_time_str)
            print("Folder '{}' created successfully.".format(current_time_str))
        else:
            print("Folder '{}' already exists.".format(current_time_str))
        model_file_name = 'result/' + current_time_str + '/' + 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result/' + current_time_str + '/' + 'result_' + model_st + '_' + dataset +  '.csv'
        loss_fig_name = 'result/' + current_time_str + '/' + 'model_' + model_st + '_' + dataset + '_loss'
        pearson_fig_name = 'result/' + current_time_str + '/' + 'model_' + model_st + '_' + dataset + '_pearson'
        for epoch in range(num_epoch):
            train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval, model_st)
            G,P = predicting(model, device, val_loader, model_st)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
                        
            G_test,P_test = predicting(model, device, test_loader, model_st)
            ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]

            train_losses.append(train_loss)
            val_losses.append(ret[1])
            val_pearsons.append(ret[2])
            G_test_ = [(-10 *(np.log((G_test[i] + 1e-6)**(-1) - 1))) for i in range(G_test.shape[0])]
            P_test_ = [(-10 *(np.log((P_test[i] + 1e-6)**(-1) - 1))) for i in range(P_test.shape[0])]
            G_test_ = np.array(G_test_)
            P_test_ = np.array(P_test_)
            ret_test_ = [rmse(G_test_,P_test_),mse(G_test_,P_test_),pearson(G_test_,P_test_),spearman(G_test_,P_test_)]
            #Reduce Learning rate on Plateau for the validation loss
            scheduler.step(ret[1])

            if ret[1]<best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str,ret_test)) + '\n')
                    f.write(','.join(map(str,ret_test_)))
                best_epoch = epoch+1
                best_mse = ret[1]
                best_pearson = ret[2]
                print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,model_st,dataset)
                patience = 0
            else:
                patience += 1
                print(' no improvement since epoch ', best_epoch, '; best_loss, best pearson:', best_mse, best_pearson, model_st, dataset)
            # if patience == 20:
            #     break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0, help='0: Transformer_ge_mut_meth, 1: Transformer_ge_mut, 2: Transformer_meth_mut, 3: Transformer_meth_ge, 4: Transformer_ge, 5: Transformer_mut, 6: Transformer_meth')
    parser.add_argument('--train_batch', type=int, required=False, default=512,  help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=512, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=512, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=300, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:4", help='Cuda')
    parser.add_argument('--seed', type = int, default=2024)

    args = parser.parse_args()
    device = torch.device(args.cuda_name if torch.cuda.is_available() else "cpu")
    # modeling = [GAT_GCN_Transformer_meth_ge_mut, GAT_GCN_Transformer_ge, GAT_GCN_Transformer_meth, GAT_GCN_Transformer_meth_ge, GAT_GCN_Transformer_ge_only, GAT_GCN_Transformer, GAT_GCN_Transformer_meth_only,GAT_GCN_Transformer_meth_ge_mut_fusion(device)][args.model]
    # model = [modeling]
    # model = GAT_GCN_Transformer_meth_ge_mut_fusion(device, add_noise=True)            
    # model = GAT_GCN_Transformer_ge_only()
    # model = GAT_GCN_Transformer_meth_ge_mut()
    # model = GAT_GCN_Transformer_mut_only()
    # model = GAT_GCN_Transformer_mut_ge()
    # model = GAT_GCN_Transformer_meth_only()
    # model = GAT_GCN_Transformer_meth_ge()
    model = GAT_GCN_Transformer_mut_meth()
    train_batch = args.train_batch
    val_batch = args.val_batch
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    # model.load_state_dict(torch.load('model_GAT_GCN_Transformer_meth_ge_mut_GDSC.model'), strict = False)
    # seeds = [2024, 2014, 2004]
    seeds = [2024]
    for seed in seeds:
        args.seed = seed
        torch.manual_seed(seed)

        random.seed(seed)

        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name)