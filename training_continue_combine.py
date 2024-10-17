import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat_gcn_transformer_meth_ge_mut_multiheadattn import GAT_GCN_Transformer_meth_ge_mut_multiheadattn
# from models.gat_gcn_transformer_meth_ge_mut_multiheadattn_mut import GAT_GCN_Transformer_meth_ge_mut_multiheadattn_mut
# from models.gat_gcn_transformer_meth_ge_mut_multiheadattn_meth import GAT_GCN_Transformer_meth_ge_mut_multiheadattn_meth
from utils import *
import datetime
import argparse
import random
import math


# training function at each epoch
# from loss_rank_contrastive import SupConLoss

def train(model, device, train_loader, optimizer, epoch, log_interval, model_st,args):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    loss_ae = nn.MSELoss()
    loss_cross = nn.CrossEntropyLoss()
    # loss_contrastive = SupConLoss(device = device)
    avg_loss = []
    weight_fn = 0.01
    weight_ae = 2
    alpha = 0.5
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        
        optimizer.zero_grad()
       
        if args.step == 2:
            # output1, output2, feature1, feature2 = model(data)
            output1, output2 = model(data)
            output = torch.add(output1, output2) /2.0
            loss1 = loss_fn(output1, data.y.view(-1, 1).float().to(device))
            loss2 = loss_fn(output2, data.y.view(-1, 1).float().to(device))
            # loss_contrast1 = loss_contrastive(feature1, data.y_rank.long().to(device))           
            # loss_contrast2 = loss_contrastive(feature2, data.y_rank.long().to(device)) 
            # loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
            # loss_contrast = (loss_contrast1 + loss_contrast2)*0.001
            # loss_combined = loss + loss_contrast
            loss_combined = loss_fn(output, data.y.view(-1, 1).float().to(device))
            
        elif args.step == 3:
            output1, output2, feature2, feature3, feature4 = model(data)
            loss_kd = alpha * loss_fn(output1, output2) + (1-alpha) * loss_fn(output2, data.y.view(-1, 1).float().to(device))
            loss_feature = torch.sqrt(torch.mean(torch.sum((feature2 - feature3)**2))) + torch.sqrt(torch.mean(torch.sum((feature2 - feature4)**2)))
            loss1 = loss_kd
            loss2 = loss_feature
            loss_combined = loss_kd + 0.5 * loss_feature
        else: 
            pass
        loss_combined.backward()
        optimizer.step()
        # output = torch.add(output1, output2) /2.0
        # loss1 = loss_fn(output1, data.y.view(-1, 1).float().to(device))
        # loss2 = loss_fn(output2, data.y.view(-1, 1).float().to(device))
        # loss_combined = loss_fn(output, data.y.view(-1, 1).float().to(device))
        # loss_combined.backward()
        # optimizer.step()
        avg_loss.append(loss_combined.item())
        if batch_idx % log_interval == 0:
            # print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss_combined: {:.6f}, loss_out: {}, loss_others: {}'.format(epoch,
            #                                                                batch_idx * len(data.x),
            #                                                                len(train_loader.dataset),
            #                                                                100. * batch_idx / len(train_loader),
            #                                                                loss_combined.item(), loss, loss_contrast))
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss_combined: {:.6f}, loss_others: {}, loss_rna: {}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss_combined.item(), loss1, loss2))
    return sum(avg_loss)/len(avg_loss)

def predicting(model, device, loader, model_st, args):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            #Non-variational autoencoder
            if args.step == 2:
                # output1, output2,_,_ = model(data)
                output1, output2 = model(data)
                output = torch.add(output1, output2) /2.0
            elif args.step == 3:
                output1, output2, _, _, _ = model(data)
                output = torch.add(output1, output2) /2.0
            total_preds_others = torch.cat((total_preds, output1.cpu()), 0)
            total_preds_rna = torch.cat((total_preds, output2.cpu()), 0)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_preds_others.numpy().flatten(),total_preds_rna.numpy().flatten()

def main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name):

    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
 
    # model_st = 'continue_combine_pretrain_multiheadattn(ge_others)_relu' + '_step_' + str(args.step) + '_rank_data_contrastloss'
    # model_st = 'continue_combine_pretrain_multiheadattn(ge_others)_relu' + '_step_' + str(args.step) + '_no_connection'
    # model_st = 'continue_combine_pretrain_multiheadattn(ge_others)_relu' + '_step_' + str(args.step) + '_rank_data_no_contrastloss'
    # model_st = 'continue_combine_pretrain_multiheadattn(ge_others)_relu' + '_step_' + str(args.step) + '_no_connection_seed_' + str(args.seed)    
    model_st = 'continue_combine_pretrain_multiheadattn(ge_others)_relu' + '_step_' + str(args.step) + '_connection_seed_' + str(args.seed) 
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
        if args.step == 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)           # step2
        elif args.step == 3:
            for name, param in model.named_parameters():          # step3 freeze parameters
                if 'xt_meth' in name or 'xt_mut' in name:             
                    param.requires_grad = False 
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)  # step3

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        best_mse = 1000
        best_pearson = 1
        best_epoch = -1

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

        for epoch in range(num_epoch):
            train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval, model_st, args)
            G,P,P1,P2 = predicting(model, device, val_loader, model_st,args)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
                        
            G_test, P_test, P_test1, P_test2 = predicting(model, device, test_loader, model_st,args)
            ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]
            ret_test1 = [rmse(G_test,P_test1),mse(G_test,P_test1),pearson(G_test,P_test1),spearman(G_test,P_test1)]
            ret_test2 = [rmse(G_test,P_test2),mse(G_test,P_test2),pearson(G_test,P_test2),spearman(G_test,P_test2)]
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
                    f.write(','.join(map(str,ret_test1)) + '\n')
                    f.write(','.join(map(str,ret_test2)) + '\n')
                    f.write(','.join(map(str,ret_test_)) + '\n')
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
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:5", help='Cuda')
    parser.add_argument('--step', type = int, default=2)
    parser.add_argument('--seed', type = int, default = 2024)

    args = parser.parse_args()

    # modeling = [GAT_GCN_Transformer_meth_ge_mut_multiple(multiple_ge = True)][args.model]
    # model = GAT_GCN_Transformer_meth_ge_mut_multiple(multiple_ge = True)
    CUDA_LAUNCH_BLOCKING=1
    device = torch.device(args.cuda_name if torch.cuda.is_available() else "cpu")
    # model = GAT_GCN_Transformer_meth_ge_mut_multiheadattn_mut(multiple_ge = True, device = device, step = args.step, connection=False, return_feature=False)
    model = GAT_GCN_Transformer_meth_ge_mut_multiheadattn(multiple_ge = True, device = device, step = args.step, connection=False, return_feature=False)
    train_batch = args.train_batch
    val_batch = args.val_batch
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    log_interval = args.log_interval
    cuda_name = args.cuda_name

    step = args.step
    if step == 2:
        # print('load model_GAT_GCN_Transformer_meth_ge_mut(rank_data)_GDSC.model')
        pretrained_dict = torch.load('model_GAT_GCN_Transformer_meth_ge_mut_GDSC.model')                           
        # pretrained_dict = torch.load('result/2024-05-21 10:18:55/model_GAT_GCN_Transformer_meth_ge_mut(rank_data)_GDSC.model')   
        # pretrained_dict = torch.load('result/2024-05-22 09:51:43/model_GAT_GCN_Transformer_meth_ge_mut(rank_data)_GDSC.model')
        # pretrained_dict = torch.load('model_continue_combine_pretrain_multiheadattn(ge_others)_relu_GDSC.model') 
    elif step == 3:
        print('load model_continue_combine_pretrain_multiheadattn(ge_others)_relu_GDSC.model')
        pretrained_dict = torch.load('model_continue_combine_pretrain_multiheadattn(ge_others)_relu_GDSC.model')   
    else:
        pass
    state_dict = {}

    # for name, param in pretrained_dict.items():
    #     if 'xt_ge' not in name:  
    #         state_dict[name] = param
    # model.load_state_dict(state_dict, strict = False)
    model.load_state_dict(pretrained_dict, strict = False)
    # seeds = [2024, 2014, 2004]
    seeds = [2024]
    for seed in seeds:
        args.seed = seed
        torch.manual_seed(seed)
        # 设置 Python 随机种子
        random.seed(seed)
        # 设置 NumPy 随机种子
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name)