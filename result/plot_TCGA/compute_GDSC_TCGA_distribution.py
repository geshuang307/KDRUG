import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
import sys
sys.path.append('./')
sys.path.append('../')

from models.gat_gcn_transformer_ge_only import GAT_GCN_Transformer_ge_only


# from utils_rank import *
from utils import *
import datetime
import argparse
import random
from loss_rank_contrastive import SupConLoss
from loss import RnCLoss
from RKD_loss import RKDLoss, MAD, MAD_uncertainty

# training function at each epoch
import pickle
loss_fn = nn.MSELoss()
alpha = 0.5
def add_gaussian_noise(matrix, mean=0, std=0.1):

    noise = torch.from_numpy(np.random.normal(mean, std, size=matrix.shape)).float().to(device)
    noisy_matrix = matrix + noise
    return noisy_matrix


def predicting(student_model, device, val_loader):

    student_model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(val_loader.dataset)))
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            # num_data = data.target_ge.shape[0]
            # teacher_label = torch.ones(num_data).to(device)
            # student_label = torch.zeros(num_data).to(device)
            # label = data.y.view(-1, 1).float().to(device)
            # output_tea, _ = teacher_model(data)
            output_stu, _ = student_model(data)
            if args.uncertainty:
                output_stu = torch.mean(output_stu, dim=0)
            total_preds = torch.cat((total_preds, output_stu.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            # tea_preds = torch.cat((total_preds, output_tea.cpu()), 0)
        
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def plot_dist(P_test_, color, label):
    # plt.hist(P_test_, bins=10, alpha=0.6, label=label,color=color)
    sns.kdeplot(P_test_, shade=True, label=label, color=color,alpha = 0.6)

def convert(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError
def TCGA_drug(model, test_batch, lr, num_epoch, log_interval, cuda_name):
    # drug_list = [ 'Doxorubicin', 'Vinblastine', 'Etoposide', 'Gemcitabine','Temozolomide','Bicalutamide']
    drug_list = ['Doxorubicin', 'Vinblastine', 'Etoposide', 'Gemcitabine','Temozolomide','Bicalutamide', '5-Fluorouracil', 'Bleomycin', 'Docetaxel', 'Paclitaxel',
                 'Sorafenib', 'Tamoxifen']         # All 12 drugs are in the GDSC dataset.

    cmap = plt.cm.get_cmap('viridis')  

    # Generate color sequence
    color = [cmap(i / 5) for i in range(6)]  
    plt.figure(figsize = (12,6))
    j = 0
    dict_TCGA = {}
    for drug in drug_list:
        dict_TCGA[drug] = []
    for drug in drug_list:
        # plt.subplot(1,6,i)
        test_data = TestbedDataset(root='data', dataset='TCGA' + '_' + drug +'_mix_continue')
        test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        print(device)
        model = model.to(device)
    
        G_test,P_test = predicting(model, device, test_loader)
        G_test = np.array(G_test).squeeze()
        P_test = np.array(P_test).squeeze()
        ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]
        G_test_ = [(-10 *(np.log((G_test[i] + 1e-6)**(-1) - 1))) for i in range(G_test.shape[0])]
        P_test_ = [(-10 *(np.log((P_test[i] + 1e-6)**(-1) - 1))) for i in range(P_test.shape[0])]
        G_test_ = np.array(G_test_)
        P_test_ = np.array(P_test_)
        ret_test_ = [rmse(G_test_,P_test_),mse(G_test_,P_test_),pearson(G_test_,P_test_),spearman(G_test_,P_test_)]
        print(ret_test)
        print(ret_test_)
        for i in range(len(P_test_)):
            dict_TCGA[drug].append(np.float32(P_test[i]))

    with open('plot_fig/plot_TCGA/drug_cell_dict_TCGA_norm.json','w') as file:
        json.dump(dict_TCGA, file, default=convert)
        ############## Calculate the average IC50 value for each drug and sort from low to high. #############
    # Calculate the mean for the list corresponding to each key.
    sorted_keys_values = sorted(dict_TCGA.items(), key=lambda item: np.mean(item[1]))

    ########### "Remove drugs that are not in the list of 223 drugs. ############
    with open('/workspace/geshuang/code/GraTransDRP-KD/drug_dict', 'rb') as file:
        drug_dict = pickle.load(file)
    ############ Save these sorted data again as a dictionary, taking the top ten and the bottom ten. #############
    sorted_data = {key: value for key, value in sorted_keys_values if key in list(drug_dict.keys())}

    
    top_3 = dict(list(sorted_data.items())[:3])
    bottom_3 = dict(list(sorted_data.items())[-3:])

    new_dict = {**top_3, **bottom_3}
    with open('plot_fig/plot_TCGA/sorted_drug_sens3_res3_TCGA_norm.json','w') as file:
        json.dump(new_dict, file, default=convert)
################## Distribution of the top three sensitive/resistant cases 
    #     pred_label0 = []
    #     pred_label1 = []
    #     for index in range(len(G_test)):
    #         if G_test[index] == 0:
    #             pred_label0.append(P_test_[index])
    #         else:
    #             pred_label1.append(P_test_[index])
                
    #     plot_dist(P_test_, color[j], drug_list[j])
    #     j+= 1
    #     # file.close()
    # for spine in plt.gca().spines.values():
    #     spine.set_edgecolor('black') 
    #     spine.set_linewidth(2)        
    # plt.legend()
    # plt.savefig('plot_fig/plot_TCGA/TCGA_hist_6drug.png',dpi=600)
def predicting_drug(student_model, device, val_loader):

    student_model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_drugs = []
    print('Make prediction for {} samples...'.format(len(val_loader.dataset)))
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            drug_name = data.drug
            # num_data = data.target_ge.shape[0]
            # teacher_label = torch.ones(num_data).to(device)
            # student_label = torch.zeros(num_data).to(device)
            # label = data.y.view(-1, 1).float().to(device)
            # output_tea, _ = teacher_model(data)
            output_stu, _ = student_model(data)
            if args.uncertainty:
                output_stu = torch.mean(output_stu, dim=0)
            total_preds = torch.cat((total_preds, output_stu.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            total_drugs.extend(drug_name)
            # tea_preds = torch.cat((total_preds, output_tea.cpu()), 0)
        
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_drugs

def GDSC_drug(student_model, test_batch, lr, num_epoch, log_interval, cuda_name):
    dataset = 'GDSC'
    test_data = TestbedDataset(root='data', dataset=dataset+'_ALL_mix_continue_rank_drug')

    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
    print("CPU/GPU: ", torch.cuda.is_available())
            
    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(device)

    student_model = student_model.to(device)

    G_test,P_test,drug_name = predicting_drug(student_model, device, test_loader)
    ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]
    G_test_ = [(-10 *(np.log((G_test[i] + 1e-6)**(-1) - 1))) for i in range(G_test.shape[0])]
    P_test_ = [(-10 *(np.log((P_test[i] + 1e-6)**(-1) - 1))) for i in range(P_test.shape[0])]
    G_test_ = np.array(G_test_)
    P_test_ = np.array(P_test_)
    ret_test_ = [rmse(G_test_,P_test_),mse(G_test_,P_test_),pearson(G_test_,P_test_),spearman(G_test_,P_test_)]
    print(ret_test)
    print(ret_test_)
    # relation_path = 'plot_fig/plot_bulk/GDSC_norm_all_druglabel_logit_list.npy'
    # np.save(relation_path, [G_test_, P_test_])
    dict_gdsc = {}
    with open('plot_fig/plot_TCGA/sorted_drug_sens3_res3_TCGA_norm.json','r') as file:
        sorted_data = json.load(file)

    with open('/workspace/geshuang/code/GraTransDRP-KD/drug_dict', 'rb') as file:
        drug_dict = pickle.load(file)

    drug_name_list = list(drug_dict.keys())
    for drug in sorted_data.keys():
        dict_gdsc[drug] = []
    for i in range(len(G_test_)):
        drug = drug_name_list[drug_name[i].detach().cpu().numpy()]
        if drug in sorted_data.keys():
            dict_gdsc[drug].append(G_test[i])
    

    with open('plot_fig/plot_TCGA/sorted_drug_sens3_res3_GDSC_norm.json','w') as file:
        json.dump(dict_gdsc, file, default=convert)
   
# def plot_distribution():


if __name__ == "__main__":
    import json
    import csv
    import seaborn as sns
    parser = argparse.ArgumentParser(description='train model')
    # parser.add_argument('--model', type=int, required=False, default=[0,1], help='0: Transformer_ge_mut_meth, 1: Transformer_ge_mut, 2: Transformer_meth_mut, 3: Transformer_meth_ge, 4: Transformer_ge, 5: Transformer_mut, 6: Transformer_meth')
    parser.add_argument('--train_batch', type=int, required=False, default=512,  help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=512, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=512, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=300, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:4", help='Cuda')
    parser.add_argument('--seed', type = int, default = 2024)
    parser.add_argument('--uncertainty', type = str, default = False)

    args = parser.parse_args()

    seeds = [2024]
    # seeds = [2024]
    for seed in seeds:
        args.seed = seed
        torch.manual_seed(seed)
        # 设置 Python 随机种子
        random.seed(seed)
        # 设置 NumPy 随机种子
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device = torch.device(args.cuda_name if torch.cuda.is_available() else "cpu")

        student_model = GAT_GCN_Transformer_ge_only(uncertainty = args.uncertainty)
    
        train_batch = args.train_batch
        val_batch = args.val_batch
        test_batch = args.test_batch
        lr = args.lr
        num_epoch = args.num_epoch
        log_interval = args.log_interval
        cuda_name = args.cuda_name
        # student_model.load_state_dict(torch.load('result/2024-05-14 18:37:42/model_GAT_GCN_Transformer_ge_only_GDSC.model'), strict=False)
        student_model.load_state_dict(torch.load('result/2024-05-27 23:30:01/model_model_continue_combine_pretrain_multiheadattn(ge_others)_relu_regressor(freeze)_step2->4_tes_stu_MAD(loss1,0.001)_alpha0.5_connection_seed_2014_GDSC.model'))  
        # student_model.load_state_dict(torch.load('result/2024-06-01 00:09:18/model_model_continue_combine_pretrain_multiheadattn(ge_others)_relu_regressor(freeze)_step3_tes_stu_onKD_MAD_alpha0.5_connection_seed_2014_GDSC.model'))
        # student_model.load_state_dict(torch.load('result/2024-06-01 06:54:21/model_model_continue_combine_pretrain_multiheadattn(ge_others)_relu_regressor(freeze)_step3_tes_stu_onKD_MAD_alpha0.5_connection_seed_2004_GDSC.model'), strict=False)
        # student_model.load_state_dict(torch.load('/workspace/geshuang/code/GraTransDRP-KD/result/2024-05-31 18:58:35/model_model_continue_combine_pretrain_multiheadattn(ge_others)_relu_regressor(freeze)_step3_tes_stu_onKD_MAD_alpha0.5_connection_seed_2024_GDSC.model'))
        # GDSC_drug(student_model, test_batch, lr, num_epoch, log_interval, cuda_name)   # save TCGA_drug.csv file

