import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat_gcn_transformer_ge_only import GAT_GCN_Transformer_ge_only
from models.gat_gcn_transformer_mut_only import GAT_GCN_Transformer_mut_only
from models.gat_gcn_transformer_mut_ge import GAT_GCN_Transformer_mut_ge
# from models.gat_gcn_transformer_meth_ge_mut_fusion import GAT_GCN_Transformer_meth_ge_mut_fusion
from models.gat_gcn_transformer_meth_ge_mut_multiheadattn import GAT_GCN_Transformer_meth_ge_mut_multiheadattn
from models.gat_gcn_transformer_meth_ge import GAT_GCN_Transformer_meth_ge
from models.gat_gcn_transformer_meth_only import GAT_GCN_Transformer_meth_only
from models.gat_gcn_transformer_mut_meth import GAT_GCN_Transformer_mut_meth
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
from utils import *
import datetime
import argparse
import random
# from loss_rank_contrastive import SupConLoss
from loss.loss import RnCLoss
from loss.RKD_loss import RKDLoss, MAD, MAD_uncertainty
# from ptflops import get_model_complexity_info

# training function at each epoch

loss_fn = nn.MSELoss()
alpha = 0.5
def add_gaussian_noise(matrix, mean=0, std=0.1):
    """
    """
    # 生成高斯噪声
    noise = torch.from_numpy(np.random.normal(mean, std, size=matrix.shape)).float().to(device)
    # 将噪声添加到输入矩阵中
    noisy_matrix = matrix + noise
    return noisy_matrix

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def train(teacher_model, student_model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    teacher_model.train()
    student_model.train()
    # discriminator_model = Discriminator().to(device)
    # loss_contrastive = SupConLoss(device = device)

    avg_loss = []
    loss_rnc = RnCLoss()
    loss_rkd = RKDLoss()
    loss_mad = MAD()
    # loss_mad_uncertainty = MAD_uncertainty()
    inference_time_list = []
    for batch_idx, data in enumerate(train_loader):

        data = data.to(device)
        label = data.y.view(-1, 1).float().to(device)
        # output_tea, _ = teacher_model(data)
        # our method
        output_tea1, output_tea2, feature_ge_tea1, _, _ = teacher_model(data)
        # output_tea1, output_tea2, feature_ge_tea1 =  teacher_model(data)
        output_tea = (output_tea1 + output_tea2)/2.0
        
        output_stu, feature_ge = student_model(data)
        

        # 输出参数量
        # total_params = count_parameters(teacher_model)
        # # print(f"Total number of teacher model parameters: {total_params}")
        # total_params = count_parameters(student_model)
        # # print(f"Total number of student model parameters: {total_params}")
        
        optimizer.zero_grad()
        if args.uncertainty:
            loss1 = loss_fn(output_tea.repeat(50,1), output_stu.view(-1))
            # loss1 = 0
            loss2 = alpha * loss_fn(label.repeat(50,1), output_stu.view(-1))
            loss_kd = loss_mad(feature_ge.unsqueeze(1), feature_ge_tea1.unsqueeze(1), output_stu, output_tea)  # org_uncertainty
      
        else:
            loss1 = loss_fn(output_tea, output_stu)
            loss2 = alpha * loss_fn(label, output_stu)
            loss_kd = loss_mad(feature_ge, feature_ge_tea1, output_stu, output_tea)
            # loss_kd = loss_rkd(feature_ge.unsqueeze(1), feature_ge_tea1.unsqueeze(1))      # RKD

        # data1 = data
        # data1.target_ge = add_gaussian_noise(data.target_ge)
        # output1, output2, feature_ge1, _, _ = teacher_model(data1)
        # output_stu1, feature_ge1 = student_model(data1)
        # feature2 = torch.cat((feature_ge_tea1.unsqueeze(1), feature_ge.unsqueeze(1)), 1)     # from teacher or student  or  teacher and student
        # loss_r = loss_rnc(feature2, data.y.view(-1, 1).float().to(device))
        # train_loss =  loss1 + alpha * loss2 + 0.001*loss_contrast      # 
        # loss_feature = loss_fn(feature_ge_tea1, feature_ge1)             # 增加一个loss feature的约束
        # train_loss =  loss1 + alpha * loss2 + 0.001*loss_r + 0.01*loss_feature # 增加一个loss feature的约束
        # train_loss =  loss1 + alpha * loss2 + 0.001*loss_r
        # train_loss = loss1 + loss2 + loss_kd*1e-3     # 加入RKD loss 
        train_loss = loss2 + loss_kd*1e-3  
        # start_time = time.time()             
        # train_loss = loss1 + loss2
        train_loss.backward()
        # end_time = time.time()
        # inference_time = end_time - start_time
        # if  batch_idx != 18:
        #     inference_time_list.append(inference_time)
        
        optimizer.step()
        avg_loss.append(train_loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, soft_loss:{}, hard_loss:{}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           train_loss.item(), loss1,loss2,))
    # mean_inference_time = np.mean(inference_time_list)
    # var_inference_time = np.var(inference_time_list)
    # print(f'Mean Inference time:{mean_inference_time:.6f} seconds')
    # print(f'Var Inference time:{var_inference_time:.6f} seconds')
    # return sum(avg_loss)/len(avg_loss), mean_inference_time
    return sum(avg_loss)/len(avg_loss)

def predicting(teacher_model, student_model, device, val_loader):

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

def main(teacher_model, student_model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name):

    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    # model_st = 'model_continue_method1_multimodal_to_mut_singlemodal_kdloss_alpha0.5_seed_2024_GDSC'
    model_st = 'model_continue_combine_pretrain_multiheadattn(ge_others)_relu_regressor(freeze)_step3_tea_stu(METH)_MAD_alpha0.5_connection_seed_'+ str(args.seed)  
    # model_st = 'model_continue_combine_pretrain_multiheadattn(ge_others)_relu_regressor(freeze)_step3_tea_stu_KD_MAD_alpha0.5_connection_uncertainty_seed_'+ str(args.seed)   # student加入对比损失
    # model_st = 'model_continue_combine_pretrain_multiheadattn(ge_others)_relu_regressor(nofreeze)_step2->4_tes_stu_MAD(loss1,0.001)_alpha0.5_connection_seed_' + str(args.seed)
    # model_st = 'model_continue_combine_pretrain_multiheadattn(ge_others)_relu_regressor(freeze)_step2->4_alpha0.5'
    # model_st = 'model_continue_combine_pretrain_multiheadattn(ge_others)_relu_regressor(freeze)_step2->4_alpha0.5_rankdata_contrast_connection'
    # model_st = 'model_continue_method1_multimodal_to_singlemodal_kdloss_alpha0.5_seed_' + str(args.seed)
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
        teacher_model = teacher_model.to(device)
               # 冻结teacher的特征提取器
        print(teacher_model)
        for param in teacher_model.parameters():     # freeze all the parameters
            param.requires_grad = False              # First, set all parameters to be frozen.
        params = list(teacher_model.parameters())
        params_to_update = []
        # for param in params[-3:]:                  # Set the parameters of the teacher part to be updatable.
        #     param.requires_grad = True              
        #     params_to_update.append(param)
        student_model = student_model.to(device)
        # discriminator_model = Discriminator().to(device)
        all_params_update = list(student_model.parameters()) + params_to_update
        optimizer = torch.optim.Adam(all_params_update, lr=lr)
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

        save_args = 'result/' + current_time_str + '/' + 'args_' + dataset + '.txt'
       
        with open(save_args, 'w') as f:
            for arg, value in args._get_kwargs():
                f.write(f"{arg}: {value}\n")
            f.close()    
        inference_epoch_list = []
        for epoch in range(num_epoch):
            
            train_loss = train(teacher_model, student_model, device, train_loader, optimizer, epoch+1)
            # train_stu_loss = train(student_model, device, train_loader, optimizer, epoch+1)
            G,P = predicting(teacher_model, student_model, device, val_loader)

            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]

            G_test,P_test = predicting(teacher_model, student_model, device, test_loader)
            ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]
            G_test_ = [(-10 *(np.log((G_test[i] + 1e-6)**(-1) - 1))) for i in range(G_test.shape[0])]
            P_test_ = [(-10 *(np.log((P_test[i] + 1e-6)**(-1) - 1))) for i in range(P_test.shape[0])]
            G_test_ = np.array(G_test_)
            P_test_ = np.array(P_test_)
            ret_test_ = [rmse(G_test_,P_test_),mse(G_test_,P_test_),pearson(G_test_,P_test_),spearman(G_test_,P_test_)]
            # print(ret_test_)
            # break
            train_losses.append(train_loss)
            val_losses.append(ret[1])
            val_pearsons.append(ret[2])

            #Reduce Learning rate on Plateau for the validation loss
            scheduler.step(ret[1])

            if ret[1] < best_mse:
                patience = 0
                best_mse = ret[1]
                best_pearson = ret[2]
                torch.save(student_model.state_dict(), model_file_name)
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str, ret_test))+'\n')
                    f.write(','.join(map(str, ret_test_)))
                
                best_epoch = epoch + 1
                print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,model_st,dataset)
                patience = 0
            else:
                patience += 1
                print(' no improvement since epoch ', best_epoch, '; best_loss, best pearson:', best_mse, best_pearson, model_st, dataset)
            # if patience == 20:
            #     break
            ####################### Computational complexity ########################
            # if epoch <10:       
            #     inference_epoch_list.append(mean_inference_time)
            # else:
            #     break
            # inference_epoch_mean = np.mean(inference_epoch_list)
            # inference_epoch_var = np.var(inference_epoch_list)
            # print(f'Mean Inference time:{inference_epoch_mean:.6f} seconds')
            # print(f'Var Inference time:{inference_epoch_var:.6f} seconds')

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser(description='train model')
    # parser.add_argument('--model', type=int, required=False, default=[0,1], help='0: Transformer_ge_mut_meth, 1: Transformer_ge_mut, 2: Transformer_meth_mut, 3: Transformer_meth_ge, 4: Transformer_ge, 5: Transformer_mut, 6: Transformer_meth')
    parser.add_argument('--train_batch', type=int, required=False, default=512,  help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=512, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=512, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=300, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:3", help='Cuda')
    parser.add_argument('--seed', type = int, default = 2024)
    parser.add_argument('--uncertainty', type = str, default = False)

    args = parser.parse_args()

    # seeds = [2024,2014,2004]
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
        device = torch.device(args.cuda_name if torch.cuda.is_available() else "cpu")
        # teacher_model = GAT_GCN_Transformer_meth_ge_mut_fusion(device=device)
        teacher_model = GAT_GCN_Transformer_meth_ge_mut_multiheadattn(multiple_ge=True, device=device, connection=True, step = 3, uncertainty = False)
        # teacher_model = GAT_GCN_Transformer_meth_ge_mut_fusion(device=device)
        # teacher_model = GAT_GCN_Transformer_meth_ge_mut()
        # student_model = GAT_GCN_Transformer_ge_only(uncertainty = False)
        # student_model = GAT_GCN_Transformer_mut_only()
        # student_model = GAT_GCN_Transformer_mut_ge()
        # student_model = GAT_GCN_Transformer_meth_ge()
        student_model = GAT_GCN_Transformer_meth_only()
        # student_model = GAT_GCN_Transformer_mut_meth()
        train_batch = args.train_batch
        val_batch = args.val_batch
        test_batch = args.test_batch
        lr = args.lr
        num_epoch = args.num_epoch
        log_interval = args.log_interval
        cuda_name = args.cuda_name
        # stu_state_dict, tea_state_dict = {}, {}
        # pretrained_dict = torch.load('model_continue_combine_pretrain_relu_GDSC.model')
        # for name, param in pretrained_dict.items():
        #     if 'xt_ge' not in name:  
        #         tea_state_dict[name] = param
        #     elif 'xt_meth' not in name and 'xt_mut' not in name:
        #         stu_state_dict[name] = param
        #     else:
        #         pass
            
        # student_model.load_state_dict(stu_state_dict, strict=False)
        # teacher_model.load_state_dict(tea_state_dict, strict=False)
        # teacher_model.load_state_dict(torch.load('result/2024-05-14 18:35:52/model_GAT_GCN_Transformer_meth_ge_mut_fusion_(ge_others_self)_GDSC.model'))
        teacher_model.load_state_dict(torch.load('model_continue_combine_pretrain_multiheadattn(ge_others)_relu_GDSC.model'), strict=False)         # our proposed multimodal
        # teacher_model.load_state_dict(torch.load('result/2024-05-29 09:39:59/model_GAT_GCN_Transformer_meth_ge_mut_seed_2024_GDSC.model'),strict=False)  # origion
        # a = torch.load('result/2024-05-29 09:39:59/model_GAT_GCN_Transformer_meth_ge_mut_seed_2024_GDSC.model')
        a = torch.load('model_continue_combine_pretrain_multiheadattn(ge_others)_relu_GDSC.model')
        if hasattr(teacher_model.conv1, 'lin_src') and a['conv1.lin_src.weight'] is not None:
            teacher_model.conv1.lin.weight.data = a['conv1.lin_src.weight'].data.clone()
            print(teacher_model.conv1.lin.weight)
        # teacher_model.load_state_dict(torch.load('result/2024-05-17 11:19:58/model_continue_combine_pretrain_multiheadattn(ge_others)_relu_step_2_connection_False_GDSC.model'))
        # teacher_model.load_state_dict(torch.load('model_GAT_GCN_Transformer_meth_ge_mut_GDSC.model'))
        # teacher_model.load_state_dict(torch.load('result/2024-05-22 20:04:50/model_continue_combine_pretrain_multiheadattn(ge_others)_relu_step_2_rank_data_no_contrastloss_GDSC.model')) # rank data
        # teacher_model.load_state_dict(torch.load('result/2024-05-15 14:45:05/model_continue_combine_pretrain_multiheadattn(ge_others)_relu_step_3_GDSC.model'))
        # student_model.load_state_dict(torch.load('result/2024-05-14 18:37:42/model_GAT_GCN_Transformer_ge_only_GDSC.model'))
        # teacher_model.load_state_dict(torch.load('result/2024-05-29 10:14:35/model_continue_combine_pretrain_multiheadattn(ge_others)_relu_step_2_no_connection_seed_2024_GDSC.model'))
        # student_model.load_state_dict(torch.load('result/2024-05-14 18:37:42/model_GAT_GCN_Transformer_ge_only_GDSC.model'), strict=False)     # final ge
        # b = torch.load('result/2024-05-14 18:37:42/model_GAT_GCN_Transformer_ge_only_GDSC.model')
        # if hasattr(student_model.conv1, 'lin_src') and b['conv1.lin_src.weight'] is not None:
        #     student_model.conv1.lin.weight.data = b['conv1.lin_src.weight'].data.clone()
        #     print(student_model.conv1.lin.weight)
        
        # student_model.load_state_dict(torch.load('result/2024-10-03 17:07:44/model_GAT_GCN_Transformer_mut_only_seed_2024_GDSC.model'), strict=False)     # final mut
        # student_model.load_state_dict(torch.load('result/2024-10-03 17:25:47/model_GAT_GCN_Transformer_mut_ge_seed_2024_GDSC.model'), strict=True)     # mut_ge 
        # student_model.load_state_dict(torch.load('result/2024-10-09 11:05:56/model_GAT_GCN_Transformer_meth_ge_seed_2024_GDSC.model'), strict=True)  # METH_GE
        # student_model.load_state_dict(torch.load('result/2024-10-09 11:11:26/model_GAT_GCN_Transformer_mut_meth_seed_2024_GDSC.model'), strict=True)    # MUT_METH
        student_model.load_state_dict(torch.load('result/2024-10-09 11:00:51/model_GAT_GCN_Transformer_meth_seed_2024_GDSC.model'), strict=True)     # METH
        # student_model.load_state_dict(torch.load('result/2024-05-21 22:01:07/model_GAT_GCN_Transformer_ge_only(rank_data)_GDSC.model'))
        
        main(teacher_model, student_model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name)