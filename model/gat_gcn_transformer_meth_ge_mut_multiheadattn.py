import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads)

    def forward(self, x):

        x = x.permute(0, 2, 1)                            # torch.Size([512, 33, 128])

        attn_output, _ = self.multihead_attn(x, x, x)     # torch.Size([512, 33, 128]) 

        attn_output = attn_output.permute(0, 2, 1)        # torch.Size([512, 128, 33])
        return attn_output



    
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads,device, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        

        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)
        self.linear_v = nn.Linear(input_dim, input_dim)
        

        self.linear_out = nn.Linear(input_dim, input_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # torch.Size([512, 33, 128])
        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)
        

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(-1, query.shape[1], self.head_dim)          # torch.Size([512, 8, 33, 16])
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(-1, key.shape[1], self.head_dim)           # torch.Size([512, 8, 10, 16])
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(-1, value.shape[1], self.head_dim) 
        
        # Dot product
        energy = torch.bmm(Q, K.transpose(1,2)) / self.scale   # # torch.Size([512, 8, 16, 10]) ->torch.Size([4096, 33, 10])     
        ############ cosine similarity ################
        # matrix1_norm = F.normalize(Q, p=2, dim=-1)  # [512, 8, 33, 16]
        # matrix2_norm = F.normalize(K, p=2, dim=-1)  # [512, 8, 10, 16]
        # energy = torch.matmul(matrix1_norm, matrix2_norm.transpose(1, 2))   
        ############  Euclidean Distance ##############
        # matrix1_expanded = Q.unsqueeze(2)  # [512, 8, 33, 1, 16]
        # matrix2_expanded = K.unsqueeze(1)  # [512, 8, 1, 10, 16]
        # energy = torch.norm(matrix1_expanded - matrix2_expanded, dim=-1)
       

        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(energy, dim=-1)
        
        attention = self.dropout(attention)
        
        output = torch.bmm(attention, V)
        
        output = output.view(batch_size, self.num_heads, Q.shape[1] , self.head_dim).permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.input_dim)
        output = self.linear_out(output)
        
        return output
    
class Gene2VecPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, device):
        super().__init__()
        gene2vec_weight = np.load('/workspace/geshuang/code/GraTransDRP-KD/data/gene2vec_16906.npy')
        gene2vec_weight = np.concatenate((gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight)
        self.device = device
    def forward(self, x):
        t = torch.tensor(x.shape[1]).to(self.device)
        return self.emb(t)

class GAT_GCN_Transformer_meth_ge_mut_multiheadattn(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, discriminator=False, multiple_ge=False, device = None, step = 2, 
                 compute_grad =False,connection=True,return_feature=False,uncertainty = False
                 ):

        super(GAT_GCN_Transformer_meth_ge_mut_multiheadattn, self).__init__()

        self.discriminator = discriminator
        self.multiple_ge = multiple_ge
        self.compute_grad = compute_grad
        self.uncertainty = uncertainty
        self.step = step
        self.connection = connection
        self.return_feature = return_feature
        self.multi_head_attention1 = MultiHeadAttention(output_dim, 8, device).to(device)
        self.self_attention1 = SelfAttention(d_model=128, n_heads=8)
        # self.multi_head_attention2 = MultiHeadAttention(output_dim * 2, 8, device).to(device)
        # self.self_attention_ge = SelfAttention(d_model=128, n_heads=8)
        # self.self_attention_mut = SelfAttention(d_model=128, n_heads=8)
        # self.self_attention_meth = SelfAttention(d_model=128, n_heads=8)
        self.n_output = n_output
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=num_features_xd, nhead=1, dropout=0.5)
        self.ugformer_layer_1 = nn.TransformerEncoder(self.encoder_layer_1, 1)
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=num_features_xd*10, nhead=1, dropout=0.5)
        self.ugformer_layer_2 = nn.TransformerEncoder(self.encoder_layer_2, 1)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # cell line mut feature
        self.conv_xt_mut_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_mut_1 = nn.MaxPool1d(3)
        self.conv_xt_mut_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_mut_2 = nn.MaxPool1d(3)
        self.conv_xt_mut_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_mut_3 = nn.MaxPool1d(3)
        self.fc1_xt_mut = nn.Linear(2944, output_dim)

        # cell line meth feature
        self.conv_xt_meth_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_meth_1 = nn.MaxPool1d(3)
        self.conv_xt_meth_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_meth_2 = nn.MaxPool1d(3)
        self.conv_xt_meth_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_meth_3 = nn.MaxPool1d(3)
        self.fc1_xt_meth = nn.Linear(1280, output_dim)

        # cell line ge feature
        self.max_seq_len = 16906
        self.token_emb = nn.Embedding(10, 200)
        self.pos_emb = Gene2VecPositionalEmbedding(200, self.max_seq_len-1, device).to(device)

        self.conv_xt_ge_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_ge_1 = nn.MaxPool1d(3)
        self.conv_xt_ge_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.pool_xt_ge_2 = nn.MaxPool1d(3)
        self.conv_xt_ge_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.pool_xt_ge_3 = nn.MaxPool1d(3)
        self.fc1_xt_ge = nn.Linear(4224, output_dim)

        # combined layers
        self.fc1 = nn.Linear(4*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        self.reg1 = nn.Linear(3*output_dim, 1024)
        self.reg2 = nn.Linear(1024, 128)
        self.out1 = nn.Linear(128, n_output)

        self.reg3 = nn.Linear(2*output_dim, 1024)
        self.reg4 = nn.Linear(1024, 128)
        self.out2 = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.unsqueeze(x, 1)
        x = self.ugformer_layer_1(x)
        x = torch.squeeze(x,1)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = torch.unsqueeze(x, 1)
        x = self.ugformer_layer_2(x)
        x = torch.squeeze(x,1)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        # target_mut input feed-forward:
        target_mut = data.target_mut                           # 512, 735
        # target_mut = self.add_noise_to_mutations(target_mut, noise_level=0.05)
        target_mut = target_mut[:,None,:]
        conv_xt_mut = self.conv_xt_mut_1(target_mut)
        conv_xt_mut = F.relu(conv_xt_mut)
        conv_xt_mut = self.pool_xt_mut_1(conv_xt_mut)
        conv_xt_mut = self.conv_xt_mut_2(conv_xt_mut)
        conv_xt_mut = F.relu(conv_xt_mut)
        conv_xt_mut = self.pool_xt_mut_2(conv_xt_mut)
        conv_xt_mut = self.conv_xt_mut_3(conv_xt_mut)
        conv_xt_mut = F.relu(conv_xt_mut)
        conv_xt_mut = self.pool_xt_mut_3(conv_xt_mut)          # torch.Size([512, 128, 23])
        
        # conv_xt_mut = self.self_attention_mut(conv_xt_mut)
        xt_mut = conv_xt_mut.reshape(-1, conv_xt_mut.shape[1] * conv_xt_mut.shape[2])
        xt_mut = self.fc1_xt_mut(xt_mut)


        target_meth = data.target_meth
        # target_meth = self.add_noise_to_mutations(target_meth, noise_level=0.05)
        target_meth = target_meth[:,None,:]
        conv_xt_meth = self.conv_xt_meth_1(target_meth)
        conv_xt_meth = F.relu(conv_xt_meth)
        conv_xt_meth = self.pool_xt_meth_1(conv_xt_meth)
        conv_xt_meth = self.conv_xt_meth_2(conv_xt_meth)
        conv_xt_meth = F.relu(conv_xt_meth)
        conv_xt_meth = self.pool_xt_meth_2(conv_xt_meth)
        conv_xt_meth = self.conv_xt_meth_3(conv_xt_meth)
        conv_xt_meth = F.relu(conv_xt_meth)
        conv_xt_meth = self.pool_xt_meth_3(conv_xt_meth)       # torch.Size([512, 128, 10])

        # conv_xt_meth = self.self_attention_meth(conv_xt_meth)  # 自注意
        xt_meth = conv_xt_meth.reshape(-1, conv_xt_meth.shape[1] * conv_xt_meth.shape[2])
        xt_meth = self.fc1_xt_meth(xt_meth)


        target_ge = data.target_ge                      # torch.Size([512, 1000])

        target_ge = target_ge[:,None,:]                 # torch.Size([512, 1, 735])
        # x_ge = x_ge.transpose(1,2)
        
        conv_xt_ge = self.conv_xt_ge_1(target_ge)       # torch.Size([512, 32, 993])
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_1(conv_xt_ge)      
        conv_xt_ge = self.conv_xt_ge_2(conv_xt_ge)
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_2(conv_xt_ge)
        conv_xt_ge = self.conv_xt_ge_3(conv_xt_ge)
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_3(conv_xt_ge)       # 512,128,33

        
        if self.connection:
            ####################### QUERY: GE ##########################
            others = torch.cat((conv_xt_meth, conv_xt_mut), 2)
            conv_xt_ge = self.multi_head_attention1(conv_xt_ge.transpose(1, 2), others.transpose(1, 2), others.transpose(1, 2))       
        # conv_xt_ge = conv_xt_ge.transpose(1,2) + conv_xt_ge                   
        xt_ge = conv_xt_ge.reshape(-1, conv_xt_ge.shape[1] * conv_xt_ge.shape[2])
        xt_ge = self.fc1_xt_ge(xt_ge)
        
        if self.multiple_ge:
            xt_mut = self.relu(xt_mut)             
            xt_meth = self.relu(xt_meth)          
            
            xt_ge = self.relu(xt_ge)               
            xt_ge.requires_grad_(True)
            feature1 = torch.cat((x, xt_mut, xt_meth), 1)     
            # xt_ge1 = torch.mul(xt_meth, xt_ge)
            # xt_ge2 = torch.mul(xt_mut, xt_ge)
            # xt_ge = xt_ge + xt_ge1 + xt_ge2
            feature2 = torch.cat((x, xt_ge), 1)
            feature3 = torch.cat((x, xt_mut), 1)
            feature4 = torch.cat((x, xt_meth), 1)
            xc1 = self.reg1(feature1)
            xc1 = self.relu(xc1)
            xc1 = self.dropout(xc1)
            xc1 = self.reg2(xc1)
            xc1 = self.relu(xc1)
            xc1 = self.dropout(xc1)
            out_other = self.out1(xc1)
            out_other = nn.Sigmoid()(out_other)
            xc2 = self.reg3(feature2)
            xc2 = self.relu(xc2)
            xc2 = self.dropout(xc2)
            xc2 = self.reg4(xc2)
            xc2 = self.relu(xc2)
            xc2 = self.dropout(xc2)
            out_rna = self.out2(xc1)
            out_rna = nn.Sigmoid()(out_rna)
            if self.step == 3:
                # return out_other, out_rna, xt_ge, xt_mut, xt_meth          # org_uncertainty
                return out_other, out_rna, feature2, feature3, feature4
            if self.compute_grad:
                
                out_rna.backward()
                cell_node_importance = torch.abs((xt_ge.grad))
            ################ org ####################################
                cell_node_importance = cell_node_importance.squeeze() 
                return cell_node_importance
            if self.return_feature:
                return out_other, out_rna, feature1, feature2
 
            return out_other, out_rna
        # concat
        xc = torch.cat((x, xt_mut, xt_meth, xt_ge), 1)
        if self.discriminator:
            return xc
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out, xc

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MultiHeadAttention(128, 8, device).to(device)
    xt_ge = torch.randn(8, 128).to(device)
    out = net(xt_ge, xt_ge, xt_ge)
    print(out)

