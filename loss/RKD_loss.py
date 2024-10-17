import torch.nn as nn
import torch
import torch.nn.functional as F



class MAD(nn.Module):
    def __init__(self):
        super(MAD, self).__init__()

    def forward(self, fm_s, fm_t, logit_s, logit_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s = torch.mm(fm_s, fm_s.t())               # 8*8
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t,reduction='none')    # 512, 512

        loss = torch.sum(loss, dim = 1)
        # logit_t_prob = F.softmax(logit_t, dim=1)
        # H_teacher = torch.sum(-logit_t_prob * torch.log(logit_t_prob), dim=1)
        # H_teacher_prob = H_teacher / torch.sum(H_teacher)

        H_teacher_prob = torch.abs(100*(logit_s-logit_t))        
        loss = torch.sum(loss * H_teacher_prob)                    

        return loss

# class 
class MAD_uncertainty(nn.Module):
    def __init__(self):
        super(MAD_uncertainty, self).__init__()
    
    def forward(self, fm_s, fm_t, logit_s, logit_t):
        fm_s = fm_s.view(fm_s.size(0),fm_s.size(2), -1)
        G_s = torch.bmm(fm_s, fm_s.permute(0, 2, 1))             # 8*8
        # norm_G_s = F.normalize(G_s, p=2, dim=-1)
        norm_G_s = torch.sum(G_s, dim = -1)
        norm_G_s = F.normalize(norm_G_s, p=2, dim=-1)

        fm_t = fm_t.view(fm_t.size(0),fm_t.size(2), -1)
        G_t = torch.bmm(fm_t, fm_t.permute(0, 2, 1))
        # norm_G_t = F.normalize(G_t, p=2, dim=-1)
        norm_G_t = torch.sum(G_t, dim = -1)
        norm_G_t = F.normalize(norm_G_t, p=2, dim=-1)

        loss = F.mse_loss(norm_G_s, norm_G_t,reduction='none')
        loss = loss.view(-1)

        
        # logit_t_prob = F.softmax(logit_t, dim=1)
        # H_teacher = torch.sum(-logit_t_prob * torch.log(logit_t_prob), dim=1)
        # H_teacher_prob = H_teacher / torch.sum(H_teacher)
        H_teacher_prob = torch.abs(100*(logit_s-logit_t))
        loss = torch.sum(loss * H_teacher_prob)

        return loss


if __name__=='__main__':
    loss_mad = MAD()
    fm_s = torch.randn(8, 128)
    fm_t = torch.randn(8, 128)
    logit_s = torch.randn(8)
    logit_t = torch.randn(8)
    result = loss_mad(fm_s, fm_t, logit_s, logit_t)
    print(result)