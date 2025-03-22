import torch
from torch.nn import Module
import math
import torch.nn.functional as F
from lru import LRU
from resnet_def import create_net
import os
import logging as logger

class FFC(Module):
    def __init__(self, net_type, feat_dim, queue_size=7409, scale=32.0, loss_type='AM', margin=0.4, momentum=0.99,
                 neg_margin=0.25, pretrained_model_path=None, num_class=None):
        '''
        先传进来需要的信息，比如网络的类型和特征维度这些
        '''
        super(FFC, self).__init__()
        assert loss_type in ('AM', 'Arc', 'SV')
        '''
        创建两个网络也就是论文里面提到的 p 网络和 g 网络，分别进行更新 DCP 和进行梯度下降损失的计算
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.probe_net = create_net(net_type, feat_dim=feat_dim, fp16=True)
        self.gallery_net = create_net(net_type, feat_dim=feat_dim, fp16=True)

        '''
        初始化队列也就是我们的 DCP 池子的大小，将这个队列记录为模型的缓冲区不参与梯度的计算，然后对队列里面的特征进行归一化
        初始的时候先生成随机的张量
        '''
        self.register_buffer('queue', torch.rand(2, queue_size, feat_dim))
        self.queue = F.normalize(self.queue, dim=2)

        self.queue_size = queue_size
        self.feat_dim = feat_dim
        self.scale = scale
        self.margin = margin
        self.loss_type = loss_type
        '''
        初始化我们的 lru，下面的 queue_position_sict 是一个字典能记录队列里面每个位置的使用情况
        '''
        self.lru = LRU(queue_size)
        self.queue_position_dict = {}
        for i in range(queue_size):
            self.queue_position_dict[i] = 0
        self.neg_margin = neg_margin
        self.register_buffer('mask', torch.zeros(self.queue_size, 1))
        self.m = momentum
        self.mask_svfc = 1.2
        self.hard_neg = min(max(int(self.queue_size * 0.0002), 3), 10)

        '''
        将 g 网络的参数 初始化为 p 网络的参数，并且冻结 g 网络的梯度
        '''
        for param_p, param_g in zip(self.probe_net.parameters(), self.gallery_net.parameters()):
            param_g.data.copy_(param_p.data)  # initialize
            param_g.requires_grad = False  # not update by gradient
    '''
    下面构建了常见的三个带边界的损失函数，开始是时候先构建正负样本的索引
    '''
    def add_margin(self, cos_theta, label):
        outlier_label = torch.where(label == -1)[0]
        pos_label_idx = torch.where(label != -1)[0]
        '''
        正样本的数量大于 0 个，则计算一下正样本的损失
        '''
        if self.loss_type == 'AM':
            if pos_label_idx.numel() > 0:
                pos_cos_theta = cos_theta[pos_label_idx]
                batch_size = pos_cos_theta.shape[0]
                pos_label = label[pos_label_idx]

                pos_cos_theta_m = pos_cos_theta[torch.arange(batch_size), pos_label].view(-1, 1) - self.margin
                pos_cos_theta.scatter_(1, pos_label.view(-1, 1), pos_cos_theta_m)
                cls_loss = F.cross_entropy(pos_cos_theta * self.scale, pos_label)
            else:
                cls_loss = 0
            if outlier_label.numel() > 0:
                outlier_cos_theta = cos_theta[outlier_label]
                outlier_idx = torch.argsort(outlier_cos_theta, dim=1, descending=True)[:, :self.hard_neg]
                hard_negative = torch.clip(torch.gather(outlier_cos_theta, 1, outlier_idx), 0)
                neg_loss = torch.mean(hard_negative)
            else:
                neg_loss = 0
            loss = cls_loss + neg_loss
            return loss
        elif self.loss_type == 'Arc':
            if pos_label_idx.numel() > 0:
                pos_cos_theta = cos_theta[pos_label_idx].float()
                pos_label = label[pos_label_idx]
                batch_size = pos_cos_theta.shape[0]
                gt = pos_cos_theta[torch.arange(0, batch_size), pos_label].view(-1, 1)
                sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
                cos_theta_m = gt * math.cos(self.margin) - sin_theta * math.sin(self.margin)
                pos_cos_theta.scatter_(1, pos_label.data.view(-1, 1), cos_theta_m)
                cls_loss = F.cross_entropy(pos_cos_theta * self.scale, pos_label)
            else:
                cls_loss = 0
            if outlier_label.numel() > 0:
                outlier_cos_theta = cos_theta[outlier_label]
                outlier_idx = torch.argsort(outlier_cos_theta, dim=1, descending=True)[:, :self.hard_neg]
                hard_negative = torch.clip(torch.gather(outlier_cos_theta, 1, outlier_idx), 0)
                neg_loss = torch.mean(hard_negative)
            else:
                neg_loss = 0
            loss = cls_loss + neg_loss
            return loss
        else:
            if pos_label_idx.numel() > 0:
                pos_cos_theta = cos_theta[pos_label_idx].float()
                pos_label = label[pos_label_idx]
                batch_size = pos_cos_theta.shape[0]
                gt = pos_cos_theta[torch.arange(0, batch_size), pos_label].view(-1, 1)
                mask = pos_cos_theta > gt - self.margin
                final_gt = torch.where(gt > self.margin, gt - self.margin, gt)
                hard_example = pos_cos_theta[mask]
                pos_cos_theta[mask] = self.mask_svfc * hard_example + self.mask_svfc - 1.0
                pos_cos_theta.scatter_(1, pos_label.data.view(-1, 1), final_gt)
                cls_loss = F.cross_entropy(pos_cos_theta * self.scale, pos_label)
            else:
                cls_loss = 0
            if outlier_label.numel() > 0:
                outlier_cos_theta = cos_theta[outlier_label]
                outlier_idx = torch.argsort(outlier_cos_theta, dim=1, descending=True)[:, :self.hard_neg]
                hard_negative = torch.clip(torch.gather(outlier_cos_theta, 1, outlier_idx), 0)
                neg_loss = torch.mean(hard_negative)
            else:
                neg_loss = 0
            loss = cls_loss + neg_loss
            return loss
    @torch.no_grad()
    def _momentum_update_gallery(self):
        """
        Momentum update of the key encoder
        """
        for param_p, param_g in zip(self.probe_net.parameters(), self.gallery_net.parameters()):
            param_g.data = param_g.data * self.m + param_p.data * (1. - self.m)
    '''
    因为我们本来的 batch 就是可以理解为一式两份的，前一份用来更新我们的 DCP，另一份用来更新模型里面的参数
    需要注意的是，他的论文原文表述的意思和代码有不相符的地方
    接下来判断现在的这个 batch 里面是不是有些人脸没出现在 lru 里面，默认调用 __contains__ 函数
    要是有新的人脸没有出现在 lru 里面，将当前位置的 rows 设置为 0，得到当前的 DCP 里面的存储的位置 存储到 cols 里面 这个位置标记为出现在了 DCP 里面
    否则的话，直接 get 得到当前的位置，双队列机制能更好的存储更多的特征的信息，避免单一队列的局限性
    '''
    def forward_impl(self, p_data, g_data, probe_label, gallery_label):
        '''
        先将 p_data 放进 p 网络里面进行前向传播，对于 p 网络的计算的时候我们不需要梯度下降，因为这个前向传播只是为了更新我们的 DCP
        '''
        p = self.probe_net(p_data)
        with torch.no_grad():
            g = self.gallery_net(g_data)
            g_label_list = gallery_label.tolist()

        rows = []
        cols = []
        # old_state = {}
        ones_idx = set([])
        for i, gl in enumerate(g_label_list):
            if gl not in self.lru:
                idx = self.lru.get(gl)
                rows.append(0)
                cols.append(idx)
                self.queue_position_dict[idx] = 1
            else:
                idx = self.lru.get(gl)
                rows.append(self.queue_position_dict[idx])
                cols.append(idx)
                ones_idx.add(idx)
                self.queue_position_dict[idx] = (self.queue_position_dict[idx] + 1) % 2

        r = torch.LongTensor(rows).cuda(g.device)
        c = torch.LongTensor(cols).cuda(g.device)
        with torch.no_grad():
            self.queue[r, c] = g
        '''
        看一下 p 标签是不是再 DCP 里面，计算一下 p 特征和队列里面的特征的余弦相似度，主要步骤为
        先和队列里面的第一行进行计算余弦相似度
        然后计算 fake_labels 这个东西队列里面有的话就会返回在队列里面的位置，否则就会返回 -1
        mask 通过动态选择队列，增强特征的多样性和模型的表达能力，感觉没什么逻辑
        '''
        fake_labels = []
        probe_label_list = probe_label.tolist()
        for pl in probe_label_list:
            fake_labels.append(self.lru.view(pl))

        label = torch.LongTensor(fake_labels).cuda(p.device)
        cos_theta1 = F.linear(p, self.queue[0]) 
        mask_idx = torch.LongTensor(list(ones_idx))
        self.mask[mask_idx, 0] = 1
        with torch.no_grad():
            weight = self.mask * self.queue[1] + (1 - self.mask) * self.queue[0]
        cos_theta2 = F.linear(p, weight)
        loss = self.add_margin(cos_theta1, label) + self.add_margin(cos_theta2, label)
        self.mask[mask_idx, 0] = 0
        return loss

    def forward_impl_rollback(self, p_data, g_data, probe_label, gallery_label):
        p = self.probe_net(p_data)
        with torch.no_grad():  # no gradient to gallery
            self._momentum_update_gallery() # update the gallery net
            g = self.gallery_net(g_data)
            g_label_list = gallery_label.tolist()
        rows = []
        cols = []
        old_state = {}
        ones_idx = set([])
        steps = 0
        for i, gl in enumerate(g_label_list):  # [0, 0, 0, 0,]
            if gl not in self.lru:
                idx = self.lru.try_get(gl)
                rows.append(0)  
                cols.append(idx)
                if idx not in old_state:
                    old_state[idx] = self.queue_position_dict[idx]
                self.queue_position_dict[idx] = 1
            else:
                idx = self.lru.try_get(gl)
                if idx not in old_state:
                    old_state[idx] = self.queue_position_dict[idx]
                rows.append(self.queue_position_dict[idx])
                cols.append(idx)
                ones_idx.add(idx)
                self.queue_position_dict[idx] = (self.queue_position_dict[idx] + 1) % 2
            steps += 1

        r = torch.LongTensor(rows).cuda(g.device)
        c = torch.LongTensor(cols).cuda(g.device)
        with torch.no_grad():
            old_tensor = self.queue[r, c]  
            self.queue[r, c] = g
        fake_labels = []
        probe_label_list = probe_label.tolist()
        for pl in probe_label_list:
            fake_labels.append(self.lru.view(pl))
        label = torch.LongTensor(fake_labels).cuda(p.device)

        cos_theta1 = F.linear(p, self.queue[0]) 
        # mask = mask.cuda(g.device)
        mask_idx = torch.LongTensor(list(ones_idx))
        self.mask[mask_idx, 0] = 1
        with torch.no_grad():
            weight = self.mask * self.queue[1] + (1 - self.mask) * self.queue[0]
        cos_theta2 = F.linear(p, weight)
        loss = self.add_margin(cos_theta1, label) + self.add_margin(cos_theta2, label)
        self.queue[r, c] = old_tensor  # restore queue state
        for k, v in old_state.items():
            self.queue_position_dict[k] = v
        self.mask[mask_idx, 0] = 0
        self.lru.rollback_steps(steps)
        return loss

    def forward(self, x, y, x_label, y_label):
        loss2 = self.forward_impl_rollback(x, y, x_label, y_label)
        loss1 = self.forward_impl(y, x, y_label, x_label)
        return loss1 + loss2
