# ffc_dcp

## 1 >> 创建 FFC 类

- \_\_init\_\_ 函数，传递的参数包含很多，主要包含网络的类型，特征维度，DCP 池的大小，损失函数的类型等，初始化里面创建了两个网络，就是论文里面提到的 p 网络和 g 网络，分别进行更新 DCP 和进行梯度下降损失的计算。将这个队列记录为模型的缓冲区不参与梯度的计算，然后对队列里面的特征进行归一化初始的时候先生成随机的张量，之后初始化我们的 lru，下面的 queue_position_sict 是一个字典能记录队列里面每个位置的使用情况，开始的时候两个网络里面的参数相同，冻结我们的 g 网络的参数，因为该网络主要是负责更新 DCP 的。

```python
class FFC(Module):
    def __init__(self, net_type, feat_dim, queue_size=7409, scale=32.0, loss_type='AM', margin=0.4, momentum=0.99,
                 neg_margin=0.25, pretrained_model_path=None, num_class=None):
        super(FFC, self).__init__()
        assert loss_type in ('AM', 'Arc', 'SV')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.probe_net = create_net(net_type, feat_dim=feat_dim, fp16=True)
        self.gallery_net = create_net(net_type, feat_dim=feat_dim, fp16=True)

        self.register_buffer('queue', torch.rand(2, queue_size, feat_dim))
        self.queue = F.normalize(self.queue, dim=2)

        self.queue_size = queue_size
        self.feat_dim = feat_dim
        self.scale = scale
        self.margin = margin
        self.loss_type = loss_type

        self.lru = LRU(queue_size)
        self.queue_position_dict = {}
        for i in range(queue_size):
            self.queue_position_dict[i] = 0
        self.neg_margin = neg_margin
        self.register_buffer('mask', torch.zeros(self.queue_size, 1))
        self.m = momentum
        self.mask_svfc = 1.2
        self.hard_neg = min(max(int(self.queue_size * 0.0002), 3), 10)

        for param_p, param_g in zip(self.probe_net.parameters(), self.gallery_net.parameters()):
            param_g.data.copy_(param_p.data)
            param_g.requires_grad = False
    ...
```

- add_margin 函数，下面构建了常见的三个带边界的损失函数，开始是时候先构建正负样本的索引，存储的是下标，传进来的参数分别是 cos_theta [batch_size, queue_size] 和标签。

```python
    def add_margin(self, cos_theta, label):
        outlier_label = torch.where(label == -1)[0]
        pos_label_idx = torch.where(label != -1)[0]
        '''
        正样本的数量大于 0 个，则计算一下正样本的损失
        正样本的损失的计算方式为先找到自己的那一行与池子里面的所有的现存的向量之间的夹角，再找到当前的 label 
        转化为一个大小为 (batch_size, 1) 的张量，然后减去 margin
        scatter_ 是将特定的值填充到目标张量的指定位置上面，常见的用法就是 tensor.scatter_(dim, index, src) 代表维度、索引、填充的值
        填充的时候，先将 pos_label_view 转化为一维的索引，表示每个样本的真实类别的标签，即在 pos_cos_theta 的第一个维度上面进行填充
        最后计算我们的交叉熵损失
        对于负样本，找到和当前的样本最相似的负样本，即困难的负样本，提取困难负样本的余弦相似度，确保其为负数，负样本的损失就是这些相似度的平均值
        最后的总的损失就是正负样本损失的和
        '''
        if self.loss_type == 'AM':
            if pos_label_idx.numel() > 0:
                pos_cos_theta = cos_theta[pos_label_idx]
                batch_size = pos_cos_theta.shape[0]
                pos_label = label[pos_label_idx]
                '''
                找到每个正样本的值，转化为一维的，然后填充回去，其他的添加边距的方法大差不差，不再赘述
                '''
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
    ...
```

## 2 >> 构建步骤

你现在已经完成了 DCP 动态池，检查没有问题的话，还剩下：

- 006 >> [训练](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/train.md)
- 007 >> [测试](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/test.md)

## 3 >> 致谢

本文受 [Build-Your-Own-Face-Model](https://github.com/siriusdemon/Build-Your-Own-Face-Model/) 与 [FFC](https://github.com/tiandunx/FFC/) 的启发，主要用作作者学习使用。
