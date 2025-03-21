import os
import argparse
import logging as logger
from optim.optimizer import get_optim_scheduler
from resnet_def import create_net
from ffc_ddp import FFC
import torch
from torch.utils.data import DataLoader
from util import *
import numpy as np
from torch.utils.data import RandomSampler
import random
import time
from util.config_helper import load_config

'''
添加日志文件记录训练的日志，调用 logger.info 函数，会将函数里面的输出，输出到日志里面，并带有下面初始化的信息
'''
logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
def get_lr(optimizer):
    '''
    从优化器里面获得当前的学习率
    '''
    for param_group in optimizer.param_groups:
        return param_group['lr']
def train_one_epoch(id_loader, instance_loader, ffc_net, optimizer,
                    cur_epoch, conf, saved_dir, real_iter, scaler, lr_policy, lr_scheduler, warmup_epochs, max_epochs):
    """Train one epoch by traditional training.
    """
    id_iter = iter(id_loader)
    instance_iter = iter(instance_loader)
    random.seed(cur_epoch)
    avg_data_load_time = 0

    db_size = len(instance_loader)
    start_time = time.time()
    for batch_idx, (ins_images, instance_label, _) in enumerate(instance_loader):
        # Note that label lies at cpu not gpu !!!
        if lr_policy != 'ReduceLROnPlateau':
            lr_scheduler.update(None, batch_idx * 1.0 / db_size)
        instance_images = ins_images.cuda(non_blocking=True)
        try:
            images1, images2, id_indexes = next(id_iter)
        except:
            id_iter = iter(id_loader)
            images1, images2, id_indexes = next(id_iter)

        images1_gpu = images1.cuda(non_blocking=True)
        images2_gpu = images2.cuda(non_blocking=True)

        instance_images1, instance_images2 = torch.chunk(instance_images, 2)
        instance_label1, instance_label2 = torch.chunk(instance_label, 2)

        optimizer.zero_grad()
        x = torch.cat([images1_gpu, instance_images1])
        y = torch.cat([images2_gpu, instance_images2])
        x_label = torch.cat([id_indexes, instance_label1])
        y_label = torch.cat([id_indexes, instance_label2])

        with torch.amp.autocast('cuda'):
            loss = ffc_net(x, y, x_label, y_label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        real_iter += 1

        if real_iter % 1 == 0:
            loss_val = loss.item()
            lr = lr_scheduler.get_lr()[0]
            duration = time.time() - start_time
            left_time = (max_epochs * db_size - real_iter) * (duration / 1000) / 3600
            logger.info('Iter %d Loss %.4f Epoch %d/%d Iter %d/%d Left %.2f hours' % (real_iter, loss_val, cur_epoch, max_epochs, batch_idx + 1, db_size, left_time))
            if lr_policy == 'ReduceLROnPlateau':
                lr_scheduler.step(loss_val)
            start_time = time.time()

        if real_iter % 1 == 0:
            snapshot_path = os.path.join(saved_dir, '%d.pt' % (real_iter // 2000))
            torch.save({'state_dict': ffc_net.probe_net.state_dict(), 'lru': ffc_net.lru.state_dict(), 'fc': ffc_net.queue.cpu(), 'qp': ffc_net.queue_position_dict}, snapshot_path)

    return real_iter


def train(conf):
    '''
    确保 cuDNN 被启用，并且 cuDNN 会自动优化计算性能。
    '''
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    '''
    将我们的 lmdb 和 kv 文件的位置传过去，从中创建名字为 instance_db 的 MultiLMDBDataset 类型的对象和名字为 id_db 的 PairLMDBDataset 类型的对象
    下面 instance_sampler 和 id_sampler 创建了随机采样器，里面是打乱的数据集
    最下面 instance_loader 和 id_loader 从上面的采样器里面进行数据加载得到的结果
    参数 instance_db 是数据集，conf.batch_size 是每个批次的样本数量，sampler=instance_sampler 指定采样器的随机采样
    num_works 使用 8个子进程来加载数据，pin_memory=True 将加载的数据存储在固定的内存中（适用于 GPU 训练，可以加速数据从 CPU 到 GPU 的传输）
    最后的参数 代表是否丢弃不足 batch_size 的样本
    '''
    instance_db = MultiLMDBDataset(conf.source_lmdb, conf.source_file)
    instance_sampler = RandomSampler(instance_db)
    instance_loader = DataLoader(instance_db, conf.batch_size, sampler=instance_sampler, num_workers=8, pin_memory=True, drop_last=False)

    id_db = PairLMDBDataset(conf.source_lmdb, conf.source_file)
    id_sampler = RandomSampler(id_db)
    id_loader = DataLoader(id_db, conf.batch_size, sampler=id_sampler, num_workers=8, pin_memory=True, drop_last=False)

    logger.info(f'id_loader length: {len(id_loader)}')
    logger.info(f'instance_loader length: {len(instance_loader)}')
    '''
    将参数传进去构建网络
    '''
    net = FFC(conf.net_type, conf.feat_dim, conf.queue_size, conf.scale, conf.loss_type, conf.margin,
              conf.alpha, conf.neg_margin, conf.pretrained_model_path, instance_db.num_class).cuda()
    '''
    网络自动继承，调用网络的名字可以直接执行网络里面的 forward 函数
    '''
    ffc_net = net
    '''
    加载配置参数文件，加载网络参数
    '''
    optim_config = load_config('config/optim_config')
    optim, lr_scheduler = get_optim_scheduler(ffc_net.parameters(), optim_config)

    real_iter = 0
    logger.info('enter training procedure...')
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(optim_config['epochs']):
        if optim_config['scheduler'] != 'ReduceLROnPlateau':
            lr_scheduler.update(epoch, 0.0)
        real_iter = train_one_epoch(id_loader, instance_loader, ffc_net, optim, epoch, conf, conf.saved_dir, real_iter, scaler, optim_config['scheduler'], lr_scheduler, optim_config['warmup'], optim_config['epochs'])

    id_db.close()
    instance_db.close()


if __name__ == '__main__':
    '''
    通过 train_ffc.sh 运行脚本，将参数传递到主函数里面，其中有些参数具有默认值，未进行复制的话按照默认来
    '''
    conf = argparse.ArgumentParser(description='fast face classification.')
    conf.add_argument('saved_dir', type=str, help='snapshot directory')
    conf.add_argument('--net_type', type=str, default='mobile', help='backbone type')
    conf.add_argument('--queue_size', type=int, default=7409, help='size of the queue.')
    conf.add_argument('--print_freq', type=int, default=1000, help='The print frequency for training state.')
    conf.add_argument('--pretrained_model_path', type=str, default='')
    conf.add_argument('--batch_size', type=int, default=256, help='batch size over all gpus.')
    conf.add_argument('--alpha', type=float, default=0.99, help='weight of moving_average')
    conf.add_argument('--loss_type', type=str, default='Arc', choices=['Arc', 'AM', 'SV'], help="loss type, can be softmax, am or arc")
    conf.add_argument('--margin', type=float, default=0.5, help='loss margin ')
    conf.add_argument('--scale', type=float, default=32.0, help='scaling parameter ')
    conf.add_argument('--neg_margin', type=float, default=0.25, help='scaling parameter ')
    conf.add_argument('--sync_bn', action='store_true', default=False)
    conf.add_argument('--feat_dim', type=int, default=512, help='feature dimension.')
    '''
    解析上面的参数，下面还添加了新的参数 source_lmdb，lmdb 数据库的位置和 kv 文件的位置，然后进行训练即可
    '''
    args = conf.parse_args()
    logger.info('Start optimization.')
    args.source_lmdb = ['./data/lmdb']
    args.source_file = ['./data/lmdb/train_kv.txt']
    logger.info(args)
    train(args)
    logger.info('Optimization done!')
