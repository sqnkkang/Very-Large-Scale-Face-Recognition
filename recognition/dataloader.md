# 数据加载器

## 1 >> id_loader 和 instance_loader 介绍

论文里面提到了两个网络，称为 g 网络和 p 网络，前者主要管理动态池，因为 DCP 里面不会有我们的太多的 id 于是我们使用 id_loader 来根据 id 获取我们想要的人脸数据，instance_loader 获取的人脸数据还是随机获取的，这样将读取到的 batch_size 大小的人脸（id_loader 和 instance_loader 分别读取了一个 batch_size）拆分之后合并就能得到两个分别包含了 一半 id_loadr 一半  instance_loader 的人脸数据，以实现正负样本均衡，保证每个 batch_size 都会有至少一半的正样本，因为 id_loader 分开了一半。

## 2 >> MultiLMDBDataset 类的创建

这个类位于 `util/lmdb_loader.py` 实现的功能是加载 instance_loader，论文可以对多个数据库加载，所以实现方法如下：
```python
class MultiLMDBDataset(Dataset):
    def __init__(self, source_lmdbs, source_files, feat_lmdbs=None, feat_files=None, transforms=None, return_feats=False):
        '''
        判断传进来的特征是不是希望看到的样子，因为作者希望是传进来一个列表包含很多的数据库，不是列表的话手动加成列表
        需要注意的是列表的元素个数必须相等，不相等的话会报错，因为每个 lmdb 对应着一个 kv 文件
        将传进来的变量转化为成员变量，方便我们后续的操作
        '''
        if (not isinstance(source_lmdbs, list)) and (not isinstance(source_lmdbs, tuple)):
            source_lmdbs = [source_lmdbs]
        if (not isinstance(source_files, list)) and (not isinstance(source_files, tuple)):
            source_files = [source_files]
        assert len(source_files) == len(source_lmdbs)
        assert len(source_lmdbs) > 0
        self.source_lmdbs = source_lmdbs
        self.train_list = []
        max_label = 0
        last_label = 0
        '''
        打开很多个数据库，max_label 和 last_label 的作用的记录当前的数据库的最后一个 label 然后下一个数据库的文件读取编号的时候从当一个位置的下一个位置开始避免重复编号
        '''
        for db_id, file_path in enumerate(source_files):
            with open(file_path, 'r') as fin:
                for line in fin:
                    l = line.rstrip().lstrip()
                    if len(l) > 0:
                        items = l.split(' ')
                        self.train_list.append([items[0], db_id, int(items[1]) + last_label])
                        max_label = max(max_label, int(items[1]) + last_label)
            if max_label != last_label:
                max_label += 1
                last_label = max_label
        self.num_class = last_label
        self.transform = transforms
        self.feat_db = None
        self.return_feats = return_feats
        self.feat_lmdbs = feat_lmdbs
        self.feat_files = feat_files
        self.txns = None
        self.envs = None
        if transforms is not None:
            assert isinstance(transforms, list) or isinstance(transforms, tuple)
            assert len(self.transform) == len(source_lmdbs)
    '''
    返回数据集的样本的数量
    '''
    def __len__(self):
        return len(self.train_list)
    def open_lmdb(self):
        self.txns = []
        self.envs = []
        '''
        理清环境和事务的关系，遍历每一个数据库先得到该事务的环境，单个数据库的话一般这么写
        env = lmdb.op(lmdb_path, ...)
        使用事务的时候必须在这个环境的基础上，txn = env.begin(write=False, buffers=False) 开始了一个只读的事务，下面就可以直接使用该事务来读取数据了
        '''
        for lmdb_path in self.source_lmdbs:

            self.envs.append(lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=10, readahead=False, meminit=False))
            self.txns.append(self.envs[-1].begin(write=False, buffers=False))

        # if self.return_feats and (self.feat_lmdbs is not None and self.feat_files is not None):
        #     self.feat_db = KVDataset(self.feat_lmdbs, self.feat_files)
    def close(self):
        if self.txns is not None:
            for i in range(len(self.txns)):
                self.txns[i].abort()
                self.envs[i].close()
        if self.feat_db is not None:
            self.feat_db.close()
    '''
    Dataloader加载数据集的时候，__getitem__会被自动调用，他的功能是给定索引 index，从数据集中加载对应的数据
    并可以进行必要的预处理，返回值，通常返回一个样本数据和对应的标签
    先检查数据库是否被打开，不是的话先打开数据库，从 train_list[index] 获取当前样本对应的信息，即初始化的时候从 kv 文件里面加载的
    lmdb_key db_id label
    让后加载 Datum 对象，为解码做好准备，下面就是读取了存在对应的数据库里面的加码的数据 raw_byte
    之后对其进行解码，到第 96 行完成了数据的读取和解码的工作，当前的 img 就是一个货真价实的图像
    DataLoader 的工作原理，现根据sampler生成对应的索引序列，再按照批次大小从数据集中加载数据，将多个样本拼接为一个批次
    调用 dataset[index] 的时候 __getitem__ 或返回一个样本，使用 DataLoader 的时候它会将多个样本拼接为一个批次，假设现在的 batch_size 为 32
    
    for batch in data_loader:
    images, labels, features = batch
    # images: [32, C, H, W]  # 32 张图片
    # labels: [32]           # 32 个标签
    # features: [32, ...]    # 32 个特征          
    
    RandomSampler 生成一个随机索引序列，例如 [3, 45, 12, ..., 78]
    DataLoader 每次从索引序列中取出 batch_size 个索引（如 32 个）
    对于每个索引，调用 dataset[index] 获取单个样本              
    '''
    def __getitem__(self, index):
        if self.envs is None:
            self.open_lmdb()
        lmdb_key, db_id, label = self.train_list[index][:3]
        datum = Datum()
        raw_byte = self.txns[db_id].get(lmdb_key.encode('utf-8'))
        datum.ParseFromString(raw_byte)
        img = cv2.imdecode(np.frombuffer(datum.data, dtype=np.uint8), -1)
        '''
        了解完上面的逻辑，下面是对图像的处理，首先先对图像 50% 概率进行一次水平的翻转，下面如果发现图像是灰度图将其转化为 3 通道，并对其进行归一化
        torch.from_numpy 将其从 Numpy 数组转化为 PyTorch 类型的张量，本身就是彩色图像的话调整通道并进行归一化，特征数据库存在的话返回，否则直接返回 -1
        '''
        if random() < 0.5:
            img = cv2.flip(img, 1)
        if img.ndim == 2:
            buf = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.uint8)
            buf[0] = img
            buf[1] = img
            buf[2] = img
            tim = torch.from_numpy((buf - 127.5).astype(np.float32) * 0.0078125)
            ret = tim
            if self.feat_db is not None:
                return ret, label, self.feat_db[lmdb_key]
            else:
                return ret, label, -1
        else:
            img = torch.from_numpy((img.transpose((2, 0, 1)).astype(np.float32) - 127.5) * 0.0078125)
            ret = img
            if self.feat_db is not None:
                return ret, label, self.feat_db[lmdb_key]
            else:
                return ret, label, -1
```

## 3 >> 构建步骤

你现在已经完成了数据加载器类的创建，检查没有问题的话，还剩下：

- 004 >> [LRU策略](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/lru.md)
- 005 >> [FFC-DCP动态池](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/ffc_dcp.md)
- 006 >> [训练](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/train.md)
- 007 >> [测试](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/test.md)

## 4 >> 致谢

本文受 [Build-Your-Own-Face-Model](https://github.com/siriusdemon/Build-Your-Own-Face-Model/) 与 [FFC](https://github.com/tiandunx/FFC/) 的启发，主要用作作者学习使用。
