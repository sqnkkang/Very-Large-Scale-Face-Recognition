# 数据加载器

## 1 >> id_loader 和 instance_loader 介绍

论文里面提到了两个网络，称为 g 网络和 p 网络，前者主要管理动态池，因为 DCP 里面不会有我们的太多的 id 于是我们使用 id_loader 来根据 id 获取我们想要的人脸数据，instance_loader 获取的人脸数据还是随机获取的，这样将读取到的 batch_size 大小的人脸（id_loader 和 instance_loader 分别读取了一个 batch_size）拆分之后合并就能得到两个分别包含了 一半 id_loadr 一半  instance_loader 的人脸数据，以实现正负样本均衡，保证每个 batch_size 都会有至少一半的正样本，因为 id_loader 分开了一半。

## 2 >> MultiLMDBDataset 类的创建

- {__init__} 初始化函数，打开很多个数据库，max_label 和 last_label 的作用的记录当前的数据库的最后一个 label 然后下一个数据库的文件读取编号的时候从当一个位置的下一个位置开始避免重复编。

这个类位于 `util/lmdb_loader.py` 实现的功能是加载 instance_loader，论文可以对多个数据库加载，所以实现方法如下：
```python
class MultiLMDBDataset(Dataset):
    def __init__(self, source_lmdbs, source_files, feat_lmdbs=None, feat_files=None, transforms=None, return_feats=False):
        ...
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
    ...
```
- __len__ 返回加载的 instance_loader 的长度，理清环境和事务的关系，遍历每一个数据库先得到该事务的环境，单个数据库的话一般这么写 env = lmdb.open(lmdb_path, ...) 使用事务的时候必须在这个环境的基础上，txn = env.begin(write=False, buffers=False) 开始了一个只读的事务，下面就可以直接使用该事务来读取数据了

```python
    ...
    def __len__(self):
        return len(self.train_list)
    ...
```
- open_lmdb、close 函数，
```python
    ...
    def open_lmdb(self):
        self.txns = []
        self.envs = []
        for lmdb_path in self.source_lmdbs:

            self.envs.append(lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=10, readahead=False, meminit=False))
            self.txns.append(self.envs[-1].begin(write=False, buffers=False))

    def close(self):
        if self.txns is not None:
            for i in range(len(self.txns)):
                self.txns[i].abort()
                self.envs[i].close()
        if self.feat_db is not None:
            self.feat_db.close()
    ...
```
- __getitem__ 函数
```python
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
## 3 >> PairLMDBDataset 类的创建

这个类位于 `util/lmdb_loader.py` 实现的功能是加载 id_loader，论文可以对多个数据库加载，所以实现方法如下：

```python
class PairLMDBDataset(Dataset):
    def __init__(self, source_lmdbs, source_files, exclude_id_set=None):
        '''
        前面的部分基本上是一样的，先看一下是不是列表，不是的话进行转化
        '''
        if (not isinstance(source_lmdbs, list)) and (not isinstance(source_lmdbs, tuple)):
            source_lmdbs = [source_lmdbs]
        if (not isinstance(source_files, list)) and (not isinstance(source_files, tuple)):
            source_files = [source_files]
        assert len(source_files) == len(source_lmdbs)
        assert len(source_lmdbs) > 0
        self.envs = None
        self.txns = None
        self.source_lmdbs = source_lmdbs
        max_label = 0
        last_label = 0
        self.label2files = {}
        self.label_set = []
        '''
        这里就非常的不同了，要知道我们这个函数实现的是根据 label 加载图片，上面的函数是随即加载的，所以我们需要构建一个 label 字典，在字典里面的就是我们已经读取到的 label
        否则就是新的 label 要加到这个字典里面，同样的多个数据库需要动态的增加编号
        '''
        for db_id, file_path in enumerate(source_files):
            with open(file_path, 'r') as fin:
                for line in fin:
                    l = line.strip()
                    if len(l) > 0:
                        items = l.split(' ')
                        the_label = int(items[1]) + last_label
                        if the_label not in self.label2files:
                            self.label2files[the_label] = [db_id, []]
                            self.label_set.append(the_label)
                        self.label2files[the_label][1].append(items[0])
                        max_label = max(max_label, the_label)
            max_label += 1
            last_label = max_label
    '''
    有多少人，我们的数据集的大小就是多少，因为我们根据 id 来加载对象的
    '''
    def __len__(self):
        return len(self.label_set)
    def open_lmdb(self):
        self.txns = []
        self.envs = []
        for lmdb_path in self.source_lmdbs:
            self.envs.append(lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=4, readahead=False, meminit=False))
            self.txns.append(self.envs[-1].begin(write=False))
    def close(self):
        if self.txns is not None:
            for i in range(len(self.txns)):
                self.txns[i].abort()
                self.envs[i].close()
    '''
    先得到我们的标签集合，然后从每个 label 读取我们的图片，前一个 db_id 就是字典里面的每一个键为 label 值的数据库的 id 后一个就是图片地址了
    要是我们发现当前的 keys >= 2 即每一个 label 的存在的图片比较多，就正常的随机选取两个元素，否则一个元素选取两份
    接下来就和之前一样进行解码，然后进行我们的图像的处理就行了
    '''
    def __getitem__(self, index):
        if self.txns is None:
            self.open_lmdb()
        label = self.label_set[index]
        db_id, keys = self.label2files[label]
        if len(keys) >= 2:
            key1, key2 = sample(keys, 2)
        else:
            key1, key2 = keys[0], keys[0]
        datum = Datum()
        raw_byte = self.txns[db_id].get(key1.encode('utf-8'))
        datum.ParseFromString(raw_byte)
        img = cv2.imdecode(np.frombuffer(datum.data, dtype=np.uint8), -1)
        p = random()
        if p < 0.5:
            img = cv2.flip(img, 1)
        if img.ndim == 2:
            buf = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.uint8)
            buf[0] = img
            buf[1] = img
            buf[2] = img
            tim = torch.from_numpy((buf - 127.5).astype(np.float32) * 0.0078125)
            img_x = tim
        else:
            img = torch.from_numpy((img.transpose((2, 0, 1)).astype(np.float32) - 127.5) * 0.0078125)
            img_x = img
        datum2 = Datum()
        raw_byte = self.txns[db_id].get(key2.encode('utf-8'))
        datum2.ParseFromString(raw_byte)
        img2 = cv2.imdecode(np.frombuffer(datum2.data, dtype=np.uint8), -1)
        p = random()
        if p < 0.5:
            img2 = cv2.flip(img2, 1)
        if img2.ndim == 2:
            buf = np.zeros((3, img2.shape[0], img2.shape[1]), dtype=np.uint8)
            buf[0] = img2
            buf[1] = img2
            buf[2] = img2
            tim = torch.from_numpy((buf - 127.5).astype(np.float32) * 0.0078125)
            img_y = tim
        else:
            img2 = torch.from_numpy((img2.transpose((2, 0, 1)).astype(np.float32) - 127.5) * 0.0078125)
            img_y = img2
        '''
        和上一个的返回值略有不同，这里返回的是两张图片和对应的标签，所以 DataLoader 会将 batch_size 个图片拼成一个批次
        '''
        return img_x, img_y, label
```

## 4 >> 构建步骤

你现在已经完成了数据加载器类的创建，检查没有问题的话，还剩下：

- 004 >> [LRU策略](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/lru.md)
- 005 >> [FFC-DCP动态池](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/ffc_dcp.md)
- 006 >> [训练](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/train.md)
- 007 >> [测试](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/test.md)

## 5 >> 致谢

本文受 [Build-Your-Own-Face-Model](https://github.com/siriusdemon/Build-Your-Own-Face-Model/) 与 [FFC](https://github.com/tiandunx/FFC/) 的启发，主要用作作者学习使用。
