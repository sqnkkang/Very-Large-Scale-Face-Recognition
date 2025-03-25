# 数据准备

## 1 >> 数据格式

先使用的 `CASIA-WebFace` 进行试验，`CASIA-WebFace` 文件的格式如下，移动到 `data` 文件夹下面并改名为 `data`：

```
data/
  ├── id1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── id2/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── ...
```

## 2 >> 转化 `lmdb`

先采用的是 `CASIA-WebFace` 数据集，因为 `MS-Celeb-1M` 过于庞大，直接使用需要的代价比较高，首先需要先将我们现存的数据集转换为 `lmdb` 类型的文件。

`LMDB` 是一个高效的键值对数据库，适合存储大规模数据，并且大大降低了直接读取照片带来的 `I/O` 开销，读取数据通过内存映射实现，速度更加的快。

`creat_lmdb.py` 函数介绍：

首先构建一个 `LMDB` 类，该类实现了以下的函数

\_\_init\_\_.py 初始化函数，该函数有一个参数，传递你想保存的 `LMDB` 数据库的地址 `lmdb_path`，然后初始化类内的变量，像是构建的数据库的大小，键值对文件，和缓存空间，值得注意的是，我们自己构建的 `LMDB` 类是一个大小动态增加的数据库，缓存空间满了的话进行一次存储，空间不够则自动进行一次扩容，详细信息见下面的 `put` `commit` `close` 函数

```python
class LMDB:
    def __init__(self, lmdb_path):
        self.map_size = 500 * 1024 * 1024
        self.env = lmdb.open(lmdb_path, map_size=self.map_size)
        self.kv = {}
        self.buf_size = 1000
    ...
```

## 3 >> 构建步骤

你现在已经完成了数据的准备，检查没有问题的话，还剩下：

- 002 >> [模型架构](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/model.md)
- 003 >> [数据加载器](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/dataloader.md)
- 004 >> [LRU策略](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/lru.md)
- 005 >> [FFC-DCP动态池](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/ffc_dcp.md)
- 006 >> [训练](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/train.md)
- 007 >> [测试](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/test.md)

## 4 >> 致谢

本文受 [Build-Your-Own-Face-Model](https://github.com/siriusdemon/Build-Your-Own-Face-Model/) 与 [FFC](https://github.com/tiandunx/FFC/) 的启发，主要用作作者学习使用。
