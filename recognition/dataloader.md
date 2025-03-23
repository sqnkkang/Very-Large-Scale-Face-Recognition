# 数据加载器

## id_loader 和 instance_loader

论文里面提到了两个网络，称为 g 网络和 p 网络，前者主要管理动态池，因为 DCP 里面不会有我们的太多的 id 于是我们使用 id_loader 来根据 id 获取我们想要的人脸数据，instance_loader 获取的人脸数据还是随机获取的，这样将读取到的 batch_size 大小的人脸（id_loader 和 instance_loader 分别读取了一个 batch_size）拆分之后合并就能得到两个分别包含了 一半 id_loadr 一半  instance_loader 的人脸数据。

## 3 >> 构建步骤

你现在已经完成了数据加载器类的创建，检查没有问题的话，还剩下：

- 004 >> [LRU策略](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/lru.md)
- 005 >> [FFC-DCP动态池](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/ffc_dcp.md)
- 006 >> [训练](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/train.md)
- 007 >> [测试](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/test.md)

## 4 >> 致谢

本文受 [Build-Your-Own-Face-Model](https://github.com/siriusdemon/Build-Your-Own-Face-Model/) 与 [FFC](https://github.com/tiandunx/FFC/) 的启发，主要用作作者学习使用。
