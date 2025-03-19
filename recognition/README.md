# Very-Large-Scale-Face-Recognition

## 配置

本项目的使用的硬件和环境的配置如下，其中项目环境使用 Anaconda 进行管理，方便作者环境切换，项目主要使用的 Python 语言和 PyTorch 库。

### 硬件配置
```python
RTX 3070Ti
cuda126 + 对应版本的cuDNN
```

### 项目环境
```python
numpy                     1.26.3
pillow                    11.0.0
lmdb                      0.9.29
python                    3.9.21
torch                     2.6.0+cu126
torchaudio                2.6.0+cu126
torchvision               0.21.0+cu126
tqdm         
```

## 数据准备

2025-03-19 先使用 CASIA-WebFace 进行试验，CASIA-WebFace文件的格式如下：

```
dataset/
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

先采用的是 CASIA-WebFace 数据集，因为 MS-Celeb-1M 过于庞大，直接使用需要的代价比较高，首先需要先将我们现存的数据集转换为 .lmdb 类型的文件。

LMDB 是一个高效的键值对数据库，适合存储大规模数据，并且大大降低了直接读取照片带来的I/O开销，读取数据通过内存映射实现。速度更加的快。以下是构建 LMDB 数据库的步骤：

- 安装 pillow 和 lmdb 库：`conda install pillow lmdb`
- 

## 配置参数

## 构建步骤

本文受[Build-Your-Own-Face-Model](https://github.com/siriusdemon/Build-Your-Own-Face-Model/)与[FFC](https://github.com/tiandunx/FFC/)的启发，主要用作作者学习使用

## 致谢
