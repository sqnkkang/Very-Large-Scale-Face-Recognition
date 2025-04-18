# Very-Large-Scale-Face-Recognition

## 1 >> 依赖环境

本项目的使用的硬件和环境的配置如下，其中项目环境使用 Anaconda 进行管理，方便作者环境切换，项目主要使用的 Python 语言和 PyTorch 库。

- 硬件配置
```
RTX 3070Ti
cuda126 + cuDNN
```

- 项目环境
```
numpy                     1.26.3
opencv-python             4.11.0.86
pillow                    11.0.0
lmdb                      1.6.2
python                    3.9.21
torch                     2.6.0+cu126
torchaudio                2.6.0+cu126
torchvision               0.21.0+cu126
tqdm         
```

## 2 >> 项目参数

详见 `config/optim_config`

## 3 >> 构建步骤

- 001 >> [数据准备](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/001.md)
- 002 >> [模型架构](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/002.md)
- 003 >> [数据加载器](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/003.md)
- 004 >> [LRU策略](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/004.md)
- 005 >> [FFC-DCP动态池](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/005.md)
- 006 >> [训练](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/006.md)
- 007 >> [测试](https://github.com/sqnkkang/Very-Large-Scale-Face-Recognition/blob/master/recognition/007.md)

## 4 >> 致谢

本文受 [Build-Your-Own-Face-Model](https://github.com/siriusdemon/Build-Your-Own-Face-Model/) 与 [FFC](https://github.com/tiandunx/FFC/) 的启发，主要用作作者学习使用
