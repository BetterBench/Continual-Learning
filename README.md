# Continual Learning
[![DOI](https://zenodo.org/badge/150479999.svg)](https://zenodo.org/badge/latestdoi/150479999)

这是使用深度神经网络进行持续学习实验的 PyTorch 实现，参见
以下文章：
* [Three types of incremental learning](https://www.nature.com/articles/s42256-022-00568-3) (2022, *Nature Machine Intelligence*)

该存储库主要支持*学术持续学习环境*中的实验，其中基于分类的问题被分成多个不重叠的*上下文context* （或“任务task”，因为它们通常被称为）必须按顺序学习。
还为运行更灵活的“无任务”持续学习实验提供了一些支持
上下文之间逐渐过渡。


### 旧版
可以在此存储库中找到早期版本的代码
[in this branch](https://github.com/GMvandeVen/continual-learning/tree/preprints).
此版本的代码用于上述文章的两篇预印本中描述的持续学习实验：
- 持续学习的三种场景（<https://arxiv.org/abs/1904.07734>）
- 具有反馈连接的生成重播作为持续学习的一般策略
(<https://arxiv.org/abs/1809.10635>)


## 安装和配置
当前版本的代码已在 Fedora 操作系统上使用“Python 3.10.4”以及以下版本的 PyTorch 和 Torchvision 进行了测试：
* `pytorch 1.11.0`
* `torchvision 0.12.0`

更多使用的 Python 包列在“requirements.txt”中。
假设已设置 Python 和 pip，可以使用以下命令安装这些包：
```bash
pip install -r requirements.txt
```

该存储库中的代码本身不需要安装，但许多脚本应该可执行：
```bash
chmod +x main*.py compare*.py all_results.sh
```


## NeurIPS tutorial "Lifelong Learning Machines"

此代码存储库用于 "Lifelong Learning Machines"](https://sites.google.com/view/neurips2022-llm-tutorial).
有关如何重新运行本教程中介绍的实验的详细信息和说明，请参阅文件 [NeurIPS_tutorial.md](NeurIPS_tutorial.md)。

## Demos
##### Demo 1: Single continual learning experiment
```bash
./main.py --experiment=splitMNIST --scenario=task --si
```
这运行一个持续学习实验：
使用学术持续学习设置的Split MNIST 任务增量学习场景的突触智能（Synaptic Intelligence）方法。 有关数据、网络、训练进度和产生的输出的信息都会打印到屏幕上。
标准台式计算机上的预计运行时间约为 6 分钟，使用 GPU 时预计需要约 3 分钟。

##### Demo 2: 持续学习方法比较
```bash
./compare.py --experiment=splitMNIST --scenario=task
```
这运行了一系列持续学习实验，比较了各种方法在 Split MNIST 任务增量学习场景上的性能。
有关不同实验、实验进度和产生的输出（例如摘要 pdf）的信息会打印到屏幕上。
标准台式计算机上的预计运行时间约为 100 分钟，使用 GPU 时预计需要约 45 分钟。


## 重新运行文章中的比较
脚本“all_results.sh”提供了重新运行实验并重新创建“增量学习的三种类型”一文中报告的表格和图形的分步说明。

尽管可以按原样运行此脚本，但这将花费很长时间，并且并行实验可能是明智的。


## Running custom experiments
#### Academic continual learning setting
学术持续学习环境中的自定义个人实验可以使用“main.py”运行。
该脚本的主要选项有：
- `--experiment`: 如何构建上下文集？ (`splitMNIST`|`permMNIST`|`CIFAR10`|`CIFAR100`)
- `--contexts`: 有多少上下文？
- `--scenario`: 根据哪种情况？ (`task`|`domain`|`class`)

要运行特定方法，可以使用以下命令：
- Separate Networks: `./main.py --separate-networks`
- Context-dependent-Gating (XdG): `./main.py --xdg`
- Elastic Weight Consolidation (EWC): `./main.py --ewc`
- Synaptic Intelligence (SI): `./main.py --si`
- Learning without Forgetting (LwF): `./main.py --lwf`
- Functional Regularization Of the Memorable Past (FROMP): `./main.py --fromp`
- Deep Generative Replay (DGR): `./main.py --replay=generative`
- Brain-Inspired Replay (BI-R): `./main.py --brain-inspired`
- Experience Replay (ER): `./main.py --replay=buffer`
- Averaged Gradient Episodic Memory (A-GEM): `./main.py --agem`
- Generative Classifier: `./main.py --gen-classifier`
- incremental Classifier and Representation Learning (iCaRL): `./main.py --icarl`

运行基线模型（有关详细信息，请参阅文章）：
- None ("lower target"): `./main.py`
- Joint ("upper target"): `./main.py --joint`

有关更多选项的信息：`./main.py -h`。
该代码支持上述几种方法的组合。
还可以通过混合不同方法的组件来创建自定义方法，
尽管并未测试所有可能的组合。

#### 更灵活、“无任务”的持续学习实验
可以在更灵活、“无任务”的持续学习环境中运行定制的单独实验
`main_task_free.py`。 该脚本的主要选项有：
- `--experiment`:如何构建上下文集？ (`splitMNIST`|`permMNIST`|`CIFAR10`|`CIFAR100`)
- `--contexts`: 有多少上下文？
- `--stream`: 如何在上下文之间转换？ (`fuzzy-boundaries`|`academic-setting`|`random`)
- `--scenario`: 根据哪种情况？ (`task`|`domain`|`class`)

有关更多选项的信息：`./main_task_free.py -h`。 该脚本支持上述几种持续学习方法，但还不是全部。 一些方法经过轻微修改，使其适用于缺乏（已知）上下文边界的情况。
特别是，通常在上下文边界执行特定合并操作的方法，而是在每个“X”迭代中执行此合并操作，其中“X”是使用选项“--update-every”设置的。

## 训练期间的实时绘图
使用此代码，可以通过动态图来跟踪训练期间的进度。 此功能需要`visdom`，可以按如下方式安装：
```bash
pip install visdom
```
在运行实验之前，应从命令行启动 visdom 服务器：
```bash
python -m visdom.server
```
visdom 服务器现已启动，可以在浏览器中通过“http://localhost:8097”进行访问（图表将显示在那里）。 然后，在调用“./main.py”或“./main_task_free.py”时，应添加标志“--visdom”以使用动态图运行实验。

有关“visdom”的更多信息，请参阅<https://github.com/facebookresearch/visdom>。


### 引用
如果您在研究中使用此代码，请考虑引用主要随附文章：
```
@article{vandeven2022three,
  title={Three types of incremental learning},
  author={van de Ven, Gido M and Tuytelaars, Tinne and Tolias, Andreas S},
  journal={Nature Machine Intelligence},
  volume={4},
  pages={1185--1197},
  year={2022}
}
```

The BibTeX citations for the two preprints that were also produced using this code repository are given below.
Generally it is however preferred to cite the officially published version of the article,
but these preprints can be cited for aspects not featured in the published article.
```
@article{vandeven2019three,
  title={Three scenarios for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1904.07734},
  year={2019}
}

@article{vandeven2018generative,
  title={Generative replay with feedback connections as a general strategy for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1809.10635},
  year={2018}
}
```


### Acknowledgments
The research project from which this code originated has been supported by an IBRO-ISN Research Fellowship,
by the ERC-funded project *KeepOnLearning* (reference number 101021347),
by the National Institutes of Health (NIH) under awards R01MH109556 (NIH/NIMH) and P30EY002520 (NIH/NEI),
by the *Lifelong Learning Machines* (L2M) program of the Defence Advanced Research Projects Agency (DARPA)
via contract number HR0011-18-2-0025 and by the Intelligence Advanced Research Projects Activity (IARPA)
via Department of Interior/Interior Business Center (DoI/IBC) contract number D16PC00003.
Disclaimer: views and conclusions contained herein are those of the authors and should not be interpreted
as necessarily representing the official policies or endorsements, either expressed or implied,
of NIH, DARPA, IARPA, DoI/IBC, or the U.S. Government.
