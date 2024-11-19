# LSTM for Pytorch

<!--命名规则 {model_name}-{dataset}-{framework}-->

[TOC]

## 模型简介

长短期记忆网络（Long Short-Term Memory, LSTM）是一种特殊类型的循环神经网络（RNN），由 Hochreiter 和 Schmidhuber 于 1997 年提出。LSTM 旨在解决传统 RNN 在处理长序列数据时遇到的梯度消失或梯度爆炸问题。
LSTM 网络的核心是三个门的机制：遗忘门（forget gate）、输入门（input gate）、输出门（output gate）。这些门通过自适应的方式控制信息的流动，从而实现对长期依赖信息的捕捉。


## 资源准备

1. 数据集资源下载

	CMD\_12\_test是一个语音分类数据集，是从wav数据预处理后得来，包含 `_slilence_, _unknown_, yes, no, up, down, left, right, on, off, stop, go` 12个类别，存放在本工程data目录下。

2. 模型资源下载

	权重存放在本工程weight目录下。

3. 清微github modelzoo仓库下载

	```git clone https://github.com/tsingmicro-toolchain/ts.knight-modelzoo.git```

## Knight环境准备

1. 联系清微智能获取Knight工具链版本包 ```ReleaseDeliverables/ts.knight-x.x.x.x.tar.gz ```。下面以ts.knight-2.0.0.4.tar.gz为例演示。

2. 检查docker环境

	​默认服务器中已安装docker（版本>=19.03）, 如未安装可参考文档ReleaseDocuments/《TS.Knight-使用指南综述_V1.4.pdf》。
	
	```
	docker -v   
	```

3. 加载镜像
	
	```
	docker load -i ts.knight-2.0.0.4.tar.gz
	```

4. 启动docker容器

	```
	docker run -v ${localhost_dir}/ts.knight-modelzoo:/ts.knight-modelzoo -it ts.knight:2.0.0.4 /bin/bash
	```
	
	localhost_dir为宿主机目录。

## 快速体验

在docker 容器内运行以下命令:

```
cd /ts.knight-modelzoo/pytorch/builtin/audio/classification/
```

```
sh lstm/scripts/run.sh
```

## 模型移植流程

### 1. 量化

-   模型准备
	
	如上述"Knight环境准备"章节所述，准备好lstm pytorch权重文件。
	

-   量化数据准备

    通过命令行参数```-i 10```指定校准数据量，```-d```指定数据集路径。

-   模型转换函数、推理函数准备
	
	已提供量化依赖的模型转换和推理函数py文件: ```/ts.knight-modelzoo/pytorch/builtin/audio/classification/lstm/src/lstm.py```

-   执行量化命令

	在容器内执行如下量化命令，生成量化后的文件 `lstm_quantize.onnx` 存放在 -s 指定输出目录。

    	Knight --chip TX5368AV200 quant onnx -m lstm
    		-w /ts.knight-modelzoo/pytorch/builtin/audio/classification/lstm/weight/lstm.pth 
    		-f pytorch 
    		-uds /ts.knight-modelzoo/pytorch/builtin/audio/classification/lstm/src/lstm.py 
    		-if infer_pytorch_rnn
			-s ./tmp/lstm 
    		-d /ts.knight-modelzoo/pytorch/builtin/audio/classification/lstm/data/cmd_12_test.pkl
    		-i 10
			-qm min_max


### 2. 编译


    Knight --chip TX5368AV200 rne-compile --onnx lstm_quantize.onnx --outpath .


### 3. 仿真

    #准备bin数据
    python3 src/make_rnn_input_onnx.py  --input /ts.knight-modelzoo/pytorch/builtin/audio/classification/lstm/data/cmd_12_test.pkl --outpath .
    #仿真
    Knight --chip TX5368AV200 rne-sim --input model_input.bin --weight lstm_quantize_r.weight --config  lstm_quantize_r.cfg --outpath .

### 4. 性能分析

```
Knight --chip TX5368AV200 rne-profiling --config  lstm_quantize_r.cfg --outpath .
```

### 5. 仿真库

### 6. 板端部署



## 支持芯片情况

| 芯片系列                                          | 是否支持 |
| ------------------------------------------------ | ------- |
| TX510x                                           | 支持     |
| TX5368x_TX5339x                                  | 支持     |
| TX5215x_TX5239x200_TX5239x220 | 支持     |
| TX5112x201_TX5239x201                            | 支持     |
| TX5336AV200                                      | 支持     |



## 版本说明

2023/11/23  第一版



## LICENSE

ModelZoo Edge 的 License 具体内容请参见LICENSE文件

## 免责说明

您明确了解并同意，以上链接中的软件、数据或者模型由第三方提供并负责维护。在以上链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。



