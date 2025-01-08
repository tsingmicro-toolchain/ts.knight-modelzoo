# VGG for Pytorch

<!--命名规则 {model_name}-{dataset}-{framework}-->

[TOC]

## 模型简介

VGG（Visual Geometry Group）是一个视觉几何组在2014年提出的深度卷积神经网络架构。
VGG在2014年ImageNet图像分类竞赛亚军，定位竞赛冠军；VGG网络采用连续的小卷积核（3x3）和池化层构建深度神经网络，网络深度可以达到16层或19层，其中VGG16和VGG19最为著名。

<!--可选-->论文：[《Very Deep Convolutional Networks for Large-Scale Image Recognition》](https://arxiv.org/abs/1409.1556)

开源模型链接：https://github.com/pytorch/vision/blob/v0.11.0/torchvision/models/vgg.py

数据集（ImageNet）：http://www.image-net.org/

## 资源准备

1. 数据集资源下载

	ImageNet是一个不可用于商业目的的数据集，必须通过教育网邮箱注册登录后下载, 请前往官方自行下载 [ImageNet](http://image-net.org/) 2012 val。

2. 模型资源下载

	下载[vgg16 pytorch权重](https://download.pytorch.org/models/vgg16-397923af.pth)

3. 清微github modelzoo仓库下载

	```git clone https://github.com/tsingmicro-toolchain/ts.knight-modelzoo.git```

## Knight环境准备

1. 联系清微智能获取Knight工具链版本包 ```ReleaseDeliverables/ts.knight-x.x.x.x.tar.gz ```。下面以ts.knight-3.0.0.11.build1.tar.gz为例演示。

2. 检查docker环境

	​默认服务器中已安装docker（版本>=19.03）, 如未安装可参考文档ReleaseDocuments/《TS.Knight-使用指南综述_V1.4.pdf》。
	
	```
	docker -v   
	```

3. 加载镜像
	
	```
	docker load -i ts.knight-3.0.0.11.build1.tar.gz
	```

4. 启动docker容器

	```
	docker run -v ${localhost_dir}/ts.knight-modelzoo:/ts.knight-modelzoo -it ts.knight:3.0.0.11.build1 /bin/bash
	```
	
	localhost_dir为宿主机目录。

## 快速体验
将下载好的数据放在`/ts.knight-modelzoo/pytorch/builtin/cv/classification/vgg16/data/imagenet/images/val`，并执行[valprep.sh](https://pan.baidu.com/s/12j74t9xiHWHiVD-pDILBng?pwd=li7r)脚本对图片进行分类。

在docker 容器内运行以下命令:

```
cd /ts.knight-modelzoo/pytorch/builtin/cv/classification/
```

```
sh vgg16/scripts/run.sh
```

## 模型移植流程

### 1. 量化&编译

-   模型准备
	
	如上述"Knight环境准备"章节所述，准备好vgg16 pytorch权重文件。
	

-   量化数据准备

    将下载好的数据放在`${localhost_dir}/ts.knight-modelzoo/pytorch/builtin/cv/classification/vgg16/data/imagenet/images/val`，并执行[valprep.sh](https://pan.baidu.com/s/1rAOzMAZhlN6sCvJMoBQROg?pwd=u2np)脚本对图片进行分类。在数据集中选200张图片作为量化校准数据集, 通过命令行参数```-i 200```指定图片数量，```-d```指定数据集路径。

-   模型转换函数、推理函数准备
	
	已提供量化依赖的模型转换和推理函数py文件: ```/ts.knight-modelzoo/pytorch/builtin/cv/classification/vgg16/src/vgg16.py```

-   执行量化命令

	在容器内执行如下量化和编译命令，具体量化、编译参数可见 vgg16_config.json

    	Knight --chip TX5368AV200 build --run-config data/vgg16_config.json
	
	量化后模型推理

    	Knight --chip TX5368AV200 quant --run-config data/vgg16_infer_config.json



### 2. 仿真

    #准备bin数据
    python src/make_image_input_onnx.py --input data/demo.jpg --outpath /TS-KnightDemo/Output/vgg16/npu
    #仿真
    Knight --chip TX5368AV200 run --run-config data/vgg16_config.json
	#仿真输出txt文件转numpy
	show_sim_result --sim-data /TS-KnightDemo/Output/vgg16/npu/result-*_p.txt --save-dir /TS-KnightDemo/Output/vgg16/npu/
	#模型后处理
	python src/post_process.py --numpys /TS-KnightDemo/Output/vgg16/npu/result-*_p.npy

### 4. 性能分析

```
Knight --chip TX5368AV200 profiling --run-config data/vgg16_config.json
```

### 5. 仿真库

### 6. 板端部署



## 支持芯片情况

| 芯片系列                                          | 是否支持 |
| ------------------------------------------------- | -------- |
| TX510x                                            | 支持     |
| TX5368x_TX5339x_TX5335x                           | 支持     |
| TX5215x_TX5239x200_TX5239x220_TX5239x300          | 支持     |
| TX5112x_TX5239x201                                | 支持     |
| TX5336x_TX5356x                                   | 支持     |



## 版本说明

2023/11/23  第一版



## LICENSE

ModelZoo Edge 的 License 具体内容请参见LICENSE文件

## 免责说明

您明确了解并同意，以上链接中的软件、数据或者模型由第三方提供并负责维护。在以上链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。



