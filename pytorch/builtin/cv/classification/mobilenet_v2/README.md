# MobilenetV2 for Pytorch

<!--命名规则 {model_name}-{dataset}-{framework}-->

[TOC]

## 模型简介

MobileNetV2是由谷歌开发的一种用于移动设备的轻量级卷积神经网络。与传统卷积神经网络相比，它具有更高的计算效率和更小的模型尺寸，可以在移动设备上实现高精度的图像识别任务。
MobileNetV2的主要原理是使用深度可分离卷积来减少模型的参数数量和计算量。深度可分离卷积将传统的卷积操作分解为两个独立的操作：深度卷积和逐点卷积。深度卷积仅在通道维度上进行卷积操作，而逐点卷积仅在空间维度上进行卷积操作。这种分解能大大降低计算复杂度，同时保持较高的分类精度。

<!--可选-->论文：[《MobileNetV2: Inverted Residuals and Linear Bottlenecks》](https://arxiv.org/abs/1801.04381)

开源模型链接：https://github.com/pytorch/vision/blob/v0.11.0/torchvision/models/mobilenetv2.py

数据集（ImageNet）：http://www.image-net.org/

## 资源准备

1. 数据集资源下载

	ImageNet是一个不可用于商业目的的数据集，必须通过教育网邮箱注册登录后下载, 请前往官方自行下载 [ImageNet](http://image-net.org/)  2012 val。

2. 模型资源下载

	下载[mobilenet_v2 pytorch权重](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)

3. 清微github modelzoo仓库下载

	```git clone https://github.com/tsingmicro-toolchain/ts.knight-modelzoo.git```

## Knight环境准备

1. 联系清微智能获取Knight工具链版本包 ```ReleaseDeliverables/ts.knight-x.x.x.x.tar.gz ```。下面以ts.knight-3.0.0.11.build1.tar.gz为例演示。

2. 检查docker环境

	​默认服务器中已安装docker（版本>=19.03）, 如未安装可参考文档ReleaseDocuments/《TS.Knight-使用指南综述_V3.0.11.pdf》。
	
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


## 模型移植流程

### 1. 量化&编译

-   模型准备
	
	如上述"Knight环境准备"章节所述，准备好mobilenet_v2 pytorch权重文件。
	

-   量化数据准备

    将下载好的数据放在`${localhost_dir}/ts.knight-modelzoo/pytorch/builtin/cv/classification/mobilenet_v2/data/imagenet/images/val`，并执行[valprep.sh](https://pan.baidu.com/s/1rAOzMAZhlN6sCvJMoBQROg?pwd=u2np)脚本对图片进行分类。

-   模型转换函数、推理函数准备
	
	已提供量化依赖的模型转换和推理函数py文件: ```/ts.knight-modelzoo/pytorch/builtin/cv/classification/mobilenet_v2/src/mobilenet_v2.py```

-   执行量化命令

	在容器内执行如下量化和编译命令，具体量化、编译参数可见 mobilenet_v2_config.json

    	Knight --chip TX5368AV200 build --run-config data/mobilenet_v2_config.json
	
	量化后模型推理

    	Knight --chip TX5368AV200 quant --run-config data/mobilenet_v2_infer_config.json



### 2. 仿真

    #准备bin数据
    python src/make_image_input_onnx.py --input data/demo.jpg --outpath /TS-KnightDemo/Output/mobilenet_v2/npu
    #仿真
    Knight --chip TX5368AV200 run --run-config data/mobilenet_v2_config.json
	#仿真输出txt文件转numpy
	show_sim_result --sim-data /TS-KnightDemo/Output/deit_tiny/npu/result-*_p.txt --save-dir /TS-KnightDemo/Output/mobilenet_v2/npu/
	#模型后处理
	python src/post_process.py --numpys /TS-KnightDemo/Output/mobilenet_v2/npu/result-*_p.npy

### 4. 性能分析

```
Knight --chip TX5368AV200 profiling --run-config data/mobilenet_v2_config.json
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



