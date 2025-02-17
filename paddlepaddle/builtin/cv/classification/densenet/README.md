# DenseNet for PaddlePaddle

<!--命名规则 {model_name}-{dataset}-{framework}-->

[TOC]

## 模型简介

从ResNet开始，有很多研究都是琢磨怎么通过提高深度网络中层与层之间的信息传递(包括正向和反向传播)来解决梯度消失的问题。 ResNet是通过增加一个Short Connection的恒等映射来传递数据，并将这个数据与卷积层输出的残差相加来作为下一层的输入。 Stochasic Depth方法就是通过在反向传播中(训练过程)随机的丢掉一些层来提供训练时的梯度。

DenseNet提出了一个新的思路：把每一层的输出，特别是浅层网络的输出做一个连接到后面所有的层，这样来保证有效的利用前面层的有效信息(因为前面层毕竟没有那么多的信息丢失)。而且不是同通过相加，而是通过featrue map的叠加方式进行。这种方式叫做featrue reuse

<!--可选-->论文：[《Densely Connected Convolutional Networks》](https://arxiv.org/abs/1608.06993)

开源模型链接：https://github.com/liuzhuang13/DenseNet

数据集（ImageNet）：http://www.image-net.org/

## 资源准备

1. 数据集资源下载

	ImageNet是一个不可用于商业目的的数据集，必须通过教育网邮箱注册登录后下载, 请前往官方自行下载 [ImageNet](http://image-net.org/) 2012 val。

2. 模型资源下载

	百度网盘： [DenseNet121_pretrained.pdmodel,DenseNet121_pretrained.pdiparams](https://pan.baidu.com/s/1A7zqiaY-QKJVgOR4VmtsjA?pwd=c5ww)

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
	
	如上述"Knight环境准备"章节所述，准备好densenet重文件。
	

-   量化数据准备

    将下载好的数据放在`${localhost_dir}/ts.knight-modelzoo/pytorch/builtin/cv/classification/densenet/data/imagenet/images/val`，并执行[valprep.sh](https://pan.baidu.com/s/1rAOzMAZhlN6sCvJMoBQROg?pwd=u2np)脚本对图片进行分类。

-   模型转换函数、推理函数准备
	
	已提供量化依赖的模型转换和推理函数py文件: ```/ts.knight-modelzoo/paddlepaddle/builtin/cv/classification/densenet/src/densenet.py```

-   执行量化命令

	在容器内执行如下量化和编译命令，具体量化、编译参数可见 densenet_config.json

    	Knight --chip TX5368AV200 build --run-config data/densenet_config.json
	
	量化后模型推理

    	Knight --chip TX5368AV200 quant --run-config data/densenet_infer_config.json



### 2. 仿真

    #准备bin数据
    python src/make_image_input_onnx.py --input data/demo.jpg --outpath /TS-KnightDemo/Output/densenet/npu
    #仿真
    Knight --chip TX5368AV200 run --run-config data/densenet_config.json
	#仿真输出txt文件转numpy
	show_sim_result --sim-data /TS-KnightDemo/Output/densenet/npu/result-*_p.txt --save-dir /TS-KnightDemo/Output/densenet/npu/
	#模型后处理
	python src/post_process.py --numpys /TS-KnightDemo/Output/densenet/npu/result-*_p.npy

### 4. 性能分析

```
Knight --chip TX5368AV200 profiling --run-config data/densenet_config.json
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



