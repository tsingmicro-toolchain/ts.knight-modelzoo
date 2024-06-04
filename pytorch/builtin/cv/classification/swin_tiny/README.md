# swin-tiny for Pytorch

<!--命名规则 {model_name}-{dataset}-{framework}-->

[TOC]

## 模型简介

**Swin Transformer**(Swin这个名字代表移位窗口)最初是在arxiv中描述的，它能够作为计算机视觉的通用骨干。它基本上是一个分层的Transformer，其表示是用移位窗口计算的。移位窗口方案将自关注计算限制在不重叠的局部窗口，同时允许跨窗口连接，从而提高了效率。

<!--可选-->论文：[《Swin Transformer: Hierarchical Vision Transformer using Shifted Windows》](http://arxiv.org/abs/2103.14030)

开源模型链接：https://github.com/microsoft/Swin-Transformer.git

数据集（ImageNet）：http://www.image-net.org/

## 资源准备

1. 数据集资源下载

	ImageNet是一个不可用于商业目的的数据集，必须通过教育网邮箱注册登录后下载, 请前往官方自行下载[ImageNet](http://image-net.org/)。

2. 模型资源下载

	下载[Swin-Transformer](https://github.com/microsoft/Swin-Transformer.git)工程和[Swin-T pytorch权重](https://objects.githubusercontent.com/github-production-release-asset-2e65be/357198522/fd006b80-9bd3-11eb-8445-769d89efab4e?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231222%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231222T032029Z&X-Amz-Expires=300&X-Amz-Signature=9d5195f7613c830d16dabaf81df4aa85ebcc7ebab6d59b1ae775df22bb9f39dc&X-Amz-SignedHeaders=host&actor_id=12314280&key_id=0&repo_id=357198522&response-content-disposition=attachment%3B%20filename%3Dswin_tiny_patch4_window7_224.pth&response-content-type=application%2Foctet-stream)

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
cd /ts.knight-modelzoo/pytorch/builtin/cv/classification/
```

```
sh swin_tiny/scripts/run.sh
```

## 模型移植流程

### 1. 量化

-   模型准备
	
	如上述"Knight环境准备"章节所述，准备好Swin-T pytorch权重文件。
	

-   量化数据准备

    在数据集中选200张图片作为量化校准数据集, 通过命令行参数```-i 200```指定图片数量，```-d```指定数据集路径。

-   模型转换函数、推理函数准备
	
	已提供量化依赖的模型转换和推理函数py文件: ```/ts.knight-modelzoo/pytorch/builtin/cv/classification/swin_tiny/src/swin_tiny.py```

-   执行量化命令

	在容器内执行如下量化命令，生成量化后的文件 swin_tiny_quantize.onnx 存放在 -s 指定输出目录。

    	Knight --chip TX5368AV200 quant onnx -m swin_tiny
    		-w /ts.knight-modelzoo/pytorch/builtin/cv/classification/swin_tiny/weight/swin_tiny_patch4_window7_224.pth 
    		-f pytorch 
    		-uds /ts.knight-modelzoo/pytorch/builtin/cv/classification/swin_tiny/src/swin_tiny.py 
    		-if infer_imagenet_benchmark 
			-s ./tmp/swin_tiny 
    		-d /ts.knight-modelzoo/pytorch/builtin/cv/classification/swin_tiny/data/imagenet/images/val 
    		-bs 1 -i 200


### 2. 编译


    Knight --chip TX5368AV200 rne-compile --onnx swin_tiny_quantize.onnx --outpath swin_tiny_example/compile_result


### 3. 仿真

    #准备bin数据
    python3 make_swin_tiny_input_bin.py.py  
    #仿真
    Knight --chip TX5368A rne-sim --input input.bin --weight _quantize_r.weight --config  _quantize_r.cfg --outpath swin_tiny_example

### 4. 性能分析

```
Knight --chip TX5368A rne-profiling --weight  _r.weight --config  _r.cfg --outpath  swin_tiny_example/
```

### 5. 仿真库

### 6. 板端部署



## 支持芯片情况

| 芯片系列                                          | 是否支持 |
| ------------------------------------------------ | ------- |
| TX231x                                           | 支持     |
| TX232                                            | 不支持   |
| TX510x                                           | 支持     |
| TX5368x_TX5339x                                  | 支持     |
| TX5215x_TX5119x_TX5112x200_TX5239x200_TX5239x220 | 支持     |
| TX5112x201_TX5239x201                            | 支持     |
| TX5336AV200                                      | 支持     |



## 版本说明

2023/11/23  第一版



## LICENSE

ModelZoo Edge 的 License 具体内容请参见LICENSE文件

## 免责说明

您明确了解并同意，以上链接中的软件、数据或者模型由第三方提供并负责维护。在以上链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。



