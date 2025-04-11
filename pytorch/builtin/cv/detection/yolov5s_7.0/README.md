# YOLOv5s-7.0 for Pytorch

<!--命名规则 {model_name}-{dataset}-{framework}-->

[TOC]

## 模型简介

**YOLOv5** 是一种在图像检测中广泛使用的深度卷积神经网络，其速度快和检测精度高，是最著名的物体检测算法之一。YOLOv5已有7个衍生版本，每个版本又根据参数两分为n、s、m、l、x五种类型，以下其最新版YOLOv5s-7.0为例。

<!--可选-->
算法介绍：https://docs.ultralytics.com/yolov5/

开源模型链接：https://github.com/ultralytics/yolov5/releases/tag/v7.0

数据集（COCO）：https://cocodataset.org/

## 资源准备

1. 数据集资源下载

	COCO数据集是一个可用于图像检测（image detection），语义分割（semantic segmentation）和图像标题生成（image captioning）的大规模数据集。这里需要下载coco128数据集。下载请前往[COCO官网](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip)。

2. 模型权重下载

	下载[权重](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)

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



## 模型部署流程

### 1. 量化&编译

-   模型准备
	
	如上述"Knight环境准备"章节所述，准备好yolov5s-7.0的pytorch权重文件。
	

-   量化数据准备

    这里使用[COCO128](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip)数据集作为量化校准数据集。

-   模型转换函数、推理函数准备
	
	已提供量化依赖的模型转换和推理函数py文件: ```/ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov5s_7.0/src/yolov5s_7_0.py```。同时下载[工程](https://github.com/ultralytics/yolov5/releases/tag/v7.0)，放到src下。

-   执行量化命令
  
    模型输入shape默认为（1，3，640，640），如需修改，请修改`yolov5s_7_0.py`中IMAGE_SIZE定义：
    ![alt text](image1.png)

	在容器内执行如下量化和编译命令，具体量化、编译参数可见 yolov5s_config.json

    	Knight --chip TX5368AV200 build --run-config data/yolov5s_config.json
	
-   量化后模型推理

    首先需要修改yolov5s_7_0.py反量化系数：

	![alt text](image.png)

    	Knight --chip TX5368AV200 quant --run-config data/yolov5s_infer_config.json



### 2. 仿真

    #准备bin数据
    python src/make_image_input_onnx.py --input test_data/bus.jpg --outpath /TS-KnightDemo/Output/yolov5s_7.0/npu
    #仿真
    Knight --chip TX5368AV200 run --run-config data/yolov5s_config.json
	#仿真输出txt文件转numpy
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov5s_7.0/npu/result-357_p.txt --save-dir /TS-KnightDemo/Output/yolov5s_7.0/npu/
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov5s_7.0/npu/result-358_p.txt --save-dir /TS-KnightDemo/Output/yolov5s_7.0/npu/
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov5s_7.0/npu/result-359_p.txt --save-dir /TS-KnightDemo/Output/yolov5s_7.0/npu/
	#模型后处理
	python src/post_process.py --image test_data/bus.jpg --numpys /TS-KnightDemo/Output/yolov5s_7.0/npu/result-357_p.npy /TS-KnightDemo/Output/yolov5s_7.0/npu/result-358_p.npy /TS-KnightDemo/Output/yolov5s_7.0/npu/result-359_p.npy --scales 0.1755103 0.1281193 0.1337741 --save_dir /TS-KnightDemo/Output/yolov5s_7.0/npu/

### 3. 性能分析

```
Knight --chip TX5368AV200 profiling --run-config data/yolov5s_config.json
```

### 4. 仿真库

### 5. 板端部署



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



