# YOLOv6 for Pytorch

<!--命名规则 {model_name}-{dataset}-{framework}-->

[TOC]

## 模型简介

**YOLOv6**是美团技术团队研发的检测框架，其融合了其它YOLO版本的多种网络结构、训练策略、测试技术、量化和模型优化等思想，在精度和效率方面也有更大提升，这里以YOLOv6-s为例。

<!--可选-->
论文地址：[YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976)

Github工程地址：https://github.com/meituan/YOLOv6/tree/main

数据集（COCO）：https://cocodataset.org/

## 资源准备

1. 数据集资源下载

	COCO数据集是一个可用于图像检测（image detection），语义分割（semantic segmentation）和图像标题生成（image captioning）的大规模数据集。这里只需要下载coco128数据集。下载请前往[COCO官网](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip)。

2. 工程及模型权重下载
   工程地址：https://github.com/meituan/YOLOv6/tree/main
   权重地址：https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt


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

在docker 容器内运行以下命令:

```
cd /ts.knight-modelzoo/pytorch/builtin/cv/detection/
```

```
sh yolov6s/scripts/run.sh
```

## 模型部署流程

### 1. 量化&编译

-   模型准备
	已提供量化依赖的模型转换和推理函数py文件: ```/ts.knight-modelzoo/pytorch/builtin/cv/detection/将yolov6s/src/将yolov6s.py```，同时下载[工程](https://github.com/meituan/YOLOv6/tree/0.4.1)，放到src下。
	由于yolov6s的后处理已经放在infer函数中处理，所以需要对原工程目录下的[line128](https://github.com/meituan/YOLOv6/blob/e9656c307ae62032f40b39c7a7a5ccc31c2f0242/yolov6/models/heads/effidehead_distill_ns.py#L128) 增加如下一行代码：  
	`return cls_score_list, reg_lrtb_list`
	执行onnx转换：
	`python deploy/ONNX/export_onnx.py 
    --weights yolov6s.pt 
    --img 640 
    --batch 1 
    --simplify` 

-   量化数据准备

    这里使用[COCO128](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip)数据集作为量化校准数据集, 通过命令行参数```-i 128```指定图片数量,```-d```指定coco128.yaml所在的路径。如有找不到包问题，设置环境变量即可。

-   推理函数准备
	
	已提供量化依赖推理函数py文件: ```/ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov6s/src/yolov6s.py```

-   执行量化及编译命令

	在容器内执行如下量化命令，具体量化、编译参数可见yolov6s_config.json。

    	Knight --chip TX5368AV200 build --run-config data/yolov6s_config.json

-   量化后模型推理
	
		Knight --chip TX5368AV200 quant --run-config data/yolov6s_infer_config.json


### 2. 仿真

    #准备bin数据
    python src/make_image_input_onnx.py --input test_data/bus.jpg --outpath /TS-KnightDemo/Output/yolov6s/npu

    #仿真
    Knight --chip TX5368AV200 run --run-config data/yolov6s_config.json

	#仿真输出txt文件转numpy
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov6s/npu/result-outputs_p.txt --save-dir /TS-KnightDemo/Output/yolov6s/npu/
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov6s/npu/result-451_p.txt --save-dir /TS-KnightDemo/Output/yolov6s/npu/
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov6s/npu/result-491_p.txt --save-dir /TS-KnightDemo/Output/yolov6s/npu/
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov6s/npu/result-419_p.txt --save-dir /TS-KnightDemo/Output/yolov6s/npu/
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov6s/npu/result-459_p.txt --save-dir /TS-KnightDemo/Output/yolov6s/npu/
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov6s/npu/result-499_p.txt --save-dir /TS-KnightDemo/Output/yolov6s/npu/

	#模型后处理。 scales为模型输出top_scale，需要根据实际量化结果指定该值
	python src/post_process.py --image test_data/bus.jpg --img-size 640 --numpys \
	/TS-KnightDemo/Output/yolov6s/npu/result-outputs_p.npy \
	/TS-KnightDemo/Output/yolov6s/npu/result-451_p.npy \
	/TS-KnightDemo/Output/yolov6s/npu/result-491_p.npy \
	/TS-KnightDemo/Output/yolov6s/npu/result-419_p.npy \
	/TS-KnightDemo/Output/yolov6s/npu/result-459_p.npy \
	/TS-KnightDemo/Output/yolov6s/npu/result-499_p.npy \
	--scales  0.002610747 0.003320219 0.003782163 0.09804556 0.05281715 0.05505726 --save_dir /TS-KnightDemo/Output/yolov6s/npu/

### 3. 性能分析

```
Knight --chip TX5368AV200 profiling --run-config data/yolov6s_config.json
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




