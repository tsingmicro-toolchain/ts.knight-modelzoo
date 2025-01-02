# YOLOv8 for Pytorch

<!--命名规则 {model_name}-{dataset}-{framework}-->

[TOC]

## 模型简介


<!--可选-->
论文地址：[Real-Time Flying Object Detection with YOLOv8](https://arxiv.org/abs/2305.09972)

Github工程地址：https://github.com/ultralytics/ultralytics

数据集（COCO）：https://cocodataset.org/

## 资源准备

1. 数据集资源下载

	COCO数据集是一个可用于图像检测（image detection），语义分割（semantic segmentation）和图像标题生成（image captioning）的大规模数据集。这里需要下载coco128数据集。下载请前往[COCO官网](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip)。
2. 模型权重下载

	下载[YOLOv8s权重](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt)

3. 清微github modelzoo仓库下载

	```git clone https://github.com/tsingmicro-toolchain/ts.knight-modelzoo.git```

## Knight环境准备

1. 联系清微智能获取Knight工具链版本包 ```ReleaseDeliverables/ts.knight-x.x.x.x.tar.gz ```。下面以ts.knight:3.0.0.11.build1.tar.gz为例演示。

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
sh yolov8s/scripts/run.sh
```

## 模型部署流程

### 1. 量化 & 编译

-   模型准备 
	以版本[ultralytics-8.3.23](https://github.com/ultralytics/ultralytics/tree/v8.3.23)为例，

	 ![head.py](https://i-blog.csdnimg.cn/direct/fb18f4b695854ea7be7adf7ac76abd06.png)

	在ultralytics/nn/modules/head.py里_inference函数修改如上图所示，并将该工程放在src目录下。

-   量化数据准备

    这里使用[COCO128](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip)数据集作为量化校准数据集, 通过命令行参数```-i 128```指定图片数量,```-d```指定coco128.yaml所在的路径。

-   模型转换函数、推理函数准备
	
	已提供量化依赖的模型转换和推理函数py文件: ```/ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov8s/src/yolov8s.py```

-   执行量化命令及编译

	在容器内执行如下量化命令，具体量化、编译参数可见yolov8s_config.json。

    	Knight --chip TX5368AV200 build --run-config data/yolov8s_config.json

-   量化后模型推理
	
		Knight --chip TX5368AV200 quant --run-config data/yolov8s_infer_config.json


### 2. 仿真

    #准备bin数据
    python3 src/make_image_input_onnx.py --input /ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov8s/test_data/ --outpath /TS-KnightDemo/Output/yolov8s/npu

    #仿真
    Knight --chip TX5368AV200 run --run-config data/yolov8s_config.json

	#仿真输出txt文件转numpy
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov8s/npu/result-699_p.txt --save-dir /TS-KnightDemo/Output/yolov8s/npu/
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov8s/npu/result-719_p.txt --save-dir /TS-KnightDemo/Output/yolov8s/npu/
	show_sim_result --sim-data /TS-KnightDemo/Output/yolov8s/npu/result-739_p.txt --save-dir /TS-KnightDemo/Output/yolov8s/npu/

	#模型后处理。 scales为模型输出top_scale，需要根据实际量化结果指定该值
    python src/post_process.py --image test_data/bus.jpg --img-size 640 --numpys /TS-KnightDemo/Output/yolov8s/npu/result-699_p.npy /TS-KnightDemo/Output/yolov8s/npu/result-719_p.npy /TS-KnightDemo/Output/yolov8s/npu/result-739_p.npy --scales  0.2689846 0.4109275  0.4786172 --save_dir output

### 3. 性能分析

```
Knight --chip TX5368AV200 profiling --run-config data/yolov8s_config.json
```

### 4. 仿真库

### 5. 板端部署

板端部署时,为了提升推理性能，需要删除--output-dequant进行量化，并且在后处理中增加反量化步骤，以保证流程的正确性。

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




