# YOLOX for Pytorch

<!--命名规则 {model_name}-{dataset}-{framework}-->

[TOC]

## 模型简介

<!--可选-->
论文地址：[YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)

Github工程地址：https://github.com/Megvii-BaseDetection/YOLOX

数据集（COCO）：https://cocodataset.org/

## 资源准备

1. 数据集资源下载

	COCO数据集是一个可用于图像检测（image detection），语义分割（semantic segmentation）和图像标题生成（image captioning）的大规模数据集。下载请前往[COCO官网](https://cocodataset.org)下载验证集 [images](http://images.cocodataset.org/zips/val2017.zip) 以及[annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) 。

2. 模型权重下载

	下载[YOLOxs权重](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)

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
sh yoloxs/scripts/run.sh
```

## 模型部署流程

### 1. 量化&编译

-   量化数据准备

    将以上验证下载解压后按以下路径存放：

		data
			annotations
				instances_val2017.json
			val2017
				*.jpg
				*.jpg
				...	



-   推理函数准备


	![alt text](image.png)
	
    如上图修改yolox/models/yolo_head.py，并将改工程放在src目录下。

    执行`python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth` 进行onnx模型转换。
    
    已提供量化依赖的模型推理函数py文件: ```/ts.knight-modelzoo/pytorch/builtin/cv/detection/yoloxs/src/yolox_s.py```。

-   执行量化及编译命令

	在容器内执行如下量化命令，具体量化、编译参数可见yolox_s_config.json。

    	Knight --chip TX5368AV200 build --run-config data/yolox_s_config.json

-   量化后模型推理
	
		Knight --chip TX5368AV200 quant --run-config data/yolox_s_infer_config.json


### 2. 仿真


    #准备bin数据
    python src/make_image_input_onnx.py --input test_data/bus.jpg --outpath /TS-KnightDemo/Output/yolox_s/npu

    #仿真
    Knight --chip TX5368AV200 run --run-config data/yolox_s_config.json

	#仿真输出txt文件转numpy
	show_sim_result --sim-data /TS-KnightDemo/Output/yolox_s/npu/result-817_p.txt --save-dir /TS-KnightDemo/Output/yolox_s/npu/
	show_sim_result --sim-data /TS-KnightDemo/Output/yolox_s/npu/result-843_p.txt --save-dir /TS-KnightDemo/Output/yolox_s/npu/
	show_sim_result --sim-data /TS-KnightDemo/Output/yolox_s/npu/result-869_p.txt --save-dir /TS-KnightDemo/Output/yolox_s/npu/

	#模型后处理。 scales为模型输出top_scale，需要根据实际量化结果指定该值
    python src/post_process.py --image test_data/bus.jpg --numpys  /TS-KnightDemo/Output/yolox_s/npu/result-817_p.npy /TS-KnightDemo/Output/yolox_s/npu/result-843_p.npy /TS-KnightDemo/Output/yolox_s/npu/result-869_p.npy  --scales 0.02812371 0.02926355 0.02422856 --save_dir output

### 3. 性能分析

	Knight --chip TX5368AV200 profiling --run-config data/yolox_s_config.json


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




