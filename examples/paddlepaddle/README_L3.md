## 目录
- [1. Benchmark](#1-benchmark数据集来自官网提供和整理)
- [2. 环境准备](#2-环境准备)
- [3. 分类网络](#3-分类网络)
- [4. 检测网络](#4-检测网络)
- [5. 分割网络](#5-分割网络)
- [6. 推荐网络](#6-推荐网络)
- [7. NLP网络](#7-NLP网络)
- [8. 硬件测试](#8-硬件测试)
- [9. 其他说明](#9-其他说明)

----

## 1. Benchmark(数据集来自官网提供和整理)

|  模型名称    | paddle浮点精度 | onnx浮点精度 | 量化精度(掉点)  | 测试数据量 | 推理速度(单张图片/ms) | 说明 |
|-------------|-----------|----------|----------|-----------|--------------------|--------|
| 1. pphgnet_tiny     | 78.75      | 78.75      | 78.9(+0.15)    | 640     |    12.06       | [官方模型](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.5/docs/zh_CN/models/ImageNet1k) |
| 2. regnetx_4gf     | 83.9      | 83.9      | 83.0(-0.9)  | 640     |     14.64       | [官方模型](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.5/docs/zh_CN/models/ImageNet1k) |
| 3. resnet50_vd     | 75.47     | 75.47       | 75.16(-0.3)     | 640    |    10.4       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 4. peleenet     | 69.84      | 69.84     | 69.06(-0.8)    | 640    |    4.9       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 5. dla60x     |   79.22   | 79.22     |  78.6(-0.6)    | 640    |    13       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 6. resnext50_32x4d     |  81.1    |  81.1    |  80.3(-0.8)    | 640    |    13.5       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 7. se_resnext50_32x4d    |   81.56   |   81.56   |  80.78(-0.8)    | 640    |    21.8       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 8. xception41     |   79.21   |  79.21    |  79.06(-0.16)    | 640    |    25.2       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 9. mobilenetv1_x0_75   |  65.78    |  65.78    |  65.3(-0.48)    | 640    |    2.6      | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 10. hardnet68     |   77.18   |  77.18    |   76.72(-0.4)   | 640    |     85.8      | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 11. cspdarknet53     |   75.15   |   75.15   |   75.16(+0.1)   | 640    |   11.5       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 12. shufflenetv2_x2_0    |  74.68    |  74.68    |  74.68(-0.0)    | 640    |    7.8      | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 13. hrnet_w18_c     |   75.31   |  75.31    |  75.47(+0.15)    | 640    |    16.7      | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 14. inceptionv3    |  71.4    |   71.4   |   71.1(-0.3)   | 640    |    9.8       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 15. vgg13     |  68.28    |   68.28   |  67.65(-0.6)    | 640    |    31.2       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 16. tinybert     |   91.5   |  91.5    |   91.16(-0.4)   | 872    |    41.7       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 17. electra     |   91.28   |  91.28    |   90.6(-0.6)   | 872    |    32.7       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 18. yolov3_mobilenetv1     |   0.506   |  0.506    |  0.499(-0.007)    | 5000    |    25.5       | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |
| 19. humansegv1_lite     |   86.0   |  86.0    |   85.9(-0.1)   | 100    |    7.7       | [官方模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/contrib/PP-HumanSeg/README_cn.md) |
| 20. dcn     |   84.3   |   84.3   |  84.1(-0.2)    |  5120    |    0.05      | [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) |

----

## 2. 环境准备

- 请联系[技术支持](../../README.md#技术讨论)获取需要用到的VPN、账号、knight docker镜像（百度提测无需该步）
- 配置远程登录清微内网环境：
  - 安装VPN，解压‘OPENVPN - ITO.zip’，参考其中文档进行安装，VPN用户名：wx_paddle   VPN密码：x1fykfb1Qt
  - 安装VMware-Horizon，参考文档《清微-vdi》，vdi用户名：wx_paddle   vdi密码：1234qwe!@#
  - 通过VMware-Horizon登录远程虚拟机，进入后通过MobaXterm登录服务器192.168.1.10（ssh wx_paddle@192.168.1.10），用户名：wx_paddle  密码：123qwe!@#
- 当前目录下启动docker容器:

  ```
  docker load -i TS.Knight-1.3.0.7-for-paddle.tar.gz
  docker run -w /TS-Knight/Quantize/Onnx/onnx_quantize_tool -v /data/examples/baidu_qa_level3:/data/examples/baidu_qa_level3 --ipc=host -it knight-1.3.0.7-for-paddle:1.0 /bin/bash
  ```
- **启动容器后，使用上述命令会自动切换目录到/TS-Knight/Quantize/Onnx/onnx_quantize_tool（没有的话请执行cd命令）**
- **数据和模型：** 模型和数据已被打包到镜像（容器）目录 /TS-Knight/Quantize/Onnx/onnx_quantize_too下的data和paddle_models文件夹；
- 工作目录说明：<br>
  ![image](https://user-images.githubusercontent.com/7539692/227886407-84648b1e-1ea6-47fd-b916-aea0b01049bc.png)
- 量化命令（集成转换工具）解析：
  - 举例：如下命令，会将paddle模型转为onnx，并对onnx模型进行量化定点；该过程输出转换浮点精度和量化精度
    ``` 
    python run_quantization.py -f paddle -r all -ch TX511 -ct TX511 -od -if infer_ocr_det_model -m example/models/ch_PP-OCRv3_det_infer_512x896/ocrv3_det.pdmodel -w example/models/ch_PP-OCRv3_det_infer_512x896/ocrv3_det.pdiparams -s /my_project/quant/ocrv3_det/ -bs 1 -i 50 -qm kl -is 1 3 512 896  --dump 
    ```
    
  - 主要参数解析：
    - "-f paddle"：指定要转换的框架为paddle；"-r all"：表示执行转换、量化、转IR3个过程；"-od"：表示输出反量化为浮点
    - "-m xxx.pdmodel"：指定paddle模型路径；"-w xxx.pdiparams"：指定paddle模型权重路径；"-s xxx"：指定量化模型保存路径；
    - "-qm kl"：指定量化系数计算方式；"-is"：指定输入形状，不可随意指定；"--dump"：指定保存浮点结果、量化结果等用于对比；

----

## 3. 分类网络

- __网络说明__
  - 该部分共15个网络；该类网络对目标图片进行分类，指标为 accuracy（简称acc）
  - 以下以PPHGNet_tiny为例，逐步介绍该部分网络的工作流程及执行命令，该部分其他网络可类似执行（网络数量多，不再一一列举，见谅）；
  - ***读者也可在 /TS-Knight/Quantize/Onnx/onnx_quantize_tool 目录下直接执行 bash scripts/scripts_cls/pphgnet_tiny.sh 运行该网络所有命令；其他网络也有相应脚本；执行scripts/scripts_cls/scripts_cls_all.sh可一次性运行所有分类网络（逐个）***
  - 执行 scripts/scripts_cls/scripts_cls_all.sh 时可以使用nohup命令，如 nohup bash scripts/scripts_cls/scripts_cls_all.sh > logs/run_all_cls.log，然后查看logs/run_all_cls.log记录的运行情况；防止模型较多导致旧的输出看不到；执行该部分所有网络大概需要1天多的时间；


- __数据预处理__
  - 无需额外预处理；数据来自imagenet，存放于/TS-Knight/Quantize/Onnx/onnx_quantize_tool/data/imagenet

- __量化__

  ```
  # 创建模型目录
  mkdir -p /my_project/quant/pphgnet_tiny
  mkdir -p /my_project/quant/to_compiler/pphgnet_tiny

  # 获取飞桨模型
  # /TS-Knight/Quantize/Onnx/onnx_quantize_tool/paddle_models/PPHGNet_tiny

  # 执行量化命令,完成paddle2onnx转换，并对模型进行量化定点
  /usr/bin/python -u run_quantization.py -qm kl_v2 -du -if infer_cls_model -f paddle -ct TX511 -ch TX511 -i 10 -bs 64 -is 64 3 224 224 -r all -m paddle_models/PPHGNet_tiny/pphgnet_tiny.pdmodel -w paddle_models/PPHGNet_tiny/pphgnet_tiny.pdiparams -s /my_project/quant/pphgnet_tiny -du

  # cpu上执行推理命令,对转换模型（或量化模型）在整个测试集上进行推理
  # 注：可以跳该部分推理，因为此处量化数据集和全集相同；上一条命令量化时已经跑过
  # python run_quantization.py -r infer -i 10 -bs 64 -ch TX511 -m /my_project/quant/pphgnet/pphgnet.onnx -if infer_cls_model
  # python run_quantization.py -r infer -i 10 -bs 64 -ch TX511 -m /my_project/quant/pphgnet/pphgnet_quantize.onnx -if infer_cls_model

  # 拷贝模型到指定目录
  cp /my_project/quant/pphgnet_tiny/pphgnet_tiny_quantize.onnx /my_project/quant/to_compiler/pphgnet_tiny/
  cp -r /my_project/quant/pphgnet_tiny/caffe_model /my_project/quant/to_compiler/pphgnet_tiny/
  cp /my_project/quant/pphgnet_tiny/dump/float/0001\:Input_x/batch_0.npy /my_project/quant/to_compiler/pphgnet_tiny/input.npy
  cp /my_project/quant/pphgnet_tiny/dump/quant/0069\:Softmax_softmax_1.tmp_0/batch_0.npy /my_project/quant/to_compiler/pphgnet_tiny/output.npy

  # 说明：以下是量化数据集及全量数据集上的浮点、量化结果；此处前后两对数值一样是正常的，因为此处量化数据集和全量数据集是相同的数据；
  ```
<div align=center>
浮点结果<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/49f808f9-d1c8-4288-a853-952f0e020db6/>
<br>量化结果<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/5e2984e8-e90f-4693-933c-00c787db419e
 />
<br>浮点结果（全量数据）<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/49f808f9-d1c8-4288-a853-952f0e020db6 />
<br>量化结果（全量数据）<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/5e2984e8-e90f-4693-933c-00c787db419e />
  

</div>

- __编译__

  ```
  mkdir -p /my_project/quant/to_compiler/pphgnet_tiny/compile_dir

  # 编译网络(使用caffe编译)
  Knight --chip TX5368AV200 rne-compile --net /my_project/quant/pphgnet_tiny/caffe_model/pphgnet_tiny/pphgnet_tiny.prototxt --weight /my_project/quant/pphgnet_tiny/caffe_model/pphgnet_tiny/pphgnet_tiny.weight --outpath /my_project/quant/to_compiler/pphgnet_tiny/compile_dir/ --general-process 1

  cp -r /my_project/quant/to_compiler/ /data/examples/baidu_qa_level3
  ```

----

## 4. 检测网络

- __网络说明__

  - 该部分共1个网络；该类网络对目标图片进行物体检测，指标为 IoU_.5
  - 以下逐步介绍该网络的工作流程及执行命令；
  - ***读者也可以在 /TS-Knight/Quantize/Onnx/onnx_quantize_tool 目录下直接执行 bash scripts/scripts_det/yolov3_mbv1_270e_coco_416.sh 运行该网络所有命令；***


- __数据预处理__
  - 无需额外预处理；数据来自coco，存放于/TS-Knight/Quantize/Onnx/onnx_quantize_tool/data/data_coco 

- __量化__

  ```
  # 创建模型目录
  mkdir -p /my_project/quant/yolov3_mobilenetv1
  mkdir -p /my_project/quant/to_compiler/yolov3_mobilenetv1

  # 获取飞桨模型
  # /TS-Knight/Quantize/Onnx/onnx_quantize_tool/paddle_models/yolov3_mobilenet_v1_270e_coco

  # 执行量化命令,完成paddle2onnx转换，并对模型进行量化定点
  /usr/bin/python -u run_quantization.py -ct TX511 -ch TX511 -m paddle_models/yolov3_mobilenet_v1_270e_coco/yolov3_mobilenetv1.pdmodel -w paddle_models/yolov3_mobilenet_v1_270e_coco/yolov3_mobilenetv1.pdiparams -f paddle -r all -b 8 -if infer_yolo3_model -qm kl_v2 -i 5000 -od -du -s /my_project/quant/yolov3_mobilenetv1

  # 执行推理命令,对转换模型(or 量化模型)在整个测试集上进行推理
  # 注：此处量化数据集即是全集数据集，因此可跳过该步
  # python run_quantization.py -r infer -ch TX511 -ct TX511 -i 5000 -bs 1 -if infer_yolo3_model -m /my_project/quant/yolov3_mobilenetv1/yolov3_mbv1_270e_coco_416.onnx
  # python run_quantization.py -r infer -ch TX511 -ct TX511 -i 5000 -bs 1 -if infer_yolo3_model -m /my_project/quant/yolov3_mobilenetv1/yolov3_mobilenetv1_quantize.onnx

  # 拷贝模型到指定目录
  cp /my_project/quant/yolov3_mobilenetv1/yolov3_mobilenetv1_quantize.onnx /my_project/quant/to_compiler/yolov3_mobilenetv1/
  cp -r /my_project/quant/yolov3_mobilenetv1/caffe_model /my_project/quant/to_compiler/yolov3_mobilenetv1/
  cp /my_project/quant/yolov3_mobilenetv1/dump/float/0001\:Input_image/batch_0.npy /my_project/quant/to_compiler/yolov3_mobilenetv1/input.npy
  cp /my_project/quant/yolov3_mobilenetv1/dump/quant/0044\:Conv_conv2d_84.tmp_1/batch_0.npy /my_project/quant/to_compiler/yolov3_mobilenetv1/output1.npy
  cp /my_project/quant/yolov3_mobilenetv1/dump/quant/0061\:Conv_conv2d_85.tmp_1/batch_0.npy /my_project/quant/to_compiler/yolov3_mobilenetv1/output2.npy
  cp /my_project/quant/yolov3_mobilenetv1/dump/quant/0075\:Conv_conv2d_86.tmp_1/batch_0.npy /my_project/quant/to_compiler/yolov3_mobilenetv1/output3.npy

  # 说明：以下是量化数据集及全量数据集上的浮点、量化结果；此处前后两对数值一样是正常的，因为此处量化数据集和全量数据集是相同的数据；
  ```
<div align=center>
浮点结果<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/e9c14ca6-1113-400d-a6a6-ac11b26c6dd4
 />  
<br>量化结果<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/69ed11c0-6a2d-4176-802c-ff1b64128b2e
 />
<br>浮点结果（全量数据）<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/e9c14ca6-1113-400d-a6a6-ac11b26c6dd4 />
<br>量化结果（全量数据）<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/69ed11c0-6a2d-4176-802c-ff1b64128b2e />
</div>

- __编译__

  ```
  mkdir -p /my_project/quant/to_compiler/yolov3_mbv1_270e_coco_416/compile_dir

  # 编译网络(使用caffe编译)
  Knight --chip TX5368AV200 rne-compile --net /my_project/quant/yolov3_mobilenetv1/caffe_model/yolov3_mobilenetv1/yolov3_mobilenetv1.prototxt --weight /my_project/quant/yolov3_mobilenetv1/caffe_model/yolov3_mobilenetv1/yolov3_mobilenetv1.weight --outpath /my_project/quant/to_compiler/yolov3_mobilenetv1/compile_dir/ --general-process 1

  cp -r /my_project/quant/to_compiler/ /data/examples/baidu_qa_level3
  ```

----

## 5. 分割网络

- __网络说明__

  - 该部分共1个网络；该类网络对目标图片进行物体分割，指标为 miou
  - 以下逐步介绍该网络的工作流程及执行命令；
  - ***读者也可以在 /TS-Knight/Quantize/Onnx/onnx_quantize_tool 目录下直接执行 bash scripts/scripts_seg/humansegv1_lite_reshape.sh 运行该网络所有命令；***


- __数据预处理__
  - 无需额外预处理；数据来自mini_supervisely，存放于/TS-Knight/Quantize/Onnx/onnx_quantize_tool/data/mini_supervisely

- __量化__

  ```
  # 创建模型目录
  mkdir -p /my_project/quant/humansegv1_lite
  mkdir -p /my_project/quant/to_compiler/humansegv1_lite

  # 获取飞桨模型
  # /TS-Knight/Quantize/Onnx/onnx_quantize_tool/paddle_models/human_pp_humansegv1_lite_192x192_pretrained/

  # 执行量化命令,完成paddle2onnx转换，并对模型进行量化定点
  /usr/bin/python -u run_quantization.py -f paddle -i 10 -ch TX511 -ct TX511 -if infer_humanseg_model -m paddle_models/human_pp_humansegv1_lite_192x192_pretrained/humansegv1_lite.pdmodel -w paddle_models/human_pp_humansegv1_lite_192x192_pretrained/humansegv1_lite.pdiparams -qm kl_v2 -b 8 -r all -du -od -s /my_project/quant/humansegv1_lite

  # 执行推理命令,对转换模型(or 量化模型)在整个测试集上进行推理
  python run_quantization.py -r infer -i 1000 -ch TX511 -ct TX511 -if infer_humanseg_model -m /my_project/quant/humansegv1_lite/humansegv1_lite.onnx
  python run_quantization.py -r infer -i 1000 -ch TX511 -ct TX511 -if infer_humanseg_model -m /my_project/quant/humansegv1_lite/humansegv1_lite_quantize.onnx
  

  # 拷贝模型到指定目录
  cp /my_project/quant/humansegv1_lite/humansegv1_lite_quantize.onnx /my_project/quant/to_compiler/humansegv1_lite/
  cp -r /my_project/quant/humansegv1_lite/caffe_model /my_project/quant/to_compiler/humansegv1_lite/
  cp /my_project/quant/humansegv1_lite/dump/float/0001\:Input_x/batch_0.npy /my_project/quant/to_compiler/humansegv1_lite/input.npy
  cp /my_project/quant/humansegv1_lite/dump/quant/0097\:Resize_translated_layer_scale_0.tmp_2/batch_0.npy /my_project/quant/to_compiler/humansegv1_lite/output.npy

  # 说明：以下是量化数据集及全量数据集上的浮点、量化结果；
  ```
<div align=center>
浮点结果<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/4aa790cd-1d62-44cc-b3aa-8b9d8952c95f
 />  
<br>量化结果<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/33850103-8434-43c7-964d-d8a947419b46
 />
<br>浮点结果（全量数据）<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/bffb52cf-60ba-44f6-bbc2-5e8c87f32723
 />
<br>量化结果（全量数据）<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/91febf2e-6712-4699-97c7-2a710121195c
 />
</div>

- __编译__

  ```
  mkdir -p /my_project/quant/to_compiler/humansegv1_lite_reshape/compile_dir

  # 编译网络(使用caffe编译)
  Knight --chip TX5368AV200 rne-compile --net /my_project/quant/humansegv1_lite/caffe_model/humansegv1_lite/humansegv1_lite.prototxt --weight /my_project/quant/humansegv1_lite/caffe_model/humansegv1_lite/humansegv1_lite.weight --outpath /my_project/quant/to_compiler/humansegv1_lite/compile_dir/ --general-process 1

  cp -r /my_project/quant/to_compiler/ /data/examples/baidu_qa_level3
  ```

----

## 6. 推荐网络

- __网络说明__

  - 该部分共有1个网络；该类网络预测点击率，指标为auc
  - 以下逐步介绍该网络的工作流程及执行命令；
  - ***读者也可以在 /TS-Knight/Quantize/Onnx/onnx_quantize_tool 目录下直接执行 bash scripts/scripts_rec/dcn.sh 运行该网络所有命令；***


- __数据预处理__
  - 无需额外预处理；数据来自官网提供slot_train_full_dataset，存放于/TS-Knight/Quantize/Onnx/onnx_quantize_tool/data/data_dcn

- __量化__

  ```
  # 创建模型目录
  mkdir -p /my_project/quant/dcn
  mkdir -p /my_project/quant/to_compiler/dcn

  # 获取飞桨模型
  # /TS-Knight/Quantize/Onnx/onnx_quantize_tool/paddle_models/dcn_static_with_emb

  # 执行量化命令,完成paddle2onnx转换，并对模型进行量化定点
  /usr/bin/python -u run_quantization.py -f paddle -i 1 -ch TX511 -ct TX511 -if infer_dcn_with_emb_model -m paddle_models/dcn_static_with_emb/dcn.pdmodel -w paddle_models/dcn_static_with_emb/dcn.pdiparams -qm percentile -per 0.9999 -b 8 -r all -du -s /my_project/quant/dcn

  # 执行推理命令,对转换模型(or 量化模型)在整个测试集上进行推理
  # 注：此步骤可以跳过，因为 量化数据集与全量数据集相同
  # python run_quantization.py -r infer -i 10 -ch TX511 -ct TX511 -if infer_dcn_with_emb_model -m /my_project/quant/dcn/dcn.onnx
  # python run_quantization.py -r infer -i 10 -ch TX511 -ct TX511 -if infer_dcn_with_emb_model -m /my_project/quant/dcn/dcn_quantize.onnx

  # 拷贝模型到指定目录
  cp /my_project/quant/dcn/dcn_quantize.onnx /my_project/quant/to_compiler/dcn/
  cp -r /my_project/quant/dcn/caffe_model /my_project/quant/to_compiler/dcn/
  cp /my_project/quant/dcn/dump/float/0001\:Input_sparse_inputs/batch_0.npy /my_project/quant/to_compiler/dcn/input1.npy
  cp /my_project/quant/dcn/dump/float/0002\:Input_dense_inputs/batch_0.npy /my_project/quant/to_compiler/dcn/input2.npy
  cp /my_project/quant/dcn/dump/quant/0021\:Sigmoid_sigmoid_0.tmp_0/batch_0.npy /my_project/quant/to_compiler/dcn/output.npy

  # 说明：以下是量化数据集及全量数据集上的浮点、量化结果；此处前后两对数值一样是正常的，因为此处量化数据集和全量数据集是相同的数据

  ```
<div align=center>
浮点结果<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/15df9ba4-f148-45c5-81b5-2f1b280e4f14 />  
<br>量化结果<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/886f6ee9-f25b-4b83-a9b9-d2aabbe8734b />
<br>浮点结果（全量数据）<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/82140ba6-d788-412f-af7b-70efc03bebac />
<br>量化结果（全量数据）<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/c4ba75e5-2950-4faa-99a2-dc6e2ec35901 />
</div>

- __编译__

  ```
  mkdir -p /my_project/quant/to_compiler/dcn/compile_dir

  # 编译网络(使用caffe编译)
  Knight --chip TX5368AV200 rne-compile --net /my_project/quant/dcn/caffe_model/dcn/dcn.prototxt --weight /my_project/quant/dcn/caffe_model/dcn/dcn.weight --outpath /my_project/quant/to_compiler/dcn/compile_dir/ --general-process 1

  cp -r /my_project/quant/to_compiler/ /data/examples/baidu_qa_level3
  ```

----

## 7. NLP网络

- __网络说明__

  - 该部分共有2个网络；该类网络对句子进行情感分类，指标为 accuracy
  - 以下以tinybert为例，逐步介绍该网络的工作流程及执行命令；另一个网络electra的流程和执行命令类似；
  - ***读者也可以在 /TS-Knight/Quantize/Onnx/onnx_quantize_tool 目录下直接执行 bash scripts/scripts_nlp/tinybert.sh运行该网络所有命令；electra网络也有相应脚本；执行bash scripts/scripts_nlp/scripts_nlp_all.sh可一次性运行该部分2个网络（逐个）***

- __数据预处理__
  - 无需额外预处理；数据来自官网提供glue/SST2，存放于/TS-Knight/Quantize/Onnx/onnx_quantize_tool/data/sst2

- __量化__

  ```
  # 创建模型目录
  mkdir -p /my_project/quant/tinybert
  mkdir -p /my_project/quant/to_compiler/tinybert

  # 获取飞桨模型
  # /TS-Knight/Quantize/Onnx/onnx_quantize_tool/paddle_models/bert_tiny

  # 执行量化命令,完成paddle2onnx转换，并对模型进行量化定点
  /usr/bin/python -u run_quantization.py -f paddle -i 10 -ch TX511 -ct TX511 -if infer_bert_model -m paddle_models/bert_tiny/tinybert.pdmodel -w paddle_models/bert_tiny/tinybert.pdiparams -s /my_project/quant/tinybert -qm kl_v2 -b 16 -du -r all -qid None None None int16

  # 执行推理命令,对转换模型(or 量化模型)在整个测试集上进行推理
  python run_quantization.py -r infer -i 1000 -ch TX511 -ct TX511 -if infer_bert_model -b 16 -m /my_project/quant/tinybert/tinybert.onnx
  python run_quantization.py -r infer -i 1000 -ch TX511 -ct TX511 -if infer_bert_model_quant -b 16 -m /my_project/quant/tinybert/tinybert_quantize.onnx

  # 拷贝模型到指定目录
  cp /my_project/quant/tinybert/tinybert_quantize.onnx /my_project/quant/to_compiler/tinybert/
  cp -r /my_project/quant/tinybert/caffe_model /my_project/quant/to_compiler/tinybert/
  cp /my_project/quant/tinybert/dump/float/0001\:Input_input_ids/batch_0.npy /my_project/quant/to_compiler/tinybert/input1.npy
  cp /my_project/quant/tinybert/dump/float/0002\:Input_input_ids/batch_0.npy /my_project/quant/to_compiler/tinybert/input2.npy
  cp /my_project/quant/tinybert/dump/float/0003\:Input_input_ids/batch_0.npy /my_project/quant/to_compiler/tinybert/input3.npy
  cp /my_project/quant/tinybert/dump/float/0004\:Input_input_ids/batch_0.npy /my_project/quant/to_compiler/tinybert/input4.npy
  cp /my_project/quant/tinybert/dump/quant/0219\:Gemm_linear_83.tmp_1/batch_0.npy /my_project/quant/to_compiler/tinybert/output.npy

  # 说明：以下是量化数据集及全量数据集上的浮点、量化结果；
  ```
<div align=center>
浮点结果<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/a33feebe-b6a2-4441-88af-4fa84287e0e4
 />  
<br>量化结果<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/8b33df4e-d998-46a5-8c7a-8d6110d5d336
 />
<br>浮点结果（全量数据）<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/bde8de76-cdba-4584-9f47-68397d628bb8
 />
<br>量化结果（全量数据）<br>
        <img src=https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/a8fc4e73-bcfb-4f26-b46a-9860069dd523
 />
</div>

- __编译__

  ```
  mkdir -p /my_project/quant/to_compiler/tinybert/compile_dir

  # 编译网络(使用caffe编译)
  Knight --chip TX5368AV200 rne-compile --net /my_project/quant/tinybert/caffe_model/tinybert/tinybert.prototxt --weight /my_project/quant/tinybert/caffe_model/tinybert/tinybert.weight --outpath /my_project/quant/to_compiler/tinybert/compile_dir/ --general-process 1

  cp -r /my_project/quant/to_compiler/ /data/examples/baidu_qa_level3
  ```

----

## 8. 硬件测试
- __数据准备__
  - **导出所在服务器 /data/examples/baidu_qa_level3 目录里的to_compiler文件夹给硬件使用**
    - 当前服务器上起一个终端，验证能否进入python脚本运行环境（简称ubuntu环境）：`ssh tsingmicro@10.11.1.127` (password: 123456)
    - 另起一个终端，执行命令：`scp -r /data/examples/baidu_qa_level3/to_compiler tsingmicro@10.11.1.127:/home/tsingmicro/paddle_level3`

  - 在ubuntu环境下 
    - **切换到工作目录**：`cd /home/tsingmicro/paddle_level3`
    - **启动conda环境（重要）**：`conda activate py36`
  
- __数据预处理__
  - 执行预处理脚本：`bash scripts/preprocess.sh all` 
    - `all`代表预处理所有模型，替换成具体模型名字可只预处理相应模型，如 `bash scripts/preprocess.sh vgg13`
  - 当前目录下的data文件夹会生成 all_bins 和 all_pd 两个文件夹，分别存放 作为硬件输入的bin文件 以及 用于后处理（含有label用于指标计算）的tensor文件；

- __硬件上运行模型__
  - 编译模型：`bash scripts/build_model.sh`，预期结果如下：<br>
    - 此时编译所有模型；如果要编译单个模型，如vgg13，则执行`bash scripts/build_model.sh vgg13` <br>
    ![hardware-build-1](https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/40a663d4-51e0-4f20-bae4-3c2ed7ec53e9)


  - 让硬件运行模型：
    - 另起一个终端，进入硬件运行环境（简称root环境）：ssh root@10.11.1.253 (password: 123456)；
    - **切换到工作目录：cd /root/paddle_level3**，注意硬件环境已通过 /root/paddle_level3 挂载 ubuntu环境的 /home/tsingmicro/paddle_level3
    - 执行硬件上运行模型的脚本：`sh scripts/infer_model.sh`，预期结果如下：<br>
      - 此时运行所有模型；如果需要运行单个模型，则执行 `bash scripts/infer_model.sh vgg13`<br>
      ![hware-infer](https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/44884469-bb38-4009-ac4e-956f913538c4)

- __后处理__ 
  - 返回ubuntu环境，**切换到工作目录：cd /home/tsingmicro/paddle_level3**
  - 执行所有网络的后处理脚本：`bash scripts/postprocess.sh all`，得到所有网络的指标
    - 也可设置模型名字 单独 执行后处理脚本，如 `bash scripts/postprocess.sh vgg13`
   
  - 分类网络预期结果如下：<br>
  ![post-dla60x](https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/eddb339b-6cb8-433e-a621-cb24487b1b53)

  - 检测网络预期结果如下：<br>
  ![post-yolo3](https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/ce44789e-9c56-4f3c-b2b5-5a5667937b78)

  - 分割网络预期结果如下（humansegv1_lite）：<br>
  ![post_humansegv1_lite](https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/7c74d4b4-5910-4984-aea8-76bba1bd4adf)

  - nlp网络预期结果如下（tinybert）：<br>
  ![post_tinybert](https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/2a608c7b-1173-475e-95e8-2f95bd31de25)

  - 推荐网络预期结果如下：<br>
  ![dcn-quant-board](https://github.com/tsingmicro-toolchain/ts.knight-modelzoo/assets/7539692/9d89f7c5-516d-48b9-b01a-aa2e4a35ac89)


    

----

## 9. 其他说明
- 上文提到的所有官方模型均可在 [官方模型](https://github.com/PaddlePaddle/models/blob/develop/docs/official/README.md) 找到


