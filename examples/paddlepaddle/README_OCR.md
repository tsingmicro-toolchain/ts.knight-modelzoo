## Benchmark(数据集来自官网提供和整理)

|  模型名称    | paddle浮点精度 | onnx浮点精度 | 量化精度  | 测试数据量 | 推理速度(单张图片) |
|-------------|-----------|----------|----------|-----------|--------------------|
| OCR_DET     | 0.411      |0.411      | 0.415     | 500     |    135ms        |
| OCR_CLS     | 0.699      |  0.699      |0.699     | 30      |    2.542ms        |
| OCR_REC     | 0.655      | 0.655      | 0.650     | 2077    |    19.456ms       |

## 环境准备

- 请联系[技术支持](../../README.md#技术讨论)获取需要用到的VPN、账号、knight docker镜像
- 上述配置步骤请参考提测文档-安装步骤；配置完成后进入服务器（即安装步骤第3步）：ssh wx_paddle@192.168.1.10
- 启动docker容器:

```
docker load -i TS.Knight-1.1.0.7-for-paddle-ocrv3.tar.gz
docker run -w /TS-Knight/Quantize/Onnx/ -v /data/examples/baidu_qa_ocrv3:/data/examples/baidu_qa_ocrv3 --ipc=host -it knight-1.1.0.7-for-paddle-ocrv3:1.0 /bin/bash
```
- **启动容器后，使用上述命令会自动切换目录到/TS-Knight/Quantize/Onnx/（没有的话请执行 cd /TS-Knight/Quantize/Onnx/）**
- **数据和模型：** 由于3个OCR模型及其数据的大小较小（几十M），因此已被打包到镜像（容器）目录 /TS-Knight/Quantize/Onnx/example下的data和models文件夹；
- 工作目录说明：<br>
  ![image](https://user-images.githubusercontent.com/7539692/227886407-84648b1e-1ea6-47fd-b916-aea0b01049bc.png)
- 量化命令（集成转换工具）解析：
  - 举例：如下命令，会将paddle模型转为onnx，并对onnx模型进行量化定点；该过程输出转换浮点精度和量化精度
    - python run_quantization.py -f paddle -r all -ch TX511 -od -if infer_ocr_det_model -m example/models/ch_PP-OCRv3_det_infer_512x896/ocrv3_det.pdmodel -w example/models/ch_PP-OCRv3_det_infer_512x896/ocrv3_det.pdiparams -s /my_project/quant/ocrv3_det/ -bs 1 -i 50 -qm kl -is 1 3 512 896  --dump
  - 主要参数解析：
    - "-f paddle"：指定要转换的框架为paddle；"-r all"：表示执行转换、量化、转IR3个过程；"-od"：表示输出反量化为浮点
    - "-m xxx.pdmodel"：指定paddle模型路径；"-w xxx.pdiparams"：指定paddle模型权重路径；"-s xxx"：指定量化模型保存路径；
    - "-qm kl"：指定量化系数计算方式；"-is"：指定输入形状，不可随意指定；"--dump"：指定保存浮点结果、量化结果等用于对比；


## OCR_DET 检测网络

- __网络说明__

  - 该网络对目标进行检测，指标为 precision、recall和hmean
  - 以下逐步介绍该网络的工作流程及执行命令；
  - ***读者也可以在 /TS-Knight/Quantize/Onnx/ 目录下直接执行 bash scripts/ocrv3_det.sh 运行该网络所有命令***
  - `注：同目录下ocrv3_all.sh可一次性运行3个网络所有命令`

- __数据预处理__
```
python example/ocrv3_process/pre_post_process.py -c example/configs/ch_PP-OCRv3_det_cml.yml -d example/data/images_ocrv3 -s example/data
```

- __量化__

```
# 创建模型目录
mkdir -p /my_project/quant/ocrv3_det
mkdir -p /my_project/quant/to_compiler/ocrv3_det

# 获取飞桨模型
/TS-Knight/Quantize/Onnx/example/models/ch_PP-OCRv3_det_infer_512x896/
（说明：我们下载了paddle官方ch_PP-OCRv3_det模型（链接见文末）；限于编译器对tensor大小的要求，我们将模型input shape设置为[1,3,512,896]；这与官方推荐的shape=[1,3,746,1312]有所差别）

# 执行量化命令,完成paddle2onnx转换，并对模型进行量化定点
python run_quantization.py -f paddle -r all -ch TX511 -od -if infer_ocr_det_model -m example/models/ch_PP-OCRv3_det_infer_512x896/ocrv3_det.pdmodel -w example/models/ch_PP-OCRv3_det_infer_512x896/ocrv3_det.pdiparams -s /my_project/quant/ocrv3_det/ -bs 1 -i 50 -qm kl -is 1 3 512 896  --dump

# 执行推理命令,对转换模型和量化模型在整个测试集上进行推理（可并行执行）
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_det/ocrv3_det.onnx -if infer_ocr_det_model -bs 1 -i 500
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_det/ocrv3_det_quantize.onnx -if infer_ocr_det_model -bs 1 -i 500

# 拷贝模型到指定目录
cp /my_project/quant/ocrv3_det/ocrv3_det_quantize.onnx /my_project/quant/to_compiler/ocrv3_det/
cp -r /my_project/quant/ocrv3_det/caffe_model /my_project/quant/to_compiler/ocrv3_det/
cp /my_project/quant/ocrv3_det/dump/float/0001:x/batch_0.npy /my_project/quant/to_compiler/ocrv3_det/input.npy
cp /my_project/quant/ocrv3_det/dump/quant/0150\:sigmoid_0.tmp_0/batch_0.npy /my_project/quant/to_compiler/ocrv3_det/output.npy
```
<div align=center>
浮点结果<br>
        <img src=https://user-images.githubusercontent.com/7539692/223619517-4b088963-a843-4b5a-aeeb-f98d8a3e3f2f.png />  
<br>量化结果<br>
        <img src=https://user-images.githubusercontent.com/7539692/223619436-390fbd93-877c-4ad9-bd2e-d801ea3ba18a.png />
<br>浮点结果（全量数据）<br>
        <img src=https://user-images.githubusercontent.com/7539692/223619202-85c128a5-a2d1-43a0-a557-7411a992bfc2.png />
<br>量化结果（全量数据）<br>
        <img src=https://user-images.githubusercontent.com/7539692/223618984-1b4ef8d1-bf17-418f-a916-4866b3aa908e.png />
</div>

- __编译__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_det/compile_dir

# 编译网络(使用caffe编译)
Knight --chip TX5368A rne-compile --net /my_project/quant/ocrv3_det/caffe_model/ocrv3_det/ocrv3_det.prototxt --weight /my_project/quant/ocrv3_det/caffe_model/ocrv3_det/ocrv3_det.weight --outpath /my_project/quant/to_compiler/ocrv3_det/compile_dir/ --general-process 1

cp -r /my_project/quant/to_compiler/ /data/examples/baidu_qa_ocrv3
```


## OCR_CLS 分类网络

- __网络说明__

  - 该网络对目标进行2分类，指标为accuracy
  - 以下逐步介绍该网络的工作流程及执行命令；
  - ***读者也可以在 /TS-Knight/Quantize/Onnx/ 目录下直接执行 bash scripts/ocrv3_cls.sh 运行该网络所有命令***
  - `注：同目录下ocrv3_all.sh可运行3个网络所有命令`

- __数据预处理__
```
python example/ocrv3_process/pre_post_process.py -c example/configs/cls_mv3.yml -d example/data/images_ocrv3 -s example/data
```

- __量化__

```
# 创建模型目录
mkdir -p /my_project/quant/ocrv3_cls
mkdir -p /my_project/quant/to_compiler/ocrv3_cls

# 获取飞桨模型
/TS-Knight/Quantize/Onnx/example/models/ch_ppocr_mobile_v2.0_cls_infer/
（说明：我们下载了paddle官方ch_ppocr_mobile_v2.0_cls模型（链接见文末））

# 执行量化命令,完成paddle2onnx转换，并对模型进行量化定点
python run_quantization.py -f paddle -r all -ch TX511 -if infer_ocr_cls_model -m example/models/ch_ppocr_mobile_v2.0_cls_infer/ocrv3_cls.pdmodel -w example/models/ch_ppocr_mobile_v2.0_cls_infer/ocrv3_cls.pdiparams -s /my_project/quant/ocrv3_cls/ -bs 1 -i 10 -is -1 3 48 192 --dump

# 执行推理命令,对转换模型和量化模型在整个测试集上进行推理（可并行执行）
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_cls/ocrv3_cls.onnx -if infer_ocr_cls_model -bs 1 -i 30
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_cls/ocrv3_cls_quantize.onnx -if infer_ocr_cls_model -bs 1 -i 30

# 拷贝模型到指定目录
cp /my_project/quant/ocrv3_cls/ocrv3_cls_quantize.onnx /my_project/quant/to_compiler/ocrv3_cls/
cp -r /my_project/quant/ocrv3_cls/caffe_model /my_project/quant/to_compiler/ocrv3_cls/
cp /my_project/quant/ocrv3_cls/dump/float/0001\:x/batch_0.npy /my_project/quant/to_compiler/ocrv3_cls/input.npy
cp /my_project/quant/ocrv3_cls/dump/quant/0112\:softmax_0.tmp_0/batch_0.npy /my_project/quant/to_compiler/ocrv3_cls/output.npy
```
<div align=center>
浮点结果<br>
        <img src=https://user-images.githubusercontent.com/7539692/223619814-ffa14486-8384-4f5d-bb98-2da0e848716d.png />  
<br>量化结果<br>
        <img src=https://user-images.githubusercontent.com/7539692/223620109-14b330fe-c12c-4c8d-8617-547fd8598886.png />
<br>浮点结果（全量数据）<br>
        <img src=https://user-images.githubusercontent.com/7539692/223612673-60145f5d-c7ca-47a3-95cd-d623f69366aa.png />
<br>量化结果（全量数据）<br>
        <img src=https://user-images.githubusercontent.com/7539692/223620183-f0603421-90a0-4370-9c3b-7eb0ef22b146.png />
</div>

- __编译__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_cls/compile_dir

# 编译网络(使用caffe编译)
Knight --chip TX5368A rne-compile --net /my_project/quant/ocrv3_cls/caffe_model/ocrv3_cls/ocrv3_cls.prototxt --weight /my_project/quant/ocrv3_cls/caffe_model/ocrv3_cls/ocrv3_cls.weight --outpath /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ --general-process 1

cp -r /my_project/quant/to_compiler/ /data/examples/baidu_qa_ocrv3
```

## OCR_REC 识别网络

- __网络说明__

  - 该网络对目标进行识别，指标为 accuracy 和 edit distiance
  - 以下逐步介绍该网络的工作流程及执行命令；
  - ***读者也可以在 /TS-Knight/Quantize/Onnx/ 目录下直接执行 bash scripts/ocrv3_rec.sh 运行该网络所有命令***
  - `注：同目录下ocrv3_all.sh可运行3个网络所有命令`

- __数据预处理__
```
python example/ocrv3_process/pre_post_process.py -c example/configs/en_PP-OCRv3_rec.yml -d example/data/images_ocrv3 -s example/data
```

- __量化__

```
# 创建模型目录
mkdir -p /my_project/quant/ocrv3_rec
mkdir -p /my_project/quant/to_compiler/ocrv3_rec

# 获取飞桨模型
/TS-Knight/Quantize/Onnx/example/models/en_PP-OCRv3_rec_infer_retrain_relu_4dims/
（说明：我们下载了paddle官方en_PP-OCRv3_rec模型（链接见文末）；限于编译器对tensor维度的要求，将原始模型内部5维操作改成4维操作，同时将hardswish替换成relu，并在官网提供数据上重训练，得到精度与原模型一致的新模型，再进行量化）

# 执行量化命令,完成paddle2onnx转换，并对模型进行量化定点
python run_quantization.py -f paddle -r all -ch TX511 -if infer_ocr_rec_model -m example/models/en_PP-OCRv3_rec_infer_retrain_relu_4dims/ocrv3_rec.pdmodel -w example/models/en_PP-OCRv3_rec_infer_retrain_relu_4dims/ocrv3_rec.pdiparams -s /my_project/quant/ocrv3_rec/ -bs 1 -i 300 -b 16 -qm min_max -is 1 3 48 320 --dump

# 执行推理命令,对转换模型和量化模型在整个测试集上进行推理（可并行执行）
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_rec/ocrv3_rec.onnx -if infer_ocr_rec_model -bs 1 -i 2100
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_rec/ocrv3_rec_quantize.onnx -if infer_ocr_rec_model -bs 1 -i 2100 -b 16

# 拷贝模型到指定目录
cp /my_project/quant/ocrv3_rec/ocrv3_rec_quantize.onnx /my_project/quant/to_compiler/ocrv3_rec/
cp -r /my_project/quant/ocrv3_rec/caffe_model /my_project/quant/to_compiler/ocrv3_rec/
cp /my_project/quant/ocrv3_rec/dump/float/0001\:x/batch_0.npy /my_project/quant/to_compiler/ocrv3_rec/input.npy
cp /my_project/quant/ocrv3_rec/dump/quant/0136\:softmax_2.tmp_0/batch_0.npy /my_project/quant/to_compiler/ocrv3_rec/output.npy
```
<div align=center>
浮点结果<br>
        <img src=https://user-images.githubusercontent.com/7539692/223621114-3e729bf5-889d-4249-a4f3-83c998a8fa1d.png />  
<br>量化结果<br>
        <img src=https://user-images.githubusercontent.com/7539692/223621039-1e519d14-9113-43d1-a029-0ae6cb478be5.png />
<br>浮点结果（全量数据）<br>
        <img src=https://user-images.githubusercontent.com/7539692/223620897-d1c1830e-4da2-4601-9a5b-6101bfbcf4bf.png />
<br>量化结果（全量数据）<br>
        <img src=https://user-images.githubusercontent.com/7539692/223620820-4d460222-f564-4812-933f-0e41996c9b76.png />
</div>

- __编译__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_rec/compile_dir

# 编译网络(使用caffe编译)
Knight --chip TX5368A rne-compile --net /my_project/quant/ocrv3_rec/caffe_model/ocrv3_rec/ocrv3_rec.prototxt --weight /my_project/quant/ocrv3_rec/caffe_model/ocrv3_rec/ocrv3_rec.weight --outpath /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ --general-process 1

cp -r /my_project/quant/to_compiler/ /data/examples/baidu_qa_ocrv3
```


## 硬件测试
- __环境准备__
  - **导出 wx_paddle@192.168.1.10:/data/examples/baidu_qa_ocrv3 目录里的 to_compiler文件夹**
  - 进入python脚本运行环境（简称ubuntu环境）：ssh ubuntu@10.11.1.190 (password: 123456)，**切换到工作目录：cd /home/ubuntu/example_ocrv3**
  - 将第一步导出的to_compiler文件夹整个 放到 ubuntu@10.11.1.190:/home/ubuntu/example_ocrv3 目录下，得到/home/ubuntu/example_ocrv3/to_compiler目录

- __数据预处理__
  - 执行预处理脚本：bash scripts/preprocess.sh
  - 当前目录下的data文件夹会生成 all_bins 和 all_pd 两个文件夹，分别存放 作为硬件输入的bin文件 以及 用于后处理（含有图像数据和label）的tensor文件；

- __硬件上运行模型__
  - 编译模型：bash scripts/process_build_demo.sh，预期结果如下：
    ![企业微信截图_16798868195832](https://user-images.githubusercontent.com/7539692/227873268-b592ddca-16cc-4f94-967c-ee3aef2fa9c6.png)
  - 让硬件运行模型：
    - 另起一个终端，进入硬件运行环境（简称root环境）：ssh root@10.11.1.250 (password: root)；
    - **切换到工作目录：cd /root/example_ocrv3**，注意硬件环境已通过 /root/example_ocrv3 挂载 ubuntu环境的 /home/ubuntu/example_ocrv3
    - 执行硬件上运行模型的脚本：./scripts/process_run_demo.sh，预期结果如下：
      ![image](https://user-images.githubusercontent.com/7539692/227880319-c9633bdb-4edc-4bb4-a338-e46b5301c2c5.png)


- __后处理__ 
  - 返回ubuntu环境，**切换到工作目录：cd /home/ubuntu/example_ocrv3**
  - 执行所有网络的后处理脚本：bash scripts/postprocess.sh，得到3个网络的精度
  - 也可3个网络分开执行后处理脚本：
    - bash scripts/postprocess.sh ocr_det 
    - bash scripts/postprocess.sh ocr_cls
    - bash scripts/postprocess.sh ocr_rec
  - 预期结果如下(det/cls/rec)：
    ![image](https://user-images.githubusercontent.com/7539692/227878069-38f55d38-3a81-40c0-8b6e-a8a684c28980.png)
    ![image](https://user-images.githubusercontent.com/7539692/227878154-3ef221f8-84e6-40fe-bd49-3cde86eaa5ab.png)
    ![image](https://user-images.githubusercontent.com/7539692/227878263-8bdedaeb-03dd-467b-adbb-01471aafb877.png)


## 其他说明
- 上文提到的所有官方模型均可在 [官方OCRv3模型](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md) 找到
- 附OCR_DET_512x896 paddle 浮点精度(与表格第一行onnx浮点精度一致)![image](https://user-images.githubusercontent.com/7539692/221813030-d3f08b1f-3533-4062-a88f-6662140fd4d4.png)

