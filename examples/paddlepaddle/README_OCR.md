## 环境准备

- 通过百度网盘下载example_ocrv3的模型权重和数据集 [下载地址](https://pan.baidu.com/s/1T1t-2410GT5oj8F0IJ417w?pwd=398k)
- knight docker镜像请联系[技术支持](../../README.md#技术讨论)获取
- 启动docker容器:

```
docker load -i TS.Knight-1.1.0.7-for-paddle-ocrv3.tar.gz
docker run -v $localhost_dir/example_ocrv3:/TS-Knight/Quantize/Onnx/example -w /TS-Knight/Quantize/Onnx/ -it knight-1.1.0.7-for-paddle-ocrv3:6.0 /bin/bash
```

&emsp;&emsp;其中docker镜像的tar包和example_ocrv3目录为下载的压缩包解压出来的，localhost_dir 为解压出来的本地目录，存放模型权重和量化数据集，目录结构如下：
```
example_ocrv3/
├── models
└── data
```
- **启动容器后，使用上述命令会自动切换目录到/TS-Knight/Quantize/Onnx/（没有的话请执行 cd /TS-Knight/Quantize/Onnx/）**


### OCR_DET 检测网络

- __网络说明__
```
该网络对目标进行检测，指标为 precision、recall和hmean
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
cp /my_project/quant/ocrv3_det/dump/float/0001:x/batch_0.npy /my_project/quant/to_compiler/ocrv3_det/input.npy
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

# 编译网络
- 使用caffe编译：
Knight --chip TX5368A rne-compile --net /my_project/quant/ocrv3_det/caffe_model/ocrv3_det/ocrv3_det.prototxt --weight /my_project/quant/ocrv3_det/caffe_model/ocrv3_det/ocrv3_det.weight --outpath /my_project/quant/to_compiler/ocrv3_det/compile_dir/ --general-process 1
- 使用onnx编译： 
# Knight --chip TX5368A rne-compile --onnx /my_project/quant/to_compiler/ocrv3_det/ocrv3_det_quantize.onnx --outpath /my_project/quant/to_compiler/ocrv3_det/compile_dir/ --general-process 1
```

- __模拟__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_det/simulator_dir

# 输入数据转换成模拟器运行需要的二级制文件
python3 /TS-Knight-software/tools/npy2bin.py --input /my_project/quant/to_compiler/ocrv3_det/input.npy --output /my_project/quant/to_compiler/ocrv3_det/simulator_dir/input.bin

#模拟网络
Knight --chip TX5368A rne-sim --input /my_project/quant/to_compiler/ocrv3_det/simulator_dir/input.bin --weight /my_project/quant/to_compiler/ocrv3_det/compile_dir/ocrv3_det_r.weight --config /my_project/quant/to_compiler/ocrv3_det/compile_dir/ocrv3_det_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_det/simulator_dir/ -fmt nchw

#性能分析器
mkdir -p /my_project/quant/to_compiler/ocrv3_det/profiling_dir

#分析网络
Knight --chip TX5368A rne-profiling --weight /my_project/quant/to_compiler/ocrv3_det/compile_dir/ocrv3_det_r.weight --config /my_project/quant/to_compiler/ocrv3_det/compile_dir/ocrv3_det_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_det/profiling_dir/

#数据比对
python3 sim2quant_compare.py -q /my_project/quant/ocrv3_det/dump/quant/0150\:sigmoid_0.tmp_0/batch_0.npy -s /my_project/quant/to_compiler/ocrv3_det/simulator_dir/result_0_p.txt
```
<div align=center>
对比结果<br>
        <img src=https://user-images.githubusercontent.com/7539692/223620619-ea50918f-345d-45cf-83db-f3e5288681bd.png />
</div>


### OCR_CLS 分类网络

- __网络说明__
```
该网络对目标进行2分类，指标为accuracy
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
cp /my_project/quant/ocrv3_cls/dump/float/0001:x/batch_0.npy /my_project/quant/to_compiler/ocrv3_cls/input.npy
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

# 编译网络
- 使用caffe编译：
Knight --chip TX5368A rne-compile --net /my_project/quant/ocrv3_cls/caffe_model/ocrv3_cls/ocrv3_cls.prototxt --weight /my_project/quant/ocrv3_cls/caffe_model/ocrv3_cls/ocrv3_cls.weight --outpath /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ --general-process 1
- 使用onnx编译： 
# Knight --chip TX5368A rne-compile --onnx /my_project/quant/to_compiler/ocrv3_cls/ocrv3_cls_quantize.onnx --outpath /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ --general-process 1
```

- __模拟__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_cls/simulator_dir

# 输入数据转换成模拟器运行需要的二级制文件
python3 /TS-Knight-software/tools/npy2bin.py --input /my_project/quant/to_compiler/ocrv3_cls/input.npy --output /my_project/quant/to_compiler/ocrv3_cls/simulator_dir/input.bin

#模拟网络
Knight --chip TX5368A rne-sim --input /my_project/quant/to_compiler/ocrv3_cls/simulator_dir/input.bin --weight /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ocrv3_cls_r.weight --config /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ocrv3_cls_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_cls/simulator_dir/ -fmt nchw

#性能分析器
mkdir -p /my_project/quant/to_compiler/ocrv3_cls/profiling_dir

#分析网络
Knight --chip TX5368A rne-profiling --weight /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ocrv3_cls_r.weight --config /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ocrv3_cls_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_cls/profiling_dir/

#数据比对
python3 sim2quant_compare.py -q /my_project/quant/ocrv3_cls/dump/quant/0112\:softmax_0.tmp_0/batch_0.npy -s /my_project/quant/to_compiler/ocrv3_cls/simulator_dir/result_0_p.txt

```
<div align=center>
对比结果<br>
        <img src=https://user-images.githubusercontent.com/7539692/223613165-2bc69c6c-3c0a-498e-8f70-7b5918df62ed.png />
</div>

### OCR_REC 识别网络

- __网络说明__
```
该网络对目标进行识别，指标为 accuracy 和 edit distiance
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
cp /my_project/quant/ocrv3_rec/dump/float/0001:x/batch_0.npy /my_project/quant/to_compiler/ocrv3_rec/input.npy
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

# 编译网络
- 使用caffe编译：
Knight --chip TX5368A rne-compile --net /my_project/quant/ocrv3_rec/caffe_model/ocrv3_rec/ocrv3_rec.prototxt --weight /my_project/quant/ocrv3_rec/caffe_model/ocrv3_rec/ocrv3_rec.weight --outpath /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ --general-process 1
- 使用onnx编译：
# Knight --chip TX5368A rne-compile --onnx /my_project/quant/to_compiler/ocrv3_rec/ocrv3_rec_quantize.onnx --outpath /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ --general-process 1
```

- __模拟__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_rec/simulator_dir

# 输入数据转换成模拟器运行需要的二级制文件
python3 /TS-Knight-software/tools/npy2bin.py --input /my_project/quant/to_compiler/ocrv3_rec/input.npy --output /my_project/quant/to_compiler/ocrv3_rec/simulator_dir/input.bin

#模拟网络
Knight --chip TX5368A rne-sim --input /my_project/quant/to_compiler/ocrv3_rec/simulator_dir/input.bin --weight /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ocrv3_rec_r.weight --config /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ocrv3_rec_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_rec/simulator_dir/ -fmt nchw

#性能分析器
mkdir -p /my_project/quant/to_compiler/ocrv3_rec/profiling_dir

#分析网络
Knight --chip TX5368A rne-profiling --weight /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ocrv3_rec_r.weight --config /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ocrv3_rec_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_rec/profiling_dir/

#数据比对
python3 sim2quant_compare.py -q /my_project/quant/ocrv3_rec/dump/quant/0136\:softmax_2.tmp_0/batch_0.npy -s /my_project/quant/to_compiler/ocrv3_rec/simulator_dir/result_0_p.txt
```
<div align=center>
对比结果<br>
        <img src=https://user-images.githubusercontent.com/7539692/223621360-8b855a68-87b3-4eba-8179-de2b9fa2f5cd.png />
</div>

## Benchmark(数据集来自官网提供和整理)

|  模型名称    | paddle浮点精度 | onnx浮点精度 | 量化精度  | 测试数据量 | 推理速度(单张图片) |
|-------------|-----------|----------|----------|-----------|--------------------|
| OCR_DET     | 0.411      |0.411      | 0.415     | 500     |    90.2911ms        |
| OCR_CLS     | 0.699      |  0.699      |0.699     | 30      |    2.3425ms        |
| OCR_REC     | 0.655      | 0.655      | 0.649     | 2077    |    19.4567ms       |

## 其他说明
- 上文提到的所有官方模型均可在 [官方OCRv3模型](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md) 找到
- 附OCR_DET_512x896 paddle 浮点精度(与表格第一行onnx浮点精度一致)![image](https://user-images.githubusercontent.com/7539692/221813030-d3f08b1f-3533-4062-a88f-6662140fd4d4.png)

