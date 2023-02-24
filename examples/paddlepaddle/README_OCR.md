## 环境准备

- 通过百度网盘下载example_ocrv3的模型权重和数据集 [下载地址](https://pan.baidu.com/s/1T1t-2410GT5oj8F0IJ417w?pwd=398k)
- knight docker镜像请联系[技术支持](../../README.md#技术讨论)获取
- 启动docker容器:

```
docker load -i TS.Knight-1.1.0.7-for-paddle-ocrv3.tar.gz
docker run -v $localhost_dir/example_ocrv3:/TS-Knight/Quantize/Onnx/example -it knight-1.1.0.7-for-paddle-ocrv3 /bin/bash
```

&emsp;&emsp;其中docker镜像的tar包和example_ocrv3目录为下载的压缩包解压出来的，localhost_dir 为解压出来的本地目录，存放模型权重和量化数据集，目录结构如下：
```
example_ocrv3/
├── models
└── data
```
- 启动容器后，请切换目录到/TS-Knight/Quantize/Onnx/（命令为：cd /TS-Knight/Quantize/Onnx/）


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
python run_quantization.py -f paddle -r quant -ch TX511 -od -if infer_ocr_det_model -m example/models/ch_PP-OCRv3_det_infer_512x896/ocrv3_det.pdmodel -w example/models/ch_PP-OCRv3_det_infer_512x896/ocrv3_det.pdiparams -s /my_project/quant/ocrv3_det/ -bs 1 -i 50 -qm kl -is 1 3 512 896  --dump

# 执行推理命令,对转换模型和量化模型在整个测试集上进行推理（可并行执行）
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_det/ocrv3_det.onnx -if infer_ocr_det_model -bs 1 -i 500
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_det/ocrv3_det_quantize.onnx -if infer_ocr_det_model -bs 1 -i 500

# 拷贝模型到指定目录
cp /my_project/quant/ocrv3_det/ocrv3_det_quantize.onnx /my_project/quant/to_compiler/ocrv3_det/
cp /my_project/quant/ocrv3_det/dump/float/0001:input/batch_0.npy /my_project/quant/to_compiler/ocrv3_det/input.npy
```


- __编译__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_det/compile_dir

# 编译网络
Knight --chip TX5368A rne-compile --onnx /my_project/quant/to_compiler/ocrv3_det/ocrv3_det_quantize.onnx --outpath /my_project/quant/to_compiler/ocrv3_det/compile_dir/ --general-process 1
```

- __模拟__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_det/simulator_dir

# 输入数据转换成模拟器运行需要的二级制文件
python3 /TS-Knight-software/tools/npy2bin.py --input /my_project/quant/to_compiler/ocrv3_det/input.npy --output /my_project/quant/to_compiler/ocrv3_det/simulator_dir/input.bin

#模拟网络
Knight --chip TX5368A rne-sim --input /my_project/quant/to_compiler/ocrv3_det/simulator_dir/input.bin --weight /my_project/quant/to_compiler/ocrv3_det/compile_dir/ocrv3_det_quantize_r.weight --config /my_project/quant/to_compiler/ocrv3_det/compile_dir/ocrv3_det_quantize_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_det/simulator_dir/

#性能分析器
mkdir -p /my_project/quant/to_compiler/ocrv3_det/profiling_dir

#分析网络
Knight --chip TX5368A rne-profiling --weight /my_project/quant/to_compiler/ocrv3_det/compile_dir/ocrv3_det_quantize_r.weight --config /my_project/quant/to_compiler/ocrv3_det/compile_dir/ocrv3_det_quantize_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_det/profiling_dir/
```

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
python run_quantization.py -f paddle -r quant -ch TX511 -if infer_ocr_cls_model -m example/models/ch_ppocr_mobile_v2.0_cls_infer/ocrv3_cls.pdmodel -w example/models/ch_ppocr_mobile_v2.0_cls_infer/ocrv3_cls.pdiparams -s /my_project/quant/ocrv3_cls/ -bs 1 -i 10 -is -1 3 48 192 --dump

# 执行推理命令,对转换模型和量化模型在整个测试集上进行推理（可并行执行）
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_cls/ocrv3_cls.onnx -if infer_ocr_cls_model -bs 1 -i 30
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_cls/ocrv3_cls_quantize.onnx -if infer_ocr_cls_model -bs 1 -i 30

# 拷贝模型到指定目录
cp /my_project/quant/ocrv3_cls/ocrv3_cls_quantize.onnx /my_project/quant/to_compiler/ocrv3_cls/
cp /my_project/quant/ocrv3_cls/dump/float/0001:input/batch_0.npy /my_project/quant/to_compiler/ocrv3_cls/input.npy
```


- __编译__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_cls/compile_dir

# 编译网络
Knight --chip TX5368A rne-compile --onnx /my_project/quant/to_compiler/ocrv3_cls/ocrv3_cls_quantize.onnx --outpath /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ --general-process 1
```

- __模拟__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_cls/simulator_dir

# 输入数据转换成模拟器运行需要的二级制文件
python3 /TS-Knight-software/tools/npy2bin.py --input /my_project/quant/to_compiler/ocrv3_cls/input.npy --output /my_project/quant/to_compiler/ocrv3_cls/simulator_dir/input.bin

#模拟网络
Knight --chip TX5368A rne-sim --input /my_project/quant/to_compiler/ocrv3_cls/simulator_dir/input.bin --weight /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ocrv3_cls_quantize_r.weight --config /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ocrv3_cls_quantize_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_cls/simulator_dir/

#性能分析器
mkdir -p /my_project/quant/to_compiler/ocrv3_cls/profiling_dir

#分析网络
Knight --chip TX5368A rne-profiling --weight /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ocrv3_cls_quantize_r.weight --config /my_project/quant/to_compiler/ocrv3_cls/compile_dir/ocrv3_cls_quantize_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_cls/profiling_dir/
```

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
python run_quantization.py -f paddle -r quant -ch TX511 -if infer_ocr_rec_model -m example/models/en_PP-OCRv3_rec_infer_retrain_relu_4dims/ocrv3_rec.pdmodel -w example/models/en_PP-OCRv3_rec_infer_retrain_relu_4dims/ocrv3_rec.pdiparams -s /my_project/quant/ocrv3_rec/ -bs 1 -i 300 -b 16 -qm min_max -is 1 3 48 320 --dump

# 执行推理命令,对转换模型和量化模型在整个测试集上进行推理（可并行执行）
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_rec/ocrv3_rec.onnx -if infer_ocr_rec_model -bs 1 -i 2100
python run_quantization.py -r infer -ch TX511 -m /my_project/quant/ocrv3_rec/ocrv3_rec_quantize.onnx -if infer_ocr_rec_model -bs 1 -i 2100 -b 16

# 拷贝模型到指定目录
cp /my_project/quant/ocrv3_rec/ocrv3_rec_quantize.onnx /my_project/quant/to_compiler/ocrv3_rec/
cp /my_project/quant/ocrv3_rec/dump/float/0001:input/batch_0.npy /my_project/quant/to_compiler/ocrv3_rec/input.npy
```


- __编译__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_rec/compile_dir

# 编译网络
Knight --chip TX5368A rne-compile --onnx /my_project/quant/to_compiler/ocrv3_rec/ocrv3_rec_quantize.onnx --outpath /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ --general-process 1
```

- __模拟__

```
mkdir -p /my_project/quant/to_compiler/ocrv3_rec/simulator_dir

# 输入数据转换成模拟器运行需要的二级制文件
python3 /TS-Knight-software/tools/npy2bin.py --input /my_project/quant/to_compiler/ocrv3_rec/input.npy --output /my_project/quant/to_compiler/ocrv3_rec/simulator_dir/input.bin

#模拟网络
Knight --chip TX5368A rne-sim --input /my_project/quant/to_compiler/ocrv3_rec/simulator_dir/input.bin --weight /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ocrv3_rec_quantize_r.weight --config /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ocrv3_rec_quantize_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_rec/simulator_dir/

#性能分析器
mkdir -p /my_project/quant/to_compiler/ocrv3_rec/profiling_dir

#分析网络
Knight --chip TX5368A rne-profiling --weight /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ocrv3_rec_quantize_r.weight --config /my_project/quant/to_compiler/ocrv3_rec/compile_dir/ocrv3_rec_quantize_r.cfg --outpath /my_project/quant/to_compiler/ocrv3_rec/profiling_dir/
```


## Benchmark(数据集来自官网提供和整理)

|  模型名称    | paddle浮点精度 | onnx浮点精度 | 量化精度  | 测试数据量 | 推理速度(单张图片) |
|-------------|-----------|----------|----------|-----------|--------------------|
| OCR_DET     | 0.411      |0.411      | 0.415     | 500     |    3.2911ms        |
| OCR_CLS     | 0.699      |  0.699      |0.699     | 30      |    2.3425ms        |
| OCR_REC     | 0.655      | 0.655      | 0.649     | 2077    |    10.4567ms       |

## 其他说明
上文提到的所有官方模型均可在 [官方OCRv3模型](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md) 找到
