# PaddlePaddle Examples

清微Knight工具链将不断更新、丰富模型库。。。

- 物体分类
  - resnet18
  - mobilenet_v2
  - densenet
  - Others......

## 芯片

&emsp;&emsp;所有示例均在清微智能 TX5368A型号 配套工具链上验证。

## 运行示例

&emsp;&emsp;样例模型均来源自国内产业级深度学习开源框架飞桨的图像分类开发套件[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)，
为了覆盖主流的模型结构和算子类型，我们选择了3个模型放到example里进行了展示。
飞桨图像识别套件[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)， 是飞桨为工业界和学术界所准备的一个图像识别和图像分类任务的工具集，助力使用者训练出更好的视觉模型和应用落地。

## 环境准备

- 通过百度网盘example模型权重和量化数据集 [下载地址](https://pan.baidu.com/s/1T1t-2410GT5oj8F0IJ417w?pwd=398k)
- knight docker镜像请联系[技术支持](../../README.md#技术讨论)获取
- 启动docker容器:

```
docker load -i TS.Knight-publish-1.0.0.4_build1.tar.gz
docker run -v $localhost_dir/examples:/examples -it ubuntu-18.04-ts-release:knight-publish-1.0.0.4 /bin/bash
```

&emsp;&emsp;容器启动成功后，在容器内任意目录下均可使用Knight命令。其中docker镜像的tar包和examples目录为下载的压缩包解压出来的，localhost_dir 为解压出来的本地目录，存放模型权重和量化数据集，目录结构如下：
```
examples/
├── images
└── paddle_weights
```
&emsp;&emsp;PaddlePaddle模型源码位于docker镜像的目录是：/Knight/TS-Quantize/Onnx/frontends/paddle/model/

### resnet18

- __量化__

```
# 创建模型目录
mkdir -p /my_project/quant/resnet18
mkdir -p /my_project/quant/to_compiler/resnet18

# 执行量化命令,完成paddle2onnx转换，并对模型进行量化定点
Knight --chip TX5368A quant onnx -mf paddle -m resnet18 -if infer_cls_model -s /my_project/quant/resnet18/ -w /examples/paddle_weights/resnet18.pdparams -d -bs 1 -qm min_max

# 拷贝模型到指定目录
cp /my_project/quant/resnet18/resnet18_quantize.onnx /my_project/quant/to_compiler/resnet18/
cp /my_project/quant/resnet18/dump/float/0001:inputs/batch_0.npy /my_project/quant/to_compiler/resnet18/input.npy
cp /my_project/quant/resnet18/dump/quant/0033:linear_1.tmp_1/batch_0.npy /my_project/quant/to_compiler/resnet18/output.npy
```
&emsp;&emsp;量化成功后的模型以及一组输入输出数据存放在目录/my_project/quant/to_compiler/resnet18下，供编译环节使用，需要说明的是，这组输入输出数据是用来对比量化的输出与板端模拟的输出完全一致，从而保证量化阶段的模型精度与板端运行的模型精度完全一致！


- __编译__

```
mkdir -p /my_project/quant/to_compiler/resnet18/compile_dir

# 编译网络
Knight --chip TX5368A rne-compile --onnx /my_project/quant/to_compiler/resnet18/resnet18_quantize.onnx --outpath /my_project/quant/to_compiler/resnet18/compile_dir
```

- __模拟__

```
#模拟器
mkdir -p /my_project/quant/to_compiler/resnet18/simulator_dir

# 输入数据转换成模拟器运行需要的二级制文件
python3 /TS-Knight-software/tools/npy2bin.py --input /my_project/quant/to_compiler/resnet18/input.npy --output /my_project/quant/to_compiler/resnet18/simulator_dir/input.bin

#模拟网络
Knight --chip TX5368A rne-sim --input /my_project/quant/to_compiler/resnet18/simulator_dir/input.bin --weight /my_project/quant/to_compiler/resnet18/compile_dir/resnet18_quantize_r.weight --config /my_project/quant/to_compiler/resnet18/compile_dir/resnet18_quantize_r.cfg --outpath /my_project/quant/to_compiler/resnet18/simulator_dir/

#性能分析器
mkdir -p /my_project/quant/to_compiler/resnet18/profiling_dir

#分析网络
Knight --chip TX5368A rne-profiling --weight /my_project/quant/to_compiler/resnet18/compile_dir/resnet18_quantize_r.weight --config /my_project/quant/to_compiler/resnet18/compile_dir/resnet18_quantize_r.cfg --outpath /my_project/quant/to_compiler/resnet18/profiling_dir/

```

### mobilenet_v2

- __量化__

```
# 创建模型目录
mkdir -p /my_project/quant/mobilenet_v2
mkdir -p /my_project/quant/to_compiler/mobilenet_v2

# 执行量化命令,完成paddle2onnx转换，并对模型进行量化定点
Knight --chip TX5368A quant onnx -mf paddle -m mobilenet_v2 -if infer_cls_model -s /my_project/quant/mobilenet_v2/ -w /examples/paddle_weights/mobilenet_v2_x1.0.pdparams -d -bs 1 -qm min_max

# 拷贝模型到指定目录
cp /my_project/quant/mobilenet_v2/mobilenet_v2_quantize.onnx /my_project/quant/to_compiler/mobilenet_v2/
cp /my_project/quant/mobilenet_v2/dump/float/0001:inputs/batch_0.npy /my_project/quant/to_compiler/mobilenet_v2/input.npy
cp /my_project/quant/mobilenet_v2/dump/quant/0067:linear_1.tmp_1/batch_0.npy /my_project/quant/to_compiler/mobilenet_v2/output.npy
```

- __编译__

```
mkdir -p /my_project/quant/to_compiler/mobilenet_v2/compile_dir

# 编译网络
Knight --chip TX5368A rne-compile --onnx /my_project/quant/to_compiler/mobilenet_v2/mobilenet_v2_quantize.onnx --outpath /my_project/quant/to_compiler/mobilenet_v2/compile_dir
```

- __模拟__

```
mkdir -p /my_project/quant/to_compiler/mobilenet_v2/simulator_dir

# 输入数据转换成模拟器运行需要的二级制文件
python3 /TS-Knight-software/tools/npy2bin.py --input /my_project/quant/to_compiler/mobilenet_v2/input.npy --output /my_project/quant/to_compiler/mobilenet_v2/simulator_dir/input.bin

#模拟网络
Knight --chip TX5368A rne-sim --input /my_project/quant/to_compiler/mobilenet_v2/simulator_dir/input.bin --weight /my_project/quant/to_compiler/mobilenet_v2/compile_dir/mobilenet_v2_quantize_r.weight --config /my_project/quant/to_compiler/mobilenet_v2/compile_dir/mobilenet_v2_quantize_r.cfg --outpath /my_project/quant/to_compiler/mobilenet_v2/simulator_dir/

#性能分析器
mkdir -p /my_project/quant/to_compiler/mobilenet_v2/profiling_dir

#分析网络
Knight --chip TX5368A rne-profiling --weight /my_project/quant/to_compiler/mobilenet_v2/compile_dir/mobilenet_v2_quantize_r.weight --config /my_project/quant/to_compiler/mobilenet_v2/compile_dir/mobilenet_v2_quantize_r.cfg --outpath /my_project/quant/to_compiler/mobilenet_v2/profiling_dir/
```

### densenet

- __量化__

```
# 创建模型目录
mkdir -p /my_project/quant/densenet
mkdir -p /my_project/quant/to_compiler/densenet

# 执行量化命令,完成paddle2onnx转换，并对模型进行量化定点
Knight --chip TX5368A quant onnx -mf paddle -m densenet -if infer_cls_model -s /my_project/quant/densenet/ -w /examples/paddle_weights/DenseNet121_pretrained.pdparams -d -bs 1 -qm min_max

# 拷贝模型到指定目录
cp /my_project/quant/densenet/densenet_quantize.onnx /my_project/quant/to_compiler/densenet/
cp /my_project/quant/densenet/dump/float/0001:inputs/batch_0.npy /my_project/quant/to_compiler/densenet/input.npy
cp /my_project/quant/densenet/dump/quant/0248:linear_1.tmp_1/batch_0.npy /my_project/quant/to_compiler/densenet/output.npy
```

- __编译__

```
mkdir -p /my_project/quant/to_compiler/densenet/compile_dir

# 编译网络
Knight --chip TX5368A rne-compile --onnx /my_project/quant/to_compiler/densenet/densenet_quantize.onnx --outpath /my_project/quant/to_compiler/densenet/compile_dir

```

- __模拟__

```
mkdir -p /my_project/quant/to_compiler/densenet/simulator_dir

# 输入数据转换成模拟器运行需要的二级制文件
python3 /TS-Knight-software/tools/npy2bin.py --input /my_project/quant/to_compiler/densenet/input.npy --output /my_project/quant/to_compiler/densenet/simulator_dir/input.bin

#模拟网络
Knight --chip TX5368A rne-sim --input /my_project/quant/to_compiler/densenet/simulator_dir/input.bin --weight /my_project/quant/to_compiler/densenet/compile_dir/densenet_quantize_r.weight --config /my_project/quant/to_compiler/densenet/compile_dir/densenet_quantize_r.cfg --outpath /my_project/quant/to_compiler/densenet/simulator_dir/

#性能分析器
mkdir -p /my_project/quant/to_compiler/densenet/profiling_dir

#分析网络
Knight --chip TX5368A rne-profiling --weight /my_project/quant/to_compiler/densenet/compile_dir/densenet_quantize_r.weight --config /my_project/quant/to_compiler/densenet/compile_dir/densenet_quantize_r.cfg --outpath /my_project/quant/to_compiler/densenet/profiling_dir/
```


## Benchmark(ImageNet)

|  模型名称     | 浮点精度  | 量化精度  | 数据量   | 推理速度(单张图片)   |
|--------------|-----------|----------|----------|--------------------|
| resnet_18    | 69.4      | 69.2     | 1000     |    4.1668ms        |
| mobilenet_v2 | 71.8      | 72.1     | 1000     |    2.7382ms        |
| densenet     | 76.6      | 76.6     | 1000     |    11.6806ms       |
