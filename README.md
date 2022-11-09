# 清微AI工具链TS.Knight-简介

## 概述

&emsp;&emsp;TS.Knight是清微智能提供的一站式开发平台，包含部署AI模型所需的全套工具链，支持模型量化、精度比对、模型编译、模拟和性能分析等功能。

## 整体框架

&emsp;&emsp;TS.Knight整体框架如下图所示。

![](https://user-images.githubusercontent.com/46103969/200753726-7c57db84-e344-4548-9e70-4106b4a4d928.png)

-   Knight量化工具(Knight-Quantize): 基于少量数据(比如图片、语音、文本等类型) 量化浮点模型。
-   Knight RNE编译器(Knight-RNE-Compiler): 编译量化模型，产生RNE执行的指令配置文件。
-   Knight RNE模拟器(Knight-RNE-Simulator) : 用于仿真神经网络在RNE上推理计算过程，输出计算层的结果。
-   Knight RNE性能分析器(Knight-RNE-Profiling): 用于分析神经网络在芯片RNE上执行时间和存储开销，并给出分析报告。
-   Knight Finetune库(Knight-Finetune-Lib) : 在使用量化工具后，精度损失较大的情况下，可使用Finetune库进行量化感知训练，得到更适合量化的浮点模型。
-   Knight RNE模拟库(Knight-RNE-Simulator-Lib) : 供用户在PC端调用编写自己的应用程序，从而实现模拟运行结果。
-   Knight RNE 运行时库(Knight-RNE-Runtime-Lib) : 供用户在PC端交叉编译时调用，从而实现板端运行。

## 开发流程

![](https://user-images.githubusercontent.com/46103969/200753736-fc476406-0d2f-49c4-bf76-32f3ec08bd58.png)

1.  用户使用Knight量化工具将提前训练好的浮点模型量化成IR定点模型。
2.  用户使用Knight RNE编译器将IR定点模型编译成芯片部署资源(cfg和weight资源)。
3.  用户使用Knight RNE模拟器对测试数据进行推理，也可以使用Knight RNE性能分析工具对模型进行性能分析。
4.  同时用户也可以调用Knight RNE模拟库编写自己的业务应用在纯软件环境仿真自己的业务模型。
5.  如果步骤3、4均通过，用户可以调用Knight RNE运行时库编写自己的实际业务应用，部署到清微芯片上。
6.  在步骤1量化后，如果模型精度损失严重，用户可以使用提供的Knight Finetune库（当前仅支持pytorch平台）编写自己的Finetune工具对浮点模型进行微调，得到更适合量化的浮点模型，之后再进行步骤1。

## 模块简介

### 模型量化

![](https://user-images.githubusercontent.com/46103969/200753745-f0323647-6fc4-46aa-9844-31d882d966d8.png)

&emsp;&emsp;ONNX量化工具支持对ONNX/PaddlePaddle/Caffe/Pytorch/Tensorflow五种格式浮点模型的量化。其中PaddlePaddle，Caffe，Pytorch和Tensorflow格式的模型需要先转换为ONNX模型后再进行模型量化。ONNX量化工具通过加载用户提供的量化数据集和ONNX浮点模型开展量化，量化完成后生成ONNX格式的IR定点模型提供给下游编译器使用。ONNX量化工具默认用int8方式量化。

### 模型编译

![](https://user-images.githubusercontent.com/46103969/200753752-38ac8d69-c3a0-417b-85fa-5cd8ac5ccea1.png)

&emsp;&emsp;ONNX编译工具支持对ONNX/Caffe格式量化好的定点模型进行编译，如果编译成功会生成cfg和weight文件，如果编译失败会有错误编码提示。编译后的文件通过模拟器产生仿真数据。

### 板端模拟

![](https://user-images.githubusercontent.com/46103969/200753717-cae654e7-1490-4d27-baf8-26816489f4a3.png)

&emsp;&emsp;RNE模拟器工具的主旨是通过Linux PC软件模拟硬件的运行情况，根据编译器生成的网络指令部署文件和网络权重部署文件，进行网络推理与功能验证。  
&emsp;&emsp;NE模拟器的原理是模拟硬件算子的处理，根据指令数据决定算子的组合，每个算子根据输入数据、权重数据、量化数据进行数据处理，从而实现整网络的推理。  
&emsp;&emsp;RNE模拟器目前支持多帧输入，支持输入nhwc/nchw格式转换、支持查看最终推理结果输出和每层结果输出，支持查看网络推理前的输入数据。  
&emsp;&emsp;RNE性能分析器工具的主旨是通过Linux PC软件模拟硬件的运行情况，根据编译器生成的网络指令部署文件和网络权重部署文件，进行网络耗时估计。  
&emsp;&emsp;RNE性能分析器的原理是模拟硬件算子的处理，根据指令数据决定算子的组合， 每个算子根据输入数据、权重数据、量化数据进行数据处理耗时分析，从而实现网络的耗时分析。  


## 技术讨论

- Github issues
- [官网](http://www.tsingmicro.com/)
- 技术支持: 010-61934861
- QQ 群: 761954000

