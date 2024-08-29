import paddle
from paddle.jit import to_static
from paddle.static import InputSpec
from resnetx import ResNeXt50_32x4d
pretrained = 'ResNeXt50_32x4d_pretrained'
model = ResNeXt50_32x4d(pretrained)
# 通过 InputSpec 设置 Placeholder 信息
x_spec = InputSpec(shape=[1, 3,  224, 224], name='x')

model = paddle.jit.to_static(model, input_spec=[x_spec])  # 动静转换
paddle.jit.save(model, 'resnet50')