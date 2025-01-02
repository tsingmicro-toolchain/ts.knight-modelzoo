#复现脚本
docker run -it --rm  -v /data/home/chenfan/yolov8_ts:/tmp/  ts.knight:2.2.0.8-test /bin/bash
Knight quant -rc yolov8s_onnx_3_config.json



