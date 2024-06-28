Knight --chip TX5368AV200 quant onnx -m yolov8s  \
	-w /ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov8s/weight/yolov8s.pth  \
	-f pytorch  \
	-uds /ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov8s/src/yolov8s.py \
	-if yolov8s \
	-s ./tmp/yolov8s \
	-d /ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov8s/data/coco128.yaml \
	-bs 1 -i 128