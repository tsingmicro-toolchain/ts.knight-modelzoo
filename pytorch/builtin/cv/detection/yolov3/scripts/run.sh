Knight --chip TX5368AV200 quant onnx -m yolov3
	-w /ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov3/weight/yolov3-tiny.pth \
	-f pytorch \
	-uds /ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov3/src/infer_yolov3.py \
	-if infer_yolov3 \
	-s ./tmp/yolov3 \
	-d /ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov3/data/coco128.yaml \
	-bs 1 -i 128 \