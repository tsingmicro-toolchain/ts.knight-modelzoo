Knight --chip TX5368AV200 quant onnx -m yolo_nas_s \ 
    		-f pytorch  \
    		-uds /ts.knight-modelzoo/pytorch/builtin/cv/detection/yolonas_s/src/export.py  \
    		-if infer_auto \
			-s ./tmp/yolonas_s 