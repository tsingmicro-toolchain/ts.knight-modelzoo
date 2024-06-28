Knight --chip TX5368AV200 quant onnx -m yolox_s \
    		-w /ts.knight-modelzoo/pytorch/builtin/cv/detection/yolovxs/weight/yolox-s.pth  \
    		-f pytorch  \
    		-uds /ts.knight-modelzoo/pytorch/builtin/cv/detection/yolovxs/src/infer_yolovx_small.py  \
    		-if infer_yolox_small \
			-s ./tmp/yolovxs \
    		-bs 1 -i 128