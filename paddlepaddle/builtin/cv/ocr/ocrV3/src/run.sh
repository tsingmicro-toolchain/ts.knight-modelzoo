Knight --chip TX5368AV200 build --run-config data/ocrV3_cls_config.json
Knight --chip TX5368AV200 build --run-config data/ocrV3_det_config.json
Knight --chip TX5368AV200 build --run-config data/ocrV3_rec_config.json

python replace_operator.py --model /TS-KnightDemo/Output/ocrV3_rec/quant/ocrV3_rec_quantize.onnx --output /TS-KnightDemo/Output/ocrV3_rec/quant/ocrV3_rec_quantize_speed.onnx
python replace_operator.py --model /TS-KnightDemo/Output/ocrV3_det/quant/ocrV3_det_quantize.onnx --output /TS-KnightDemo/Output/ocrV3_det/quant/ocrV3_det_quantize_speed.onnx
python replace_operator.py --model /TS-KnightDemo/Output/ocrV3_cls/quant/ocrV3_cls_quantize.onnx --output /TS-KnightDemo/Output/ocrV3_cls/quant/ocrV3_cls_quantize_speed.onnx

python3 tools/infer/predict_system.py --use_gpu=False --cls_model_dir=/TS-KnightDemo/Output/ocrV3_cls/quant/ocrV3_cls_quantize_speed.onnx --rec_model_dir=/TS-KnightDemo/Output/ocrV3_rec/quant/ocrV3_rec_quantize_speed.onnx --det_model_dir=/TS-KnightDemo/Output/ocrV3_det/quant/ocrV3_det_quantize_speed.onnx --image_dir=deploy/lite/imgs/lite_demo.png --use_onnx=True