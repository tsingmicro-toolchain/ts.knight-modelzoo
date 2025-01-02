Knight --chip TX5368AV200 build --run-config data/yolov8s_config.json
Knight --chip TX5368AV200 quant --run-config data/yolov8s_infer_config.json

python3 src/make_image_input_onnx.py --input /ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov8s/test_data/ --outpath /TS-KnightDemo/Output/yolov8s/npu

Knight --chip TX5368AV200 run --run-config data/yolov8s_config.json

show_sim_result --sim-data /TS-KnightDemo/Output/yolov8s/npu/result-699_p.txt --save-dir /TS-KnightDemo/Output/yolov8s/npu/
show_sim_result --sim-data /TS-KnightDemo/Output/yolov8s/npu/result-719_p.txt --save-dir /TS-KnightDemo/Output/yolov8s/npu/
show_sim_result --sim-data /TS-KnightDemo/Output/yolov8s/npu/result-739_p.txt --save-dir /TS-KnightDemo/Output/yolov8s/npu/


python src/post_process.py --image test_data/bus.jpg --img-size 640 --numpys /TS-KnightDemo/Output/yolov8s/npu/result-699_p.npy /TS-KnightDemo/Output/yolov8s/npu/result-719_p.npy /TS-KnightDemo/Output/yolov8s/npu/result-739_p.npy --scales  0.2689846 0.4109275  0.4786172 --save_dir output






