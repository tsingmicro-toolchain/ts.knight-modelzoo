Knight --chip TX5368AV200 quant --run-config data/yolox_s_config.json
Knight --chip TX5368AV200 quant --run-config data/yolox_s_infer_config.json
Knight --chip TX5368AV200 build --run-config data/yolox_s_config.json
python src/make_image_input_onnx.py --input test_data/bus.jpg --outpath /TS-KnightDemo/Output/yolox_s/npu

Knight --chip TX5368AV200 run --run-config data/yolox_s_config.json
show_sim_result --sim-data /TS-KnightDemo/Output/yolox_s/npu/result-817_p.txt --save-dir /TS-KnightDemo/Output/yolox_s/npu/
show_sim_result --sim-data /TS-KnightDemo/Output/yolox_s/npu/result-843_p.txt --save-dir /TS-KnightDemo/Output/yolox_s/npu/
show_sim_result --sim-data /TS-KnightDemo/Output/yolox_s/npu/result-869_p.txt --save-dir /TS-KnightDemo/Output/yolox_s/npu/
python src/post_process.py --image test_data/bus.jpg --numpys  /TS-KnightDemo/Output/yolox_s/npu/result-817_p.npy /TS-KnightDemo/Output/yolox_s/npu/result-843_p.npy /TS-KnightDemo/Output/yolox_s/npu/result-869_p.npy  --scales 0.02812371 0.02926355 0.02422856 --save_dir output