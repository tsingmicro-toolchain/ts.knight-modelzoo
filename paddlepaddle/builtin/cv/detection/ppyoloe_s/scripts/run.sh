Knight --chip TX5368AV200 quant onnx -m /ts.knight-modelzoo/paddlepaddle/builtin/cv/detection/ppyoloe_s/weight/ppyoloe_s.pdmodel -w /ts.knight-modelzoo/paddlepaddle/builtin/cv/detection/ppyoloe_s/weight/ppyoloe_s.pdiparams -f paddle -r convert -s ./tmp/ppyoloe_s

cd /ts.knight-modelzoo/paddlepaddle/builtin/cv/detection/ppyoloe_s/src
python cut_graph.py --model ./tmp/ppyoloe_s/ppyoloe_s.onnx -on transpose_3.tmp_0 p2o.Concat.31 -sn ppyoloe_s -s ./tmp/ppyoloe_s

Knight --chip TX5368AV200 quant onnx -m tmp/ppyoloe_s/ppyoloe_s.onnx -uds /ts.knight-modelzoo/paddlepaddle/builtin/cv/detection/ppyoloe_s/src/ppyoloe_s.py -if ppyoloe_s -s ./tmp/ppyoloe_s -d /ts.knight-modelzoo/paddlepaddle/builtin/cv/detection/ppyoloe_s/data/coco128.yaml -bs 1 -i 128