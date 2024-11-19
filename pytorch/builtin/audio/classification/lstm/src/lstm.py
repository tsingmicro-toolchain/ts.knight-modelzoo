import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from onnx_quantize_tool.common.register import onnx_infer_func, pytorch_model
class RnnCommon(nn.Module):
    def __init__ (self, model_type, input_dim, hidden_dim,n_layer, ed_class):
        super(RnnCommon,self).__init__()
        self.model_type = model_type
        self.n_layer= n_layer
        self.ed_class= ed_class
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if self.model_type == 'gru':
            self.gru =nn.GRU(input_dim, hidden_dim, n_layer, batch_first=True)
            self.fc_1 =nn.Linear(hidden_dim, ed_class)
        elif self.model_type == 'bgru':
            self.gru= nn.GRU(input_dim, hidden_dim,n_layer, batch_first=True, bidirectional=True)
            self.fc_1=nn.Linear(2 *hidden_dim, ed_class)
        elif self.model_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, hidden_dim,n_layer, batch_first=True)
            self.fc_1 =nn.Linear(hidden_dim, ed_class)
        elif self.model_type == 'blstm':
            self.lstm = nn.LSTM(input_dim, hidden_dim, n_layer, batch_first=True, bidirectional=True)
            self.fc_1 =nn.Linear(2 *hidden_dim, ed_class)
        elif self.model_type == 'rnn':
            self.rnn = nn.RNN(input_dim, hidden_dim, n_layer, batch_first=True)
            self.fc = nn.Linear(hidden_dim, ed_class)
        elif self.model_type == 'brnn':
            self.rnn = nn.RNN(input_dim, hidden_dim, n_layer, batch_first=True, bidirectional=True)
            self.fc= nn.Linear(2 *hidden_dim,ed_class)
        else:
            raise ValueError("Not support model:{}".format(self.model_type))
    def forward(self,x):
        if self.model_type == 'gru' or self.model_type== 'bgru':
            out, h_n=self.gru(x)
            out = out[:,24:25,:]
            out = out.squeeze(1)
            out =self.fc_1(out)
        elif self.model_type =='lstm' or self.model_type == 'blstm':
            out,hn= self.lstm(x)
            out = out[:, 24:25, :]
            out = out.squeeze(1)
            out = self.fc_1(out)
        elif self.model_type =='rnn' or self.model_type == 'brnn':
            out,h_n= self.rnn(x)
            out = out[:,24:25,:]
            out =out.squeeze(1)
            out = self.fc(out)
        else:
            raise
        return out
def load_cmd_data():
    test_data ="test/model_zoo/data_set/dnn/cmd_12_test.pkl"
    with open(test_data, 'rb') as fp:
        feat_label =pickle.load(fp,encoding='iso-8859-1')
        test_idx = np.asarray(range(len(feat_label)))
    return feat_label, test_idx

@onnx_infer_func.register("infer_pytorch_rnn")
def infer_pytorch_rnn(executor):
    batch_size =executor.batch_size
    iters =executor.iteration
    feat_label,test_idx = load_cmd_data()
    data_idx =test_idx
    num_of_batch = int(len(data_idx)/ batch_size)+ 1
    eval_acc =0.
    with torch.no_grad():
        for i in range(num_of_batch):
            np_feats = np.zeros(shape=(batch_size,25,40),dtype=np.float32)
            ed_label =np.zeros(shape=(batch_size,1),dtype=np.int64)
            for j in range(batch_size):
                idx = data_idx[int(i * batch_size + j)% len(data_idx)]
                np_feats[j]= feat_label[idx]['input']
                ed_label[j]= feat_label[idx]['label'].argmax()
            th_feats = np_feats
            ed_label = ed_label
            th_feats = th_feats
            ed_out = executor.forward(th_feats)
            pred =np.argmax(ed_out[0],1)
            num_correct =(pred == ed_label[:,0]).sum()
            eval_acc += num_correct
            print("num correct:",num_correct.item()*100./batch_size)
            if iters == i+1:
                break
        total_acurracy =(100.0*eval_acc)/((i+1)*batch_size)
        print("Inference Accuracy:{:.6f}".format(total_acurracy))
        return total_acurracy

@pytorch_model.register("lstm")
def lstm(weight_path=None):
    model = RnnCommon('lstm',40,24,4,12)
    if weight_path:
        model.load_state_dict(torch.load(weight_path,map_location=lambda storage, loc: storage))
        in_dict ={"model": model,"inputs":[torch.randn((2,25,40))]}
    return in_dict