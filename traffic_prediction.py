import os
import time
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from traffic_dataset import LoadData
from utils import Evaluation
from utils import visualize_result
from chebnet import ChebNet
from gat import GATNet
import seaborn as sns

from sklearn.metrics import confusion_matrix

class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        graph_data = GCN.process_graph(graph_data)

        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1

        output_1 = self.linear_1(flow_x)  # [B, N, hid_C]
        output_1 = self.act(torch.matmul(graph_data, output_1))  # [N, N], [B, N, Hid_C]

        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C]

        return output_2.unsqueeze(2)

    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
        graph_data += matrix_i  # A~ [N, N]

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
        degree_matrix = degree_matrix.pow(-1 / 2)
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

        degree_matrix = torch.diag(degree_matrix)  # [N, N]

        return torch.mm(torch.mm(degree_matrix, graph_data), degree_matrix)  # D^(-1) * A = \hat(A)

class GCN_LSTM(nn.Module):
    def __init__(self, input_num, hid_num, layers_num, out_num, batch_first=True):
        super().__init__()
        self.l1 = nn.LSTM(
            input_size=input_num,
            hidden_size=hid_num,
            num_layers=layers_num,
            batch_first=batch_first
        )
        self.out = nn.Linear(out_num * 6, out_num)
        self.act = nn.ReLU()

    def forward(self, data,device):
        flow_x = data['flow_x'].to(device)  # 把x放入到GPU中运算（B,N,T,F）->(B,307,6,1),把6个时间点的特征压缩到一起作为输入，所以需要view一下
        B, T, N = flow_x.size(0), flow_x.size(2), flow_x.size(1)




        # flow_x = flow_x.transpose(2, 1)
        # flow_x = flow_x.view(B, T, -1)  # 把特征压缩到一起（B，T，N，F）->（B,T,N*F）==(B,6,307,1)->(B,6,307*1)
        # # print(flow_x.shape)
        # l_out, (h_n, c_n) = self.l1(flow_x, None)  # None表示第一次 hidden_state是0
        # # print(l_out.shape)
        # l_out = l_out.view(B, T, N, -1)  # (B,6,307,1)
        # l_out = l_out.transpose(2, 1)
        # l_out = l_out.view(B, N, -1)





        flow_x = flow_x.view(B, N, -1)#(B,307,6,1)--->(B,307,6*1)
        l_out, (h_n, c_n) = self.l1(flow_x, None)  # None表示第一次 hidden_state是0





        adj_matrix = data['graph'].to(device)[0]  # 把邻接矩阵提取出来并加载进cuda进行运算
        adj_matrix = GCN_LSTM.process_adjacent_matrix(adj_matrix)  # 处理后的邻接矩阵hat A

        out_put1 = self.act(torch.matmul(adj_matrix, l_out))

        out = self.out(out_put1)

        out = self.act(torch.matmul(adj_matrix, out))

        out_put2 = out.unsqueeze(2)  ##(B,T,out_channel)->(B,T,1,out_channel),out_channel=1,和label的特征维度一致
        # print(out.shape)
        return out_put2

    @staticmethod
    def process_adjacent_matrix(adj_matrix):
        '''

        :param adj_matrix: 这里是得到原始邻接矩阵
        :return: 返回处理好的hat 邻接矩阵
        '''
        N = adj_matrix.size(0)  # 得到节点数量，便于下面构造度矩阵
        # print(N)
        init_degree = torch.eye(N, dtype=adj_matrix.dtype,
                                device=adj_matrix.device)  # （N,N）->(307,307)初始化一个主对角线全是1的对角矩阵，这其实也即是一个自连接的邻接矩阵并把模型放在GPU中

        adj_matrix = init_degree + adj_matrix  # 建立自连接，A+I

        degree_matrix = torch.sum(adj_matrix, dim=1, keepdim=False)  # （N）,这一步将邻接矩阵的每个点的各边权重加起来，也就是列相加起来，然后压缩成1行N列
        degree_matrix = torch.pow(degree_matrix, -1)  # 求度矩阵的逆->D_-1,如果有的度为0，那么求逆就会得到无穷大inf

        degree_matrix[degree_matrix == float('inf')] = 0  # 将无穷大的数全部改为0

        degree_matrix = torch.diag(degree_matrix)  # 将（N）形状的度矩阵扩展成（N，N），对角矩阵

        return torch.mm(degree_matrix, adj_matrix)  # 返回hatA=D_-1 * adj_matrix


class Baseline(nn.Module):
    def __init__(self, in_c, out_c):
        super(Baseline, self).__init__()
        self.layer = nn.Linear(in_c, out_c)

    def forward(self, data, device):
        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1

        output = self.layer(flow_x)  # [B, N, Out_C], Out_C = D

        return output.unsqueeze(2)  # [B, N, 1, D=Out_C]


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Loading Dataset
    train_data = LoadData(data_path=["PeMS_08/PeMS08.csv", "PeMS_08/PeMS08.npz"], num_nodes=170, divide_days=[48, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=48, shuffle=True, num_workers=0)

    test_data = LoadData(data_path=["PeMS_08/PeMS08.csv", "PeMS_08/PeMS08.npz"], num_nodes=170, divide_days=[48, 14],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=24, shuffle=False, num_workers=0)

    # Loading Model

    # my_net = GATNet(in_c=6 * 1, hid_c=6, out_c=1,n_heads=2)
    #my_net = ChebNet(in_c= 6* 1, hid_c=6, out_c=1,K=2,input_num=6,hid_num=6,layers_num=3)
    #my_net = GCN_LSTM(input_num=170,hid_num=170,layers_num=3,out_num=1)

    my_net = GCN(in_c=6, hid_c=6, out_c=1)
    #my_net = GCN_LSTM(input_num=6, hid_num=6, layers_num=3, out_num=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    my_net = my_net.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(params=my_net.parameters())

    # Train model
    Epoch = 50

    my_net.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            my_net.zero_grad()

            predict_value = my_net(data, device).to(torch.device("cpu"))  # [0, 1] -> recover

            loss = criterion(predict_value, data["flow_y"])

            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()
        end_time = time.time()

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                          (end_time-start_time)/60))

    # Test Model
    my_net.eval()
    with torch.no_grad():
        MAE, MAPE, RMSE = [], [], []
        Target = np.zeros([170, 1, 1]) # [N, 1, D]
        Predict = np.zeros_like(Target)  #[N, T, D]

        total_loss = 0.0
        for data in test_loader:

            predict_value = my_net(data, device).to(torch.device("cpu"))  # [B, N, 1, D]  -> [1, N, B(T), D]

            loss = criterion(predict_value, data["flow_y"])

            total_loss += loss.item()

            predict_value = predict_value.transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]
            target_value = data["flow_y"].transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]

            performance, data_to_save = compute_performance(predict_value, target_value, test_loader)

            Predict = np.concatenate([Predict, data_to_save[0]], axis=1)
            Target = np.concatenate([Target, data_to_save[1]], axis=1)

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])

        print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))

    print("Performance:  MAE {:2.2f}    {:2.2f}%    {:2.2f}".format(np.mean(MAE), np.mean(MAPE * 100), np.mean(RMSE)))

    Predict = np.delete(Predict, 0, axis=1)
    Target = np.delete(Target, 0, axis=1)

    result_file = "LSTM_chebnet1_result_08.h5"
    file_obj = h5py.File(result_file, "w")

    file_obj["predict"] = Predict  
    file_obj["target"] = Target


def compute_performance(prediction, target, data):
    try:
        dataset = data.dataset  # dataloader
    except:
        dataset = data  # dataset

    prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy())
    target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], target.numpy())

    mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data


if __name__ == '__main__':
    main()
    visualize_result(h5_file="LSTM_chebnet1_result_08.h5",
                      nodes_id=120,
                     time_se=[0, 24*12],
                     visualize_file="LSTM_chebnet_08_node_120")

