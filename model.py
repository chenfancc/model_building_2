import torch
from torch import nn

from hyperparameters import Feature_number


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)

        output, _ = self.lstm(x, (h0, c0))
        # print(output.shape)
        output_hd = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        output_bn = self.bn(output_hd)
        # print(output.shape)
        output_fc1 = torch.relu(self.fc1(output_bn))
        # print(output.shape)
        output_fc2 = torch.relu(self.fc2(output_fc1))
        # print(output.shape)
        output_fc3 = torch.relu(self.fc3(output_fc2))
        # print(output.shape)
        output_fc4 = torch.relu(self.fc4(output_fc3))
        # print(output.shape)
        output_sigmoid = torch.sigmoid(self.fc5(output_fc4))
        # print(output.shape)
        return output_sigmoid.squeeze(1).to(x.device)


class BiLSTM_BN(nn.Module):
    def __init__(self):
        super(BiLSTM_BN, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(32)  # 批标准化层

        self.fc4 = nn.Linear(32, 8)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(8)  # 批标准化层

        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class BiLSTM_BN_larger(nn.Module):
    def __init__(self):
        super(BiLSTM_BN_larger, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=256, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 128)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(32)  # 批标准化层

        self.fc4 = nn.Linear(32, 8)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(8)  # 批标准化层

        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 256).to(x.device)
        c0 = torch.zeros(4, x.size(0), 256).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class BiLSTM_BN_Resnet(nn.Module):
    def __init__(self):
        super(BiLSTM_BN_Resnet, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(256)

        self.fc1_1 = nn.Linear(256, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.ReLU()

        self.fc1_2 = nn.Linear(256, 256)
        self.bn1_2 = nn.BatchNorm1d(256)
        self.relu1_2 = nn.ReLU()

        self.fc1_3 = nn.Linear(256, 256)
        self.bn1_3 = nn.BatchNorm1d(256)
        self.relu1_3 = nn.ReLU()

        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(32, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output_1 = self.fc1_1(output)
        output_1 = self.bn1_1(output_1)
        output_1 = self.relu1_1(output_1)
        output_1 = output_1 + output

        output_2 = self.fc1_2(output_1)
        output_2 = self.bn1_2(output_2)
        output_2 = self.relu1_2(output_2)
        output_2 = output_2 + output_1

        output_3 = self.fc1_3(output_2)
        output_3 = self.bn1_3(output_3)
        output_3 = self.relu1_3(output_3)
        output_3 = output_3 + output_2

        output = self.fc2(output_3)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class BiLSTM_BN_3layers(nn.Module):
    def __init__(self):
        super(BiLSTM_BN_3layers, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=512, num_layers=3, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(6, x.size(0), 512).to(x.device)
        c0 = torch.zeros(6, x.size(0), 512).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class GRU_BN(nn.Module):
    def __init__(self):
        super(GRU_BN, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=1024, num_layers=2, batch_first=True)
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 1024).to(x.device)

        output, _ = self.gru(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class GRU_BN_3layers(nn.Module):
    def __init__(self):
        super(GRU_BN_3layers, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=1024, num_layers=3, batch_first=True)
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(3, x.size(0), 1024).to(x.device)

        output, _ = self.gru(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class GRU_BN_4layers(nn.Module):
    def __init__(self):
        super(GRU_BN_4layers, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=1024, num_layers=4, batch_first=True)
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(4, x.size(0), 1024).to(x.device)

        output, _ = self.gru(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class RNN_BN(nn.Module):
    def __init__(self):
        super(RNN_BN, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=1024, num_layers=2, batch_first=True)  # 修改此处
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 1024).to(x.device)

        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class RNN_BN_3layers(nn.Module):
    def __init__(self):
        super(RNN_BN_3layers, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=1024, num_layers=3, batch_first=True)  # 修改此处
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(3, x.size(0), 1024).to(x.device)

        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class RNN_BN_4layers(nn.Module):
    def __init__(self):
        super(RNN_BN_4layers, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=1024, num_layers=4, batch_first=True)  # 修改此处
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(4, x.size(0), 1024).to(x.device)

        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        self.relu2 = nn.ReLU()

        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu2(out)

        return out


class BiLSTM_BN_ResBlock(nn.Module):
    def __init__(self):
        super(BiLSTM_BN_ResBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(256)

        self.resblock1 = ResBlock(256, 128)  # 添加残差块
        self.resblock2 = ResBlock(128, 64)  # 添加残差块
        self.resblock3 = ResBlock(64, 32)  # 添加残差块

        self.fc4 = nn.Linear(32, 8)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(8)  # 批标准化层

        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.resblock1(output)  # 通过残差块
        output = self.resblock2(output)  # 通过残差块
        output = self.resblock3(output)  # 通过残差块

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class GRU_BN_ResBlock(nn.Module):
    def __init__(self):
        super(GRU_BN_ResBlock, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=1024, num_layers=2, batch_first=True)
        self.bn = nn.BatchNorm1d(1024)

        self.resblock1 = ResBlock(1024, 512)  # 添加残差块
        self.resblock2 = ResBlock(512, 128)  # 添加残差块
        self.resblock3 = ResBlock(128, 64)  # 添加残差块

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 1024).to(x.device)

        output, _ = self.gru(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.resblock1(output)  # 通过残差块
        output = self.resblock2(output)  # 通过残差块
        output = self.resblock3(output)  # 通过残差块

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class RNN_BN_ResBlock(nn.Module):
    def __init__(self):
        super(RNN_BN_ResBlock, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=1024, num_layers=2, batch_first=True)  # 修改此处
        self.bn = nn.BatchNorm1d(1024)

        self.resblock1 = ResBlock(1024, 512)  # 添加残差块
        self.resblock2 = ResBlock(512, 128)  # 添加残差块
        self.resblock3 = ResBlock(128, 64)  # 添加残差块

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 1024).to(x.device)

        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        output = self.resblock1(output)  # 通过残差块
        output = self.resblock2(output)  # 通过残差块
        output = self.resblock3(output)  # 通过残差块

        output = self.fc4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class Custom(nn.Module):
    def __init__(self):
        super(Custom, self).__init__()
        self.fc5 = nn.Linear(5, 1)


    def forward(self, x):
        x = self.fc5(x)
        return torch.zeros(x.shape[0],).to(x.device)

