import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
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
        self.lstm = nn.LSTM(input_size=7, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
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


class BiLSTM_BN_Resnet(nn.Module):
    def __init__(self):
        super(BiLSTM_BN_Resnet, self).__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
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
        self.lstm = nn.LSTM(input_size=7, hidden_size=512, num_layers=3, batch_first=True, bidirectional=True)
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
