import math
import warnings
from matplotlib import pyplot as plt
from model_trainer import TrainModel
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from model import *
from function import plot_info

warnings.filterwarnings("ignore", category=UserWarning)

# 设置随机种子
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def synthetic_data(rate, num_sequences, time_steps, num_features, beta=1.0):
    """
    0为正弦，1为直线
    形状为[num_sequences, time_steps, num_features]
    :param rate:
    :param num_sequences:
    :param time_steps:
    :param num_features:
    :param beta:
    :return:
    """
    X = torch.zeros(num_sequences, time_steps, num_features)
    Y = torch.zeros(num_sequences, )
    generate_type = 0
    num_sequences_int = math.ceil(num_sequences * rate)
    for i in tqdm(range(num_sequences_int), desc="Generating sine sequences"):  # 添加进度条
        Y[i] = generate_type
        for k in range(num_features):
            for j in range(time_steps):
                X[i, j, k] = np.sin(j) - (np.random.random() - 0.5) * beta
    generate_type = 1
    for i in tqdm(range(num_sequences_int, num_sequences), desc="Generating linear sequences"):  # 添加进度条
        Y[i] = generate_type
        for k in range(num_features):
            for j in range(time_steps):
                X[i, j, k] = (np.random.random() - 0.5) * beta
    return X, Y


if __name__ == '__main__':
    rate_m = 0.02
    beta_m = 2
    data_train, label_train = synthetic_data(1 - rate_m, 1000, 24, 7, beta_m)
    data_test, label_test = synthetic_data(1 - rate_m, 10, 24, 7, beta_m)
    data_val, label_val = synthetic_data(1 - rate_m, 1000, 24, 7, beta_m)

    y_1_values = data_train[0, :, 0].tolist()
    y_2_values = data_train[-1, :, 0].tolist()
    x_values = torch.arange(24).tolist()
    plt.plot(x_values, y_1_values, y_2_values)
    plt.show()

    dataset_train = TensorDataset(data_train, label_train)
    dataset_test = TensorDataset(data_test, label_test)
    dataset_val = TensorDataset(data_val, label_val)

    BATCH_SIZE = 32
    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    ##################
    for model in [BiLSTM, BiLSTM_BN, BiLSTM_BN_Resnet, BiLSTM_BN_3layers]:
    # for model in [BiLSTM_BN]:
        model_name = f"{model.__name__}_simulation_test_batchnorm"
        trainer = TrainModel(model_name, model, train_dataloader, val_dataloader)
        info = trainer.train()
        plot_info(info, model_name)
