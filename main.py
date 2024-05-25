from function import *
from model_trainer import TrainModel
import numpy as np
from model import *
from function import plot_info

# 设置随机种子
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == '__main__':
    for i in [20, 24, 30, 36, 48]:
        for model in [BiLSTM, BiLSTM_BN, BiLSTM_BN_Resnet, BiLSTM_BN_3layers]:
            tensor_direction =f'E:\deeplearning\Model_building\data_label_1/data_tensor_{i}.pth'
            train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD)
            model_name = f"{model.__name__}_best_model_{SAMPLE_METHOD}_{i}"
            trainer = TrainModel(model_name, model, train_dataloader, val_dataloader)
            info = trainer.train()
            plot_info(info, model_name)