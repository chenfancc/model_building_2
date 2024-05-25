from function import main_data_loader
from model_trainer import TrainModel
from model import *

BATCH_SIZE = 256
EPOCH = 50
LR = 1e-4
GAMMA = 0.5
STEP_SIZE = 5  # 每隔多少个 epoch 衰减一次学习率
DECAY = 1e-4
DEVICE = "cuda"
SAMPLE_METHOD = "undersample"

if __name__ == '__main__':
    for i in [20, 24, 30, 36, 48]:
        tensor_direction = f""
        train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD)
        model = BiLSTM_BN
        model_name = f"{model.__name__}_best_model_{SAMPLE_METHOD}sample_{i}"
        trainer = TrainModel(model_name, train_dataloader, val_dataloader)
        trainer.train()
