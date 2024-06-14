from function import *
from model_trainer import TrainModel
import numpy as np
from model import *
from function import plot_info

BATCH_SIZE = 512
EPOCH = 100
LR = 5e-06
GAMMA = 0.95
STEP_SIZE = 20  # 每隔多少个 epoch 衰减一次学习率
DECAY = 1e-4
DEVICE = "cuda"
SEED = 42
ALPHA_LOSS = 1
GAMMA_LOSS = 3

hyperparameters = {
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCH": EPOCH,
    "LEARNING_RATE": LR,
    "GAMMA": GAMMA,
    "STEP_SIZE": STEP_SIZE,
    "DECAY": DECAY,
    "device": DEVICE,
    "SEED": SEED,
    "ALPHA_LOSS": ALPHA_LOSS,
    "GAMMA_LOSS": GAMMA_LOSS
}

# 设置随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)

# if __name__ == '__main__':
#     for SAMPLE_METHOD in ["undersample", "smote"]:
#         for model in [BiLSTM_BN, BiLSTM, BiLSTM_BN_Resnet, BiLSTM_BN_3layers]:
#             for i in [20, 24, 30, 36, 48]:
#                 tensor_direction =f'E:\deeplearning\Model_building\data_label_1/data_tensor_{i}.pth'
#                 train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD)
#                 model_name = f"{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{i}"
#                 trainer = TrainModel(model_name, model, train_dataloader, val_dataloader,
#                                      criterion_class=FocalLoss(ALPHA_LOSS, GAMMA_LOSS))
#                 info = trainer.train()
#                 plot_info(info, model_name)


# if __name__ == '__main__':
#     for SAMPLE_METHOD in ["undersample", "smote"]:
#         for model in [BiLSTM_BN, BiLSTM, BiLSTM_BN_Resnet, BiLSTM_BN_3layers]:
#             tensor_direction = f'E:\deeplearning\Zhongda\data_tensor_zhongda.pth'
#             train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD)
#             model_name = f"Zhongda_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss"
#             trainer = TrainModel(model_name, model, train_dataloader, val_dataloader,
#                                  criterion_class=FocalLoss(ALPHA_LOSS, GAMMA_LOSS))
#             info = trainer.train()
#             plot_info(info, model_name)


# if __name__ == '__main__':
#     for SAMPLE_METHOD in ["origin"]:
#         for model in [BiLSTM_BN]:
#             tensor_direction = f'E:\deeplearning\Zhongda\data_tensor_zhongda.pth'
#             train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD)
#             model_name = f"Zhongda_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss"
#             trainer = TrainModel(model_name, model, train_dataloader, val_dataloader,
#                                  criterion_class=FocalLoss(ALPHA_LOSS, GAMMA_LOSS))
#             info = trainer.train()
#             plot_info(info, model_name)


# if __name__ == '__main__':
#     for SAMPLE_METHOD in ["undersample", "origin"]:
#         for model in [GRU_BN]:
#             tensor_direction = f'E:\deeplearning\Zhongda\data_tensor_zhongda.pth'
#             train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD)
#             model_name = f"Zhongda_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{EPOCH}"
#             trainer = TrainModel(model_name, model, train_dataloader, val_dataloader,
#                                  criterion_class=FocalLoss(ALPHA_LOSS, GAMMA_LOSS))
#             info = trainer.train()
#             plot_info(info, model_name)

if __name__ == '__main__':
    for SAMPLE_METHOD in ["undersample", "origin", "smote", "oversample"]:
        for model in [BiLSTM, BiLSTM_BN, BiLSTM_BN_larger, BiLSTM_BN_Resnet, BiLSTM_BN_3layers, BiLSTM_BN_4layers,
                      GRU_BN, GRU_BN_3layers, GRU_BN_4layers, GRU_BN_4layers, RNN_BN, RNN_BN_3layers, RNN_BN_4layers,
                      BiLSTM_BN_ResBlock, GRU_BN_ResBlock, RNN_BN_ResBlock,
                      BiLSTM_BN_ResBlock_3layers, GRU_BN_ResBlock_3layers, RNN_BN_ResBlock_3layers,
                      BiLSTM_BN_single, GRU_BN_single, RNN_BN_single]:
            tensor_direction = f'E:\deeplearning\Zhongda\data_tensor_zhongda.pth'
            train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD, BATCH_SIZE)
            model_name = f"ZYY_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{EPOCH}"
            loss_f = FocalLoss(ALPHA_LOSS, GAMMA_LOSS)
            trainer = TrainModel(model_name, model, hyperparameters, train_dataloader, val_dataloader,
                                 criterion_class=loss_f, root_dir='ZYY')
            info = trainer.train()
            trainer.save_model()
            plot_info(info, model_name, root_dir='ZYY')

# if __name__ == '__main__':
#     for SAMPLE_METHOD in ["undersample"]:
#         for model in [RNN_BN_ResBlock]:
#             tensor_direction = f'E:\deeplearning\Zhongda\data_tensor_zhongda.pth'
#             train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD)
#             model_name = f"Zhongda_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{EPOCH}"
#             trainer = TrainModel(model_name, model, train_dataloader, val_dataloader,
#                                  criterion_class=FocalLoss(ALPHA_LOSS, GAMMA_LOSS))
#             info = trainer.train()
#             plot_info(info, model_name)
