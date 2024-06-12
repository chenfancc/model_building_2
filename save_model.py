import json

from function import main_data_loader, FocalLoss, plot_info
from model import *
from model_trainer import TrainModel

BATCH_SIZE = 512
EPOCH = 100
LR = 1e-5
GAMMA = 0.95
STEP_SIZE = 20  # 每隔多少个 epoch 衰减一次学习率
DECAY = 1e-4
DEVICE = "cuda"
SEED = 42
ALPHA_LOSS = 1
GAMMA_LOSS = 3
Feature_number = 5

dict_epoch = {
    # 'Zhongda_RNN_BN_ResBlock_model_oversample_FocalLoss_40': [12],
    'Zhongda_RNN_BN_model_oversample_FocalLoss_50': [4, 3],
    'Zhongda_RNN_BN_4layers_model_undersample_FocalLoss_50': [18],
    'Zhongda_GRU_BN_model_undersample_FocalLoss_50': [35],
    'Zhongda_GRU_BN_model_origin_FocalLoss_60': [38],
    'Zhongda_GRU_BN_4layers_model_oversample_FocalLoss_40': [1],
    'Zhongda_BiLSTM_BN_Resnet_model_smote_FocalLoss': [1, 3],
    'Zhongda_BiLSTM_BN_ResBlock_model_oversample_FocalLoss_50': [6],
    'Zhongda_BiLSTM_BN_model_smote_FocalLoss': [2],
    'Zhongda_3_BiLSTM_BN_model_undersample_FocalLoss': [48, 97],
    'Zhongda_2_BiLSTM_BN_model_undersample_FocalLoss': [123]
}

dict_sample = {
    'Zhongda_RNN_BN_ResBlock_model_oversample_FocalLoss_40': "oversample",
    'Zhongda_RNN_BN_model_oversample_FocalLoss_50': "oversample",
    'Zhongda_RNN_BN_4layers_model_undersample_FocalLoss_50': "undersample",
    'Zhongda_GRU_BN_model_undersample_FocalLoss_50': "undersample",
    'Zhongda_GRU_BN_model_origin_FocalLoss_60': "origin",
    'Zhongda_GRU_BN_4layers_model_oversample_FocalLoss_40': "oversample",
    'Zhongda_BiLSTM_BN_Resnet_model_smote_FocalLoss': "smote",
    'Zhongda_BiLSTM_BN_ResBlock_model_oversample_FocalLoss_50': "oversample",
    'Zhongda_BiLSTM_BN_model_smote_FocalLoss': "smote",
    'Zhongda_3_BiLSTM_BN_model_undersample_FocalLoss': "undersample",
    'Zhongda_2_BiLSTM_BN_model_undersample_FocalLoss': "undersample"
}

dict_model = {
    'Zhongda_RNN_BN_ResBlock_model_oversample_FocalLoss_40': RNN_BN_ResBlock,
    'Zhongda_RNN_BN_model_oversample_FocalLoss_50': RNN_BN,
    'Zhongda_RNN_BN_4layers_model_undersample_FocalLoss_50': RNN_BN_4layers,
    'Zhongda_GRU_BN_model_undersample_FocalLoss_50': GRU_BN,
    'Zhongda_GRU_BN_model_origin_FocalLoss_60': GRU_BN,
    'Zhongda_GRU_BN_4layers_model_oversample_FocalLoss_40': GRU_BN_4layers,
    'Zhongda_BiLSTM_BN_Resnet_model_smote_FocalLoss': BiLSTM_BN_Resnet,
    'Zhongda_BiLSTM_BN_ResBlock_model_oversample_FocalLoss_50': BiLSTM_BN_ResBlock,
    'Zhongda_BiLSTM_BN_model_smote_FocalLoss': BiLSTM_BN,
    'Zhongda_3_BiLSTM_BN_model_undersample_FocalLoss': BiLSTM_BN,
    'Zhongda_2_BiLSTM_BN_model_undersample_FocalLoss': BiLSTM_BN
}

if __name__ == '__main__':
    for path in dict_epoch:
        file_path = f'{path}/hyperparameters.json'
        with open(file_path, 'r') as file:
            hyperparameters = json.load(file)

        BATCH_SIZE = hyperparameters.get("BATCH_SIZE", 128)
        EPOCH = hyperparameters.get("EPOCH", 10)
        LR = hyperparameters.get("LR", 1e-5)
        GAMMA = hyperparameters.get("GAMMA", 1)
        STEP_SIZE = hyperparameters.get("STEP_SIZE", 100000)
        DECAY = hyperparameters.get("DECAY", 0)
        DEVICE = hyperparameters.get("DEVICE", "cuda")
        SEED = hyperparameters.get("SEED", 42)
        ALPHA_LOSS = hyperparameters.get("ALPHA_LOSS", 1)
        GAMMA_LOSS = hyperparameters.get("GAMMA_LOSS", 1)
        Feature_number = hyperparameters.get("Feature_number", 5)

        SAMPLE_METHOD = dict_sample[path]
        model = dict_model[path]

        tensor_direction = f'E:\deeplearning\Zhongda\data_tensor_zhongda.pth'
        train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD)
        model_name = f"Zhongda_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{EPOCH}"
        trainer = TrainModel(model_name, model, train_dataloader, val_dataloader,
                             criterion_class=FocalLoss(ALPHA_LOSS, GAMMA_LOSS), valid=False, save_model_index=dict_epoch[path])
        _ = trainer.train()


    # Zhongda_RNN_BN_model_oversample_FocalLoss_50	4、3
    # Zhongda_RNN_BN_4layers_model_undersample_FocalLoss_50	18
    # Zhongda_GRU_BN_model_undersample_FocalLoss_50	35
    # Zhongda_GRU_BN_model_origin_FocalLoss_60	38
    # Zhongda_GRU_BN_4layers_model_oversample_FocalLoss_40	1
    # Zhongda_BiLSTM_BN_Resnet_model_smote_FocalLoss	1、3
    # Zhongda_BiLSTM_BN_ResBlock_model_oversample_FocalLoss_50	6
    # Zhongda_BiLSTM_BN_model_smote_FocalLoss	2
    # Zhongda_3_BiLSTM_BN_model_undersample_FocalLoss	48、97
    # Zhongda_2_BiLSTM_BN_model_undersample_FocalLoss	123