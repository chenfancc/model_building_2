from function import *
from model_trainer import TrainModel
import numpy as np
from model import *
from function import plot_info


class model_trainer_factory():
    BATCH_SIZE = 256
    EPOCH = 50
    LR = 5e-6
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

    def __init__(self):
        self.BATCH_SIZE = self.hyperparameters["BATCH_SIZE"]
        self.EPOCH = self.hyperparameters["EPOCH"]
        self.LR = self.hyperparameters["LEARNING_RATE"]
        self.GAMMA = self.hyperparameters["GAMMA"]
        self.STEP_SIZE = self.hyperparameters["STEP_SIZE"]
        self.DECAY = self.hyperparameters["DECAY"]
        self.DEVICE = self.hyperparameters["device"]
        self.SEED = self.hyperparameters["SEED"]
        self.ALPHA_LOSS = self.hyperparameters["ALPHA_LOSS"]
        self.GAMMA_LOSS = self.hyperparameters["GAMMA_LOSS"]
        pass

    def ZYY_train(self):
        for tensor_direction in [f'E:\deeplearning\Zhongda\zyy_fbfill_stdscaler_tensor.pth',
                                 f'E:\deeplearning\Zhongda\zyy_miceimpute_minmaxscaler_tensor.pth',
                                 f'E:\deeplearning\Zhongda\zyy_fbfill_minmaxscaler_tensor.pth',
                                 f'E:\deeplearning\Zhongda\zyy_miceimpute_stdscaler_tensor.pth']:

            root_dir = 'ZYY'
            if tensor_direction == f'E:\deeplearning\Zhongda\zyy_miceimpute_minmaxscaler_tensor.pth':
                name = 'ZYY_mice_mm'
                data_process = 'MICE + 最大最小值'
            elif tensor_direction == f'E:\deeplearning\Zhongda\zyy_fbfill_minmaxscaler_tensor.pth':
                name = 'ZYY_fb_mm'
                data_process = '前后填充 + 最大最小值'
            elif tensor_direction == f'E:\deeplearning\Zhongda\zyy_miceimpute_stdscaler_tensor.pth':
                name = 'ZYY_mice_std'
                data_process = 'MICE + 均值方差标准化'
            elif tensor_direction == f'E:\deeplearning\Zhongda\zyy_fbfill_stdscaler_tensor.pth':
                name = 'ZYY_fb_std'
                data_process = '前后填充 + 均值方差标准化'

            # for SAMPLE_METHOD in ["undersample", "origin", "smote", "oversample"]:
            for SAMPLE_METHOD in ["undersample"]:
                # for model in [BiLSTM, BiLSTM_BN, BiLSTM_BN_larger, BiLSTM_BN_Resnet, BiLSTM_BN_3layers, BiLSTM_BN_4layers,
                #               GRU_BN, GRU_BN_3layers, GRU_BN_4layers, GRU_BN_4layers,
                #               RNN_BN, RNN_BN_3layers, RNN_BN_4layers,
                #               BiLSTM_BN_ResBlock, GRU_BN_ResBlock, RNN_BN_ResBlock,
                #               BiLSTM_BN_ResBlock_3layers, GRU_BN_ResBlock_3layers, RNN_BN_ResBlock_3layers,
                #               BiLSTM_BN_single, GRU_BN_single, RNN_BN_single]:
                for model in [RNN_BN_4layers]:
                    print(SAMPLE_METHOD, "_", model.__name__)

                    model_name = f"{name}_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{self.EPOCH}_{self.LR}"

                    print(f"\n")
                    print(
                        "==========================================模型训练开始：==========================================")
                    print(f"\n数据集：中大医院数据\t数据处理：{data_process}\t采样方法：{SAMPLE_METHOD}")
                    print(f'模型：{model_name}')

                    # 设置随机种子
                    np.random.seed(self.SEED)
                    torch.manual_seed(self.SEED)
                    train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction,
                                                                                         SAMPLE_METHOD,
                                                                                         self.BATCH_SIZE)
                    loss_f = FocalLoss(self.ALPHA_LOSS, self.GAMMA_LOSS)
                    trainer = TrainModel(model_name, model, self.hyperparameters, train_dataloader, val_dataloader,
                                         criterion_class=loss_f, root_dir=root_dir)
                    info = trainer.train()
                    trainer.save_model()
                    plot_info(info, model_name, root_dir=root_dir)
        return None

    def ZYY_train_death(self):
        for tensor_name in ['fbfill_mmscaler_death',
                     'fbfill_stdscaler_death',
                     'mice_mmscaler_death',
                     'mice_stdscaler_death']:
            tensor_direction = f'E:\deeplearning\Zhongda\zyy_{tensor_name}_tensor.pth'
            root_dir = 'ZYY_label_death'
            if tensor_direction == f'E:\deeplearning\Zhongda\zyy_mice_mmscaler_death_tensor.pth':
                name = 'ZYY_mice_mm'
                data_process = 'MICE + 最大最小值'
            elif tensor_direction == f'E:\deeplearning\Zhongda\zyy_fbfill_mmscaler_death_tensor.pth':
                name = 'ZYY_fb_mm'
                data_process = '前后填充 + 最大最小值'
            elif tensor_direction == f'E:\deeplearning\Zhongda\zyy_mice_stdscaler_death_tensor.pth':
                name = 'ZYY_mice_std'
                data_process = 'MICE + 均值方差标准化'
            elif tensor_direction == f'E:\deeplearning\Zhongda\zyy_fbfill_stdscaler_death_tensor.pth':
                name = 'ZYY_fb_std'
                data_process = '前后填充 + 均值方差标准化'

            for SAMPLE_METHOD in ["undersample", "origin", "smote", "oversample"]:
                for model in [BiLSTM, BiLSTM_BN, BiLSTM_BN_larger, BiLSTM_BN_Resnet, BiLSTM_BN_3layers, BiLSTM_BN_4layers,
                              GRU_BN, GRU_BN_3layers, GRU_BN_4layers, GRU_BN_4layers,
                              RNN_BN, RNN_BN_3layers, RNN_BN_4layers,
                              BiLSTM_BN_ResBlock, GRU_BN_ResBlock, RNN_BN_ResBlock,
                              BiLSTM_BN_ResBlock_3layers, GRU_BN_ResBlock_3layers, RNN_BN_ResBlock_3layers,
                              BiLSTM_BN_single, GRU_BN_single, RNN_BN_single]:
                    print(SAMPLE_METHOD, "_", model.__name__)

                    model_name = f"{name}_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{self.EPOCH}_{self.LR}"

                    print(f"\n")
                    print(
                        "==========================================模型训练开始：==========================================")
                    print(f"\n数据集：中大医院数据\t数据处理：{data_process}\t采样方法：{SAMPLE_METHOD}")
                    print(f'模型：{model_name}')

                    # 设置随机种子
                    np.random.seed(self.SEED)
                    torch.manual_seed(self.SEED)
                    train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction,
                                                                                         SAMPLE_METHOD,
                                                                                         self.BATCH_SIZE)
                    loss_f = FocalLoss(self.ALPHA_LOSS, self.GAMMA_LOSS)
                    trainer = TrainModel(model_name, model, self.hyperparameters, train_dataloader, val_dataloader,
                                         criterion_class=loss_f, root_dir=root_dir)
                    info = trainer.train()
                    trainer.save_model()
                    plot_info(info, model_name, root_dir=root_dir)
            return None
    def ZYY_train_icu(self):
        for tensor_name in ['fbfill_mmscaler_icu',
                     'fbfill_stdscaler_icu',
                     'mice_mmscaler_icu',
                     'mice_stdscaler_icu']:
            tensor_direction = f'E:\deeplearning\Zhongda\zyy_{tensor_name}_tensor.pth'
            root_dir = 'ZYY_label_icu'
            if tensor_direction == f'E:\deeplearning\Zhongda\zyy_mice_mmscaler_icu_tensor.pth':
                name = 'ZYY_mice_mm'
                data_process = 'MICE + 最大最小值'
            elif tensor_direction == f'E:\deeplearning\Zhongda\zyy_fbfill_mmscaler_icu_tensor.pth':
                name = 'ZYY_fb_mm'
                data_process = '前后填充 + 最大最小值'
            elif tensor_direction == f'E:\deeplearning\Zhongda\zyy_mice_stdscaler_icu_tensor.pth':
                name = 'ZYY_mice_std'
                data_process = 'MICE + 均值方差标准化'
            elif tensor_direction == f'E:\deeplearning\Zhongda\zyy_fbfill_stdscaler_icu_tensor.pth':
                name = 'ZYY_fb_std'
                data_process = '前后填充 + 均值方差标准化'

            for SAMPLE_METHOD in ["undersample", "origin", "smote", "oversample"]:
                for model in [BiLSTM, BiLSTM_BN, BiLSTM_BN_larger, BiLSTM_BN_Resnet, BiLSTM_BN_3layers, BiLSTM_BN_4layers,
                              GRU_BN, GRU_BN_3layers, GRU_BN_4layers, GRU_BN_4layers,
                              RNN_BN, RNN_BN_3layers, RNN_BN_4layers,
                              BiLSTM_BN_ResBlock, GRU_BN_ResBlock, RNN_BN_ResBlock,
                              BiLSTM_BN_ResBlock_3layers, GRU_BN_ResBlock_3layers, RNN_BN_ResBlock_3layers,
                              BiLSTM_BN_single, GRU_BN_single, RNN_BN_single]:
                    print(SAMPLE_METHOD, "_", model.__name__)

                    model_name = f"{name}_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{self.EPOCH}_{self.LR}"

                    print(f"\n")
                    print(
                        "==========================================模型训练开始：==========================================")
                    print(f"\n数据集：中大医院数据\t数据处理：{data_process}\t采样方法：{SAMPLE_METHOD}")
                    print(f'模型：{model_name}')

                    # 设置随机种子
                    np.random.seed(self.SEED)
                    torch.manual_seed(self.SEED)
                    train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction,
                                                                                         SAMPLE_METHOD,
                                                                                         self.BATCH_SIZE)
                    loss_f = FocalLoss(self.ALPHA_LOSS, self.GAMMA_LOSS)
                    trainer = TrainModel(model_name, model, self.hyperparameters, train_dataloader, val_dataloader,
                                         criterion_class=loss_f, root_dir=root_dir)
                    info = trainer.train()
                    trainer.save_model()
                    plot_info(info, model_name, root_dir=root_dir)
            return None
    def Zhongda2_train(self):
        for tensor_direction in [f'E:\deeplearning\Zhongda\data_tensor_zhongda.pth']:

            root_dir = 'Zhongda2'
            name = 'Zhongda2'
            data_process = '前后填充 + 均值方差标准化'

            for SAMPLE_METHOD in ["undersample", "origin", "smote", "oversample"]:
                for model in [BiLSTM, BiLSTM_BN, BiLSTM_BN_larger, BiLSTM_BN_Resnet, BiLSTM_BN_3layers,
                              BiLSTM_BN_4layers,
                              GRU_BN, GRU_BN_3layers, GRU_BN_4layers, GRU_BN_4layers, RNN_BN, RNN_BN_3layers,
                              RNN_BN_4layers,
                              BiLSTM_BN_ResBlock, GRU_BN_ResBlock, RNN_BN_ResBlock,
                              BiLSTM_BN_ResBlock_3layers, GRU_BN_ResBlock_3layers, RNN_BN_ResBlock_3layers,
                              BiLSTM_BN_single, GRU_BN_single, RNN_BN_single]:
                    print(SAMPLE_METHOD, "_", model.__name__)

                    model_name = f"{name}_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{self.EPOCH}_{self.LR}"

                    print(f"\n")
                    print(
                        "==========================================模型训练开始：==========================================")
                    print(f"\n数据集：中大医院数据\t数据处理：{data_process}\t采样方法：{SAMPLE_METHOD}")
                    print(f'模型：{model_name}')

                    # 设置随机种子
                    np.random.seed(self.SEED)
                    torch.manual_seed(self.SEED)
                    train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction,
                                                                                         SAMPLE_METHOD,
                                                                                         self.BATCH_SIZE)
                    loss_f = FocalLoss(self.ALPHA_LOSS, self.GAMMA_LOSS)
                    trainer = TrainModel(model_name, model, self.hyperparameters, train_dataloader, val_dataloader,
                                         criterion_class=loss_f, root_dir=root_dir)
                    info = trainer.train()
                    trainer.save_model()
                    plot_info(info, model_name, root_dir=root_dir)
        return None

    def mimic_train(self):
        for i in [20, 24, 30, 36, 48]:
            tensor_direction = f'E:\deeplearning\Model_building\data_label_1/data_tensor_{i}.pth'
            root_dir = 'mimic2'
            name = 'mimic2'
            data_process = '前后填充 + 均值方差标准化'

            for SAMPLE_METHOD in ["undersample", "origin", "smote", "oversample"]:
                for model in [BiLSTM, BiLSTM_BN, BiLSTM_BN_larger, BiLSTM_BN_Resnet, BiLSTM_BN_3layers,
                              BiLSTM_BN_4layers,
                              GRU_BN, GRU_BN_3layers, GRU_BN_4layers, GRU_BN_4layers, RNN_BN, RNN_BN_3layers,
                              RNN_BN_4layers,
                              BiLSTM_BN_ResBlock, GRU_BN_ResBlock, RNN_BN_ResBlock,
                              BiLSTM_BN_ResBlock_3layers, GRU_BN_ResBlock_3layers, RNN_BN_ResBlock_3layers,
                              BiLSTM_BN_single, GRU_BN_single, RNN_BN_single]:
                    print(SAMPLE_METHOD, "_", model.__name__)

                    model_name = f"{name}_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{self.EPOCH}_{self.LR}"

                    print(f"\n")
                    print(
                        "==========================================模型训练开始：==========================================")
                    print(f"\n数据集：MIMIC数据\t数据处理：{data_process}\t采样方法：{SAMPLE_METHOD}")
                    print(f'模型：{model_name}')

                    # 设置随机种子
                    np.random.seed(self.SEED)
                    torch.manual_seed(self.SEED)
                    train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction,
                                                                                         SAMPLE_METHOD,
                                                                                         self.BATCH_SIZE)
                    loss_f = FocalLoss(self.ALPHA_LOSS, self.GAMMA_LOSS)
                    trainer = TrainModel(model_name, model, self.hyperparameters, train_dataloader, val_dataloader,
                                         criterion_class=loss_f, root_dir=root_dir, Feature_number=7)
                    info = trainer.train()
                    trainer.save_model()
                    plot_info(info, model_name, root_dir=root_dir)


if __name__ == '__main__':
    Trainer = model_trainer_factory()
    # Trainer.ZYY_train_death()
    Trainer.ZYY_train_icu()
