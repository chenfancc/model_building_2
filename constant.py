from hyperparameters import ALPHA_LOSS, GAMMA_LOSS
from model_trainer import TrainModel
from function import main_data_loader, FocalLoss, plot_info
from model import *
EPOCH = 2
if __name__ == '__main__':
    for SAMPLE_METHOD in ["undersample"]:
        for model in [Custom]:
            tensor_direction = f'E:\deeplearning\Zhongda\data_tensor_zhongda.pth'
            train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD)
            model_name = f"Zhongda_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{EPOCH}"
            trainer = TrainModel(model_name, model, train_dataloader, val_dataloader,
                                 criterion_class=FocalLoss(ALPHA_LOSS, GAMMA_LOSS))
            info = trainer.train()
            plot_info(info, model_name)