import numpy as np
from model import *

if __name__ == '__main__':

    # 定义加载模型的函数
    def load_model(model_path, model_class):
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
        return model


    # 加载模型
    model_paths = {
        'model1': 'zzz_saved_model/Zhongda_GRU_BN_model_undersample_FocalLoss_50_model_35.pth',
        'model2': 'zzz_saved_model/Zhongda_RNN_BN_4layers_model_undersample_FocalLoss_50_model_18.pth',
        'model3': 'zzz_saved_model/Zhongda_RNN_BN_model_oversample_FocalLoss_50_model_3.pth',
        'model4': 'zzz_saved_model/Zhongda_RNN_BN_model_oversample_FocalLoss_50_model_4.pth',
        'model5': 'zzz_saved_model/Zhongda_RNN_BN_ResBlock_model_oversample_FocalLoss_40_model_12.pth'
    }

    model_kinds = {
        'model1': GRU_BN,
        'model2': RNN_BN_4layers,
        'model3': RNN_BN,
        'model4': RNN_BN,
        'model5': RNN_BN_ResBlock
    }

    models = []
    for name, path in model_paths.items():
        model_class = model_kinds[name]
        model = load_model(path, model_class)
        models.append(model)

    class VotingEnsemble(nn.Module):
        def __init__(self, models):
            super(VotingEnsemble, self).__init__()
            self.models = models

        def forward(self, x):
            outputs = [model(x) for model in self.models]
            outputs = torch.stack(outputs, dim=0)
            avg_output = torch.mean(outputs, dim=0)
            return avg_output

    from sklearn.metrics import roc_auc_score

    def evaluate_model_auroc(model, X, y):
        model.eval()
        with torch.no_grad():
            y_pred_proba = model(X.float()).cpu().numpy()
            auroc = roc_auc_score(y, y_pred_proba)
        return auroc

    # 创建投票集成模型
    voting_model = VotingEnsemble(models)

    models.append(voting_model)

    data = torch.load('E:\deeplearning\Zhongda\data_tensor_zhongda.pth')

    data_train = data['data_tensor_train']
    label_train = data['label_tensor_train']
    data_val = data['data_tensor_val']
    label_val = data['label_tensor_val']
    data_test = data['data_tensor_test']
    label_test = data['label_tensor_test']

    for model in models:
        print(model._get_name())
        # 计算 AUROC
        auroc_val = evaluate_model_auroc(model, data_val, label_val)
        print("AUROC Vali:", auroc_val)

        auroc_test = evaluate_model_auroc(model, data_test, label_test)
        print("AUROC Test:", auroc_val)