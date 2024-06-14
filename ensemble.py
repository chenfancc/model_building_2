import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
from torch.utils.data import TensorDataset, DataLoader


class VotingEnsemble(nn.Module):
    def __init__(self, models, voting_type='soft'):
        super(VotingEnsemble, self).__init__()
        self.models = models
        self.voting_type = voting_type

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)

        if self.voting_type == 'soft':
            avg_output = torch.mean(outputs, dim=0)
            return avg_output

        elif self.voting_type == 'hard':
            binary_output = (outputs > 0.5).int()
            votes = torch.sum(binary_output, dim=0)
            num = outputs.shape[0] - outputs.shape[0] // 2
            hard_voted_result = (votes >= num).int()
            return hard_voted_result


        else:
            raise ValueError("voting_type must be 'soft' or 'hard'")


def evaluate_model_auc(model, data_loader):
    model.eval()
    all_preds = []
    all_y = []
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to("cuda")
            y_pred_proba = model(X.float()).cpu().numpy()
            all_preds.append(y_pred_proba)
            all_y.append(y)
    all_preds = np.concatenate(all_preds, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    fpr, tpr, _ = roc_curve(all_y, all_preds)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_y, all_preds)
    auprc = auc(recall, precision)
    return auroc, auprc


def voting(data_dir, models_info, selected_model, batch_size):
    train_ROC = []
    train_PRC = []
    val_ROC = []
    val_PRC = []
    test_ROC = []
    test_PRC = []

    if len(selected_model) == 0:
        selected_model = [idx for idx, _ in models_info.items()]

    models = []
    for idx, model_info in models_info.items():
        if idx in selected_model:
            model = torch.load(model_info['model_path']).to('cuda')
            models.append(model)

    data = torch.load(data_dir)
    data_train = data['data_tensor_train']
    label_train = data['label_tensor_train']
    data_val = data['data_tensor_val']
    label_val = data['label_tensor_val']
    data_test = data['data_tensor_test']
    label_test = data['label_tensor_test']
    i = 1

    dataset_train = TensorDataset(data_train, label_train)
    dataset_val = TensorDataset(data_val, label_val)
    dataset_test = TensorDataset(data_test, label_test)

    # 利用 DataLoader 来加载数据集
    train_dataloader_f = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_dataloader_f = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    test_dataloader_f = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    voting_type = 'soft'
    print("Voting Type:", voting_type)
    voting_model = VotingEnsemble(models, voting_type=voting_type)
    print("Voting Model Results:")

    auroc, auprc = evaluate_model_auc(voting_model, train_dataloader_f)
    train_ROC.append(auroc)
    train_PRC.append(auprc)
    print("\tAUROC train:", auroc)
    print("\tAUPRC train:", auprc)

    auroc, auprc = evaluate_model_auc(voting_model, val_dataloader_f)
    val_ROC.append(auroc)
    val_PRC.append(auprc)
    print("\tAUROC val:", auroc)
    print("\tAUPRC val:", auprc)

    auroc, auprc = evaluate_model_auc(voting_model, test_dataloader_f)
    test_ROC.append(auroc)
    test_PRC.append(auprc)
    print("\tAUROC test:", auroc)
    print("\tAUPRC test:", auprc)

    voting_type = 'hard'
    print("\nVoting Type:", voting_type)
    voting_model = VotingEnsemble(models, voting_type=voting_type)
    print("Voting Model Results:")

    auroc, auprc = evaluate_model_auc(voting_model, train_dataloader_f)
    train_ROC.append(auroc)
    train_PRC.append(auprc)
    print("\tAUROC train:", auroc)
    print("\tAUPRC train:", auprc)

    auroc, auprc = evaluate_model_auc(voting_model,val_dataloader_f)
    val_ROC.append(auroc)
    val_PRC.append(auprc)
    print("\tAUROC val:", auroc)
    print("\tAUPRC val:", auprc)

    auroc, auprc = evaluate_model_auc(voting_model, test_dataloader_f)
    test_ROC.append(auroc)
    test_PRC.append(auprc)
    print("\tAUROC test:", auroc)
    print("\tAUPRC test:", auprc)

    print("\n")

    for name in selected_model:
        print(f'{i}.', models_info[name]['model_name'])
        auroc, auprc = evaluate_model_auc(models[i - 1], train_dataloader_f)
        train_ROC.append(auroc)
        train_PRC.append(auprc)
        print("\tAUROC train:", auroc)
        print("\tAUPRC train:", auprc)

        auroc, auprc = evaluate_model_auc(models[i - 1],val_dataloader_f)
        val_ROC.append(auroc)
        val_PRC.append(auprc)
        print("\tAUROC val:", auroc)
        print("\tAUPRC val:", auprc)

        auroc, auprc = evaluate_model_auc(models[i - 1], test_dataloader_f)
        test_ROC.append(auroc)
        test_PRC.append(auprc)
        print("\tAUROC test:", auroc)
        print("\tAUPRC test:", auprc)
        i += 1

    auc_dict = {
        'train_roc': train_ROC,
        'train_prc': train_PRC,
        'val_roc': val_ROC,
        'val_prc': val_PRC,
        'test_roc': test_ROC,
        'test_prc': test_PRC
    }

    return auc_dict


def T(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def plot_auc(auc, selected_model):
    labels = ['soft', 'hard'] + [f'model{i - 2}' for i in range(3, len(auc))]
    x = T([auc['train_prc'], auc['val_prc'], auc['test_prc']])
    y = T([auc['train_roc'], auc['val_roc'], auc['test_roc']])
    # Example assuming len(auc) is known or calculated
    num_series = len(auc)
    base_colors = ['red', 'blue']
    cmap = plt.get_cmap('tab10')  # 选择不同的颜色映射
    colors_add = [cmap(i) for i in range(num_series - 2)]
    colors = base_colors + colors_add
    markers = ['o', 's', '*']  # Define markers for train, val, test respectively
    labels_2 = ['train', 'val', 'test']

    plt.figure(figsize=(6, 4))  # Adjust figure size as needed
    for i, label in enumerate(labels):
        for j in range(len(x[i])):
            plt.scatter(x[i][j], y[i][j], label=str(label) + '_' + labels_2[j], color=colors[i], s=50, marker=markers[j])

    name_list = '_'.join(selected_model)
    plt.xlabel('PRC')
    plt.ylabel('ROC')
    plt.title('ROC vs PRC')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'ZYY/zzz_saved_model/auc_{name_list}.png', bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    data_dir = f'E:\deeplearning\Zhongda\zyy_tensor.pth'

    # models_info = {
    #     "model1": {
    #         "model_name": "BiLSTM_BN, smote",
    #         "model_path": "zzz_saved_model/Zhongda_BiLSTM_BN_model_smote_FocalLoss_100_model_2.pth"
    #     },
    #     "model2": {
    #         "model_name": "BiLSTM_BN, undersample",
    #         "model_path": "zzz_saved_model/Zhongda_BiLSTM_BN_model_undersample_FocalLoss_500_model_123.pth"
    #     },
    #     "model3": {
    #         "model_name": "BiLSTM_BN, undersample",
    #         "model_path": "zzz_saved_model/Zhongda_BiLSTM_BN_model_undersample_FocalLoss_1000_model_48.pth"
    #     },
    #     "model4": {
    #         "model_name": "BiLSTM_BN, undersample",
    #         "model_path": "zzz_saved_model/Zhongda_BiLSTM_BN_model_undersample_FocalLoss_1000_model_97.pth"
    #     },
    #     "model5": {
    #         "model_name": "BiLSTM_BN_ResBlock, oversample",
    #         "model_path": "zzz_saved_model/Zhongda_BiLSTM_BN_ResBlock_model_oversample_FocalLoss_50_model_6.pth"
    #     },
    #     "model6": {
    #         "model_name": "BiLSTM_BN_ResBlock, smote",
    #         "model_path": "zzz_saved_model/Zhongda_BiLSTM_BN_Resnet_model_smote_FocalLoss_100_model_1.pth"
    #     },
    #     "model7": {
    #         "model_name": "BiLSTM_BN_ResBlock, smote",
    #         "model_path": "zzz_saved_model/Zhongda_BiLSTM_BN_Resnet_model_smote_FocalLoss_100_model_3.pth"
    #     }
    # }

    models_info_zyy = {
        "model1": {
            "model_name": "BiLSTM_BN, undersample, 52",
            "model_path": "ZYY/zzz_saved_model/ZYY_BiLSTM_BN_model_undersample_FocalLoss_100_0.01_model_52.pth"
        },
        "model2": {
            "model_name": "BiLSTM_BN, undersample, 57",
            "model_path": "ZYY/zzz_saved_model/ZYY_BiLSTM_BN_model_undersample_FocalLoss_100_0.01_model_57.pth"
        },
        "model3": {
            "model_name": "BiLSTM_BN_larger, undersample, 21",
            "model_path": "ZYY/zzz_saved_model/ZYY_BiLSTM_BN_larger_model_undersample_FocalLoss_100_0.01_model_21.pth"
        }
    }

    # selected_model = ['model2', 'model3', 'model4']
    # auc = voting(models_info, selected_model)
    # plot_auc(auc, selected_model)

    selected_model = [] # 默认全选
    auc_dict = voting(data_dir, models_info_zyy, selected_model, 512)
    # plot_auc(auc, selected_model)
