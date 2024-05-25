import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from hyperparameters import *
from sampled import BalancedData


def plot_confusion_matrix(model_name, idx, cm, classes, normalize=False, title='混淆矩阵', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # # 设置中文字体
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # 设置其他字体属性，如字号
    plt.rcParams.update({'font.size': 12})
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("归一化混淆矩阵")
    else:
        print('混淆矩阵，未归一化')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "red")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"{model_name}/{model_name}_CM_EPOCH_{idx+1}.png")
    plt.close()


def calculate_metrics(true_labels_flat, predicted_probs_flat, best_threshold):
    # 计算混淆矩阵
    TP = np.sum((predicted_probs_flat > best_threshold) & (true_labels_flat == 1))
    FP = np.sum((predicted_probs_flat > best_threshold) & (true_labels_flat == 0))
    TN = np.sum((predicted_probs_flat <= best_threshold) & (true_labels_flat == 0))
    FN = np.sum((predicted_probs_flat <= best_threshold) & (true_labels_flat == 1))

    confusion_matrix = np.array([[TN, FP], [FN, TP]])

    # 计算特异性、敏感性和准确率
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    alarm_accuracy = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return confusion_matrix, specificity, sensitivity, alarm_accuracy, accuracy


def main_data_loader(data_dir, sample_method):
    """
    加载数据
    :param data_dir: 数据位置。如：data_label_1/data_tensor_24.pth
    :param sample_method: 采样方式。如："undersample" or "oversample" or "smote"
    :return: train_dataloader_f, val_dataloader_f, test_dataloader_f
    """
    data = torch.load(data_dir)

    data_train = data['data_tensor_train']
    label_train = data['label_tensor_train']
    data_val = data['data_tensor_val']
    label_val = data['label_tensor_val']
    data_test = data['data_tensor_test']
    label_test = data['label_tensor_test']

    balancer = BalancedData(data_train, label_train)
    data_train_b, label_train_b = balancer.sample(method=sample_method)

    dataset_train = TensorDataset(data_train_b, label_train_b)
    dataset_val = TensorDataset(data_val, label_val)
    dataset_test = TensorDataset(data_test, label_test)

    # 利用 DataLoader 来加载数据集
    train_dataloader_f = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader_f = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader_f = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader_f, val_dataloader_f, test_dataloader_f


def plot_info(info, model_name):
    epochs = range(1, len(info['train_loss_list']) + 1)

    plt.figure()
    plt.plot(range(len(info['train_total_loss'])), info['train_total_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Total Train Loss')
    plt.savefig(f'{model_name}/01_total_loss.png')
    plt.close()

    # 绘制训练和验证损失
    plt.figure()
    plt.plot(epochs, info['train_loss_list'], label='Train Loss')
    plt.plot(epochs, info['val_loss_list'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.savefig(f'{model_name}/02_loss_curve.png')
    plt.close()

    # 绘制各个指标
    plt.figure()
    plt.plot(epochs, info['accuracy_list_auc'], label='Accuracy')
    plt.plot(epochs, info['specificity_list_auc'], label='Specificity')
    plt.plot(epochs, info['alarm_sen_list_auc'], label='Alarm Sensitivity')
    plt.plot(epochs, info['alarm_acc_list_auc'], label='Alarm Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Percent')
    plt.title('Model Performance when best AUC')
    plt.legend()
    plt.savefig(f'{model_name}/03_model_performance_auc.png')
    plt.close()

    # 绘制各个指标
    plt.figure()
    plt.plot(epochs, info['accuracy_list_prc'], label='Accuracy')
    plt.plot(epochs, info['specificity_list_prc'], label='Specificity')
    plt.plot(epochs, info['alarm_sen_list_prc'], label='Alarm Sensitivity')
    plt.plot(epochs, info['alarm_acc_list_prc'], label='Alarm Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Percent')
    plt.title('Model Performance when beat PRC')
    plt.legend()
    plt.savefig(f'{model_name}/04_model_performance_prc.png')
    plt.close()

    # 绘制ROC AUC
    plt.figure()
    plt.plot(epochs, info['roc_auc_list'], label='ROC AUC')
    plt.plot(epochs, info['prc_auc_list'], label='PRC AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('AUC over Epochs')
    plt.legend()
    plt.savefig(f'{model_name}/05_auc_curve.png')
    plt.show()
