import json
import os

from function import calculate_metrics, plot_confusion_matrix
import numpy as np
import torch.optim
from sklearn.metrics import *
import matplotlib.pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import StepLR


class TrainModel:
    """
    训练网络的类
    """

    def __init__(self, my_model_name, model, hyperparameters, train_loader, valid_loader=None, test_loader=None,
                 optimizer_class=torch.optim.Adam, criterion_class=nn.BCELoss, scheduler_class=StepLR,
                 valid=True, save_model_index=None):
        """
        初始化训练
        :param my_model_name: 模型名称
        :param model: 定义好的模型结构，在model.py文件中，直接传入模型的类名，如BiLSTM_BN
        :param hyperparameters: 传入超参数，类型为字典
        :param train_loader: 训练集
        :param valid_loader: 验证集
        :param test_loader: 测试集
        :param optimizer_class: 优化器
        :param criterion_class: 损失函数
        :param scheduler_class: 学习率的策略
        :param valid: 如果为False，将跳过输出
        :param save_model_index: 训练效果较好的模型的Epoch（从1开始）
        """
        self.BATCH_SIZE = hyperparameters.get("BATCH_SIZE")
        self.EPOCH = hyperparameters.get("EPOCH")
        self.LR = hyperparameters.get("LEARNING_RATE")
        self.GAMMA = hyperparameters.get("GAMMA")
        self.STEP_SIZE = hyperparameters.get("STEP_SIZE")
        self.DECAY = hyperparameters.get("DECAY", 1e-6)
        self.DEVICE = hyperparameters.get("device")
        self.SEED = hyperparameters.get("SEED")
        self.ALPHA_LOSS = hyperparameters.get("ALPHA_LOSS")
        self.GAMMA_LOSS = hyperparameters.get("GAMMA_LOSS")

        self.model_name = my_model_name
        self.model = model()
        self.train_dataloader = train_loader
        self.valid_dataloader = valid_loader
        self.test_dataloader = test_loader
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.LR, weight_decay=self.DECAY)
        self.criterion = criterion_class
        self.scheduler = scheduler_class(self.optimizer, step_size=self.STEP_SIZE, gamma=self.GAMMA)
        self.valid = valid
        self.save_model_index = save_model_index
        self.model.to(self.DEVICE)
        self.saved_num = 0

    def train(self):
        """
        训练主函数
        :return: info.json
        """
        train_total_loss = []
        train_loss_list = []
        val_loss_list = []
        accuracy_list_auc = []
        specificity_list_auc = []
        alarm_sen_list_auc = []
        alarm_acc_list_auc = []
        accuracy_list_prc = []
        specificity_list_prc = []
        alarm_sen_list_prc = []
        alarm_acc_list_prc = []
        auc_list = []
        prc_list = []
        brier_list = []
        info = {}

        for epoch in range(self.EPOCH):
            print(f"---------------------------------------"
                  f"Epoch: {epoch + 1}"
                  f"---------------------------------------")
            train_loss = self.train_one_epoch()
            train_total_loss.extend(train_loss)
            train_loss_list.append(train_loss[-1])
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss[-1]:.4f}')

            model_directory = f"./{self.model_name}/"
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            if self.save_model_index and (epoch + 1) in self.save_model_index:
                torch.save(self.model, f'zzz_saved_model/{self.model_name}_model_{epoch + 1}.pth')
                print(f'Model has been saved into: zzz_saved_model/{self.model_name}_model_{epoch}')
                self.saved_num += 1
                if self.saved_num == len(self.save_model_index):
                    break

            if self.valid:
                (loss, accuracy_auc, specificity_auc, alarm_sen_auc, alarm_acc_auc,
                 accuracy_prc, specificity_prc, alarm_sen_prc, alarm_acc_prc, auc, prc, brier) = self.validate(epoch)
                val_loss_list.append(loss)
                accuracy_list_auc.append(accuracy_auc)
                specificity_list_auc.append(specificity_auc)
                alarm_sen_list_auc.append(alarm_sen_auc)
                alarm_acc_list_auc.append(alarm_acc_auc)
                accuracy_list_prc.append(accuracy_prc)
                specificity_list_prc.append(specificity_prc)
                alarm_sen_list_prc.append(alarm_sen_prc)
                alarm_acc_list_prc.append(alarm_acc_prc)
                auc_list.append(auc)
                prc_list.append(prc)
                brier_list.append(brier)

                info = {
                    "train_total_loss": train_total_loss,
                    "train_loss_list": train_loss_list,
                    "val_loss_list": val_loss_list,
                    "accuracy_list_auc": accuracy_list_auc,
                    "specificity_list_auc": specificity_list_auc,
                    "alarm_sen_list_auc": alarm_sen_list_auc,
                    "alarm_acc_list_auc": alarm_acc_list_auc,
                    "accuracy_list_prc": accuracy_list_prc,
                    "specificity_list_prc": specificity_list_prc,
                    "alarm_sen_list_prc": alarm_sen_list_prc,
                    "alarm_acc_list_prc": alarm_acc_list_prc,
                    "roc_auc_list": auc_list,
                    "prc_auc_list": prc_list,
                    "brier_list": brier_list
                }
                # Save hyperparameters
                hyperparameters = {
                    "BATCH_SIZE": self.BATCH_SIZE,
                    "EPOCH": self.EPOCH,
                    "LEARNING_RATE": self.LR,
                    "GAMMA": self.GAMMA,
                    "STEP_SIZE": self.STEP_SIZE,
                    "device": self.DEVICE,
                    "SEED": self.SEED,
                    "ALPHA_LOSS": self.ALPHA_LOSS,
                    "GAMMA_LOSS": self.GAMMA_LOSS
                }
                with open(f'./{self.model_name}/00_hyperparameters.json', 'w') as json_file:
                    json.dump(hyperparameters, json_file, indent=4)
                with open(f'./{self.model_name}/00_info.json', 'w') as json_file:
                    json.dump(info, json_file, indent=4)

            self.scheduler.step()

        return info

    def save_model(self):
        """
        保存终末状态模型
        :return: None
        """
        torch.save(self.model, f"{self.model_name}/00_{self.model_name}_final_model.pth")
        return None

    def train_one_epoch(self):
        """
        每个EPOCH的训练函数
        :return: 每个Epoch的损失列表
        """
        one_epoch_loss = []
        epoch_train_step = 0

        self.model.train()
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(inputs.float())
            loss = self.criterion(outputs, labels.float())
            # print(loss.item())
            loss.backward()
            self.optimizer.step()

            epoch_train_step += 1
            if epoch_train_step % 100 == 0:
                print(f'Train iter:{epoch_train_step}, Loss: {loss.item()}, Device: {self.DEVICE}')

            one_epoch_loss.append(loss.item())

        return one_epoch_loss

    def validate(self, epoch):
        """
        计算每一个epoch结束的模型性能
        :param epoch: 当前Epoch
        :return: valid_loss, valid_accuracy, valid_specificity, valid_alarm_sen, valid_alarm_acc, valid_auc
        """
        self.model.eval()
        self.model.to(self.DEVICE)
        eps = 1e-6
        total_valid_loss = eps
        true_labels = []
        predicted_probs = []
        count = 0

        with torch.no_grad():
            for inputs, targets in self.valid_dataloader:
                count += 1
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                outputs = self.model(inputs.float())
                true_labels.append(targets.cpu().numpy())
                predicted_probs.append(outputs.cpu().numpy())
                loss = self.criterion(outputs, targets.float())
                total_valid_loss += loss.item()

            valid_loss = total_valid_loss / count
            print("整体测试集上的Loss: {}".format(valid_loss))
            true_labels_flat = np.concatenate(true_labels)
            predicted_probs_flat = np.concatenate(predicted_probs)

            brier_score = np.mean((predicted_probs_flat - true_labels_flat) ** 2)

            valid_auc, best_threshold_auc = self._plot_roc_curve(true_labels_flat, predicted_probs_flat, epoch)
            valid_prc, best_threshold_prc = self._plot_prc_curve(true_labels_flat, predicted_probs_flat, epoch)
            valid_accuracy_auc, valid_specificity_auc, valid_alarm_sen_auc, valid_alarm_acc_auc = self._calculate_criterion(
                true_labels_flat, predicted_probs_flat, best_threshold_auc, epoch, "auc")
            valid_accuracy_prc, valid_specificity_prc, valid_alarm_sen_prc, valid_alarm_acc_prc = self._calculate_criterion(
                true_labels_flat, predicted_probs_flat, best_threshold_prc, epoch, "prc")

        return (valid_loss, valid_accuracy_auc, valid_specificity_auc, valid_alarm_sen_auc, valid_alarm_acc_auc,
                valid_accuracy_prc, valid_specificity_prc, valid_alarm_sen_prc, valid_alarm_acc_prc,
                valid_auc, valid_prc, brier_score)

    def _plot_roc_curve(self, true_labels_flat, predicted_probs_flat, epoch):
        fpr, tpr, thresholds = roc_curve(true_labels_flat, predicted_probs_flat)
        valid_auc = auc(fpr, tpr)
        best_threshold_index = (tpr - fpr).argmax()
        best_threshold = thresholds[best_threshold_index]

        print(f"AUROC: {valid_auc:.4f}")
        print(f"Best threshold: {best_threshold:.4f}")
        # 绘制ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {valid_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - Epoch {epoch + 1}')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.model_name}/{self.model_name}_ROC_EPOCH_{epoch + 1}.png")
        plt.close()

        return valid_auc, best_threshold

    def _plot_prc_curve(self, true_labels_flat, predicted_probs_flat, epoch):
        precision, recall, thresholds = precision_recall_curve(true_labels_flat, predicted_probs_flat)
        prc_auc = auc(recall, precision)
        best_threshold_index = (precision * recall / (precision + recall + 1e-6)).argmax()
        best_threshold = thresholds[best_threshold_index]
        print(f"AUPRC: {prc_auc:.4f}")
        print(f"Best threshold: {best_threshold:.4f}")

        plt.figure()
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PRC curve (area = {prc_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Epoch {epoch + 1}')
        plt.legend(loc='lower left')
        plt.savefig(f"{self.model_name}/{self.model_name}_PRC_EPOCH_{epoch + 1}.png")
        plt.grid()
        plt.close()

        return prc_auc, best_threshold

    def _calculate_criterion(self, true_labels_flat, predicted_probs_flat, best_threshold, epoch, name):
        cm, valid_specificity, valid_alarm_sen, valid_alarm_acc, valid_accuracy = calculate_metrics(
            true_labels_flat, predicted_probs_flat, best_threshold)

        print(name)
        print("Confusion Matrix:")
        print(cm)
        print(f"Specificity: {valid_specificity:.4f}")
        print(f"Sensitivity: {valid_alarm_sen:.4f}")
        print(f"Alarm Accuracy: {valid_alarm_acc:.4f}")
        print(f"Accuracy: {valid_accuracy:.4f}")

        # 绘制混淆矩阵
        plot_confusion_matrix(self.model_name, name, epoch, cm, classes=['Survive', 'Death'])

        return valid_accuracy, valid_specificity, valid_alarm_sen, valid_alarm_acc
