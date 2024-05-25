import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss
        :param alpha: 平衡因子
        :param gamma: 调整因子
        :param reduction: 指定应用于输出的减少方式: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        前向传播计算损失
        :param inputs: 预测值 (logits)
        :param targets: 实际标签
        :return: 计算后的 Focal Loss
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 计算 pt
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# 示例用法
if __name__ == "__main__":
    alpha=1
    gamma=4
    loss_fn = FocalLoss(alpha, gamma, reduction='mean')
    inputs = torch.tensor([[0.1, 0.9], [0.9, 0.1]], dtype=torch.float32)  # 预测值 (logits)
    targets = torch.tensor([[0, 1], [0, 1]], dtype=torch.float32)  # 实际标签

    loss = loss_fn(inputs, targets)
    print("Focal Loss (mean reduction):", loss.item())

    loss_fn_none = FocalLoss(alpha, gamma, reduction='none')
    loss_none = loss_fn_none(inputs, targets)
    print("Focal Loss (none reduction):", loss_none)

    loss_fn_sum = FocalLoss(alpha, gamma, reduction='sum')
    loss_sum = loss_fn_sum(inputs, targets)
    print("Focal Loss (sum reduction):", loss_sum.item())

    loss_bce = nn.BCELoss()
    loss = loss_bce(inputs, targets)
    print("Focal Loss (sum reduction):", loss.item())
