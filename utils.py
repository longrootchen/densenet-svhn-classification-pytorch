import os
import numpy as np
from matplotlib import pyplot as plt


class AverageMeter:

    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Evaluator:

    def __init__(self, num_classes):
        """
        Params:
            num_classes (int): 分类类别总数
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)

    def accuracy(self):
        """计算准确率"""
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    def error(self):
        """计算错误率"""
        return 1 - self.accuracy()

    def class_accuracy(self):
        """计算类准确率"""
        return np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)

    def mean_class_accuracy(self):
        """计算平均类准确率"""
        return np.nanmean(self.class_accuracy())

    def class_precision(self):
        """计算类精确率"""
        return np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)

    def mean_class_precision(self):
        """计算平均精确率"""
        return np.nanmean(self.class_precision())

    def show_matrix(self, cls_to_idx, save_matrix=False):
        """
        可视化混淆矩阵
        Params:
            class_to_idx (dict): original class names and corresponding index for label
            save_matrix (bool): whether to save the confusion matrix
        """
        cls_to_idx = sorted(cls_to_idx.items(), key=lambda x: x[1])
        classes = [pair[0] for pair in cls_to_idx]

        # 归一化
        norm_conf_mat = np.zeros(self.confusion_matrix.shape)
        for i in range(self.num_classes):
            norm_conf_mat[i, :] = self.confusion_matrix[i, :] / self.confusion_matrix[i, :].sum()

        # 设置颜色
        cmap = plt.cm.get_cmap('pink')  # reference: http://matplotlib.org/examples/color/colormaps_reference.html
        plt.imshow(norm_conf_mat, cmap=cmap)
        plt.colorbar()

        # 设置文字
        xlocations = np.array(range(self.num_classes))
        plt.xticks(xlocations, classes, rotation=60)
        plt.yticks(xlocations, classes, rotation=60)
        plt.xlabel('Predict Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        # 显示数值
        for i in range(norm_conf_mat.shape[0]):
            for j in range(norm_conf_mat.shape[1]):
                plt.text(x=j, y=i, s=int(self.confusion_matrix[i, j]),
                         va='center', ha='center', color='red', fontsize=10)

        # 保存
        if save_matrix:
            save_path = os.path.join(os.curdir, 'confusion_matrix.pdf')
            plt.savefig(save_path)

        # 显示
        plt.show()
        plt.close()

    def update_matrix(self, trues, preds):
        """
        传入一个 batch 的真实标签和预测标签，更新混淆矩阵
        Params:
            trues (numpy.array): shape (batch_size, num_classes), 真实标签
            preds (numpy.array): shape (batch_size, num_classes), 预测标签
        """
        assert trues.shape == preds.shape
        self.confusion_matrix += self._generate_matrix(trues, preds)

    def _generate_matrix(self, trues, preds):
        """
        对于一个 batch 的真实标签和预测标签, 生成混淆矩阵
        Params:
            trues (numpy.array): shape (batch_size, num_classes), 真实标签
            preds (numpy.array): shape (batch_size, num_classes), 预测标签
        Returns:
            conf_mat (numpy.array): shape (num_classes, num_classes), 混淆矩阵, 行为真实, 列为预测
        """
        mask = (trues >= 0) & (trues < self.num_classes)
        conf_mat = np.bincount(self.num_classes * trues[mask].astype(int) + preds[mask],
                               minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return conf_mat


if __name__ == '__main__':
    import torch

    evaluator = Evaluator(num_classes=4)

    outputs = torch.tensor([
        [0.3, 0.1, 0.1, 0.2, 0.1, 0.2],
        [0.1, 0.2, 0.4, 0.1, 0.1, 0.1],
        [0.2, 0.29, 0.09, 0.31, 0.11, 0.2]
    ])
    targets = torch.tensor([0, 1, 2])

    trues = targets.numpy()
    preds = outputs.max(dim=1)[1].numpy()
    evaluator.update_matrix(trues, preds)
    print(evaluator.accuracy())
    print(evaluator.error())
    print(evaluator.class_accuracy())
    print(evaluator.mean_class_accuracy())
    print(evaluator.class_precision())
    print(evaluator.mean_class_precision())

    cls_to_idx = {'cat': 0, 'dog': 1, 'horse': 2, 'frog': 3}
    evaluator.show_matrix(cls_to_idx)
