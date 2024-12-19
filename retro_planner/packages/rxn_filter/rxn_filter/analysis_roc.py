import numpy as np
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import auc


# ROC空间将伪阳性率（FPR）定义为 X 轴，真阳性率（TPR）定义为 Y 轴。
#
# TPR：在所有实际为阳性的样本中，被正确地判断为阳性之比率。
# {\displaystyle TPR=TP/(TP+FN)}{\displaystyle TPR=TP/(TP+FN)}
# FPR：在所有实际为阴性的样本中，被错误地判断为阳性之比率。
# {\displaystyle FPR=FP/(FP+TN)}{\displaystyle FPR=FP/(FP+TN)}
# 给定一个二元分类模型和它的阈值，就能从所有样本的（阳性／阴性）真实值和预测值计算出一个 (X=FPR, Y=TPR) 坐标点。
def roc_curve(y_true, y_prob, thresholds):
    fpr = []
    tpr = []

    for threshold in thresholds:
        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]


if __name__ == '__main__':
    predict_score_dic = torch.load(
        './data/predict_score_dic_filter_train_data_random_gen_aizynth_filter_dataset_2021-12-22_12h-15m-45s.pkl')

    y_true_epoch = predict_score_dic['y_true_epoch']
    y_score_epoch = predict_score_dic['y_score_epoch']
    y_pred_epoch = (y_score_epoch >= 0.5)
    thresholds = [i * 0.01 for i in range(0, 101)]
    fpr, tpr = roc_curve(y_true_epoch, y_score_epoch, thresholds)
    print(metrics.classification_report(y_true_epoch, y_pred_epoch, digits=4))
    print(auc(fpr, tpr))
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC')
    plt.plot(thresholds, thresholds, linestyle=':', color='red', label='')
    # plt.legend()
    plt.title(u'ROC/AUC: {:.5f}'.format(auc(fpr, tpr)))
    plt.xlabel(u'FPR')
    plt.ylabel(u'TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(fname="./data/roc_fix.svg", format="svg")
    plt.show()
