from glob import glob
import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt


def specificity_score(mask_label, mask_pred):
    true_negatives = ((mask_label == 0) & (mask_pred == 0)).sum()
    false_positives = ((mask_label == 1) & (mask_pred == 0)).sum()
    specificity_value = true_negatives / (true_negatives + false_positives)
    return specificity_value

def Negative_Predictive_Value_score(mask_label, mask_pred):
    true_negatives = ((mask_label == 0) & (mask_pred == 0)).sum()
    false_negatives = ((mask_label == 0) & (mask_pred == 1)).sum()
    NPV = true_negatives / (true_negatives + false_negatives)
    return NPV

def calculate_metrics(y_true, y_pred):
    # Ground truth
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    # Prediction
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    Accuracy = accuracy_score(y_true, y_pred)
    Sensitivity = recall_score(y_true, y_pred)
    Specificity = specificity_score(y_true, y_pred)
    False_Positive_rate = 1 - Specificity
    Positive_Predictive_Value = precision_score(y_true, y_pred)
    Negative_Predictive_Value = Negative_Predictive_Value_score(y_true, y_pred)
    return Accuracy, Sensitivity, Specificity, False_Positive_rate, Positive_Predictive_Value, Negative_Predictive_Value

# Load dataset
y_true = ""
y_pre = ""

# 获取所有PNG文件的文件列表
png_files_y1 = [f for f in os.listdir(y_true) if f.endswith('.png')]
png_files_y1.sort(key=lambda a: int(a.split("_")[-1].split(".")[0].split("mask")[1]))
png_files_y2 = [f for f in os.listdir(y_pre) if f.endswith('.png')]
#png_files_y2.sort(key=lambda a: int(a.split("_")[-1].split(".")[0]))
#png_files_y2.sort(key=lambda a: int(a.split("_")[-1].split(".")[0].split("segmentation")[1]))
png_files_y2.sort(key=lambda a: int(a.split("_")[-1].split(".")[0].split("mask")[1]))
acc_list=[]
se_list=[]
sp_list=[]
fp_rate_list=[]
ppv_list = []
npv_list = []
for png_x, png_y in zip(png_files_y1, png_files_y2):
    png_path_x = os.path.join(y_true, png_x)
    mask = cv2.imread(png_path_x, cv2.IMREAD_GRAYSCALE)
    png_path_y = os.path.join(y_pre , png_y)
    mask_pre = cv2.imread(png_path_y, cv2.IMREAD_GRAYSCALE)
    Accuracy, Sensitivity, Specificity, False_Positive_rate, Positive_Predictive_Value, Negative_Predictive_Value = (
        calculate_metrics(mask, mask_pre))
    acc_list.append(Accuracy)
    if Sensitivity!=0:
        se_list.append(Sensitivity)
    sp_list.append(Specificity)
    fp_rate_list.append(False_Positive_rate)
    if Positive_Predictive_Value!=0:
        ppv_list.append(Positive_Predictive_Value)
    npv_list.append(Negative_Predictive_Value)

acc = sum(acc_list) / len(acc_list)
se = sum(se_list) / len(se_list)
sp = sum(sp_list) / len(sp_list)
fp_rate = sum(fp_rate_list) / len(fp_rate_list)
ppv = sum(ppv_list) / len(ppv_list)
npv = sum(npv_list) /len(npv_list)
print(
        f"Accuracy: {acc:1.4f}, Sensitivity: {se:1.4f}, Specificity: {sp:1.4f}, False Positive rate: {fp_rate:1.4f}, "
        f"Positive Predictive Value: {ppv:1.4f}, npv: {npv:1.4f}")













