import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import chardet

# 检测文件编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# 读取CSV文件
df1 = pd.read_csv('ROI分析图/origin_Values.csv',encoding=detect_encoding('ROI分析图/origin_Values.csv'))
df2 = pd.read_csv('ROI分析图/wavelet_Values.csv',encoding=detect_encoding('ROI分析图/wavelet_Values.csv'))
df3 = pd.read_csv('ROI分析图/adaptive_Values.csv',encoding=detect_encoding('ROI分析图/adaptive_Values.csv'))
df4 = pd.read_csv('ROI分析图/AI_Values.csv',encoding=detect_encoding('ROI分析图/AI_Values.csv'))


# 提取 Distance_(μm) 和 Gray_Value 列
x1 = df1.iloc[1:, 0].values.astype(float)
y1 = df1.iloc[1:, 1].values.astype(float)
x2 = df2.iloc[1:, 0].values.astype(float)
y2 = df2.iloc[1:, 1].values.astype(float)
x3 = df3.iloc[1:, 0].values.astype(float)
y3 = df3.iloc[1:, 1].values.astype(float)
x4 = df4.iloc[1:, 0].values.astype(float)
y4 = df4.iloc[1:, 1].values.astype(float)

max_list = [max(y1), max(y2), max(y3), max(y4)]
max_value = max(max_list)


# 归一化 y 数据
y1_normalized = y1 / max_value
y2_normalized = y2 / max_value
y3_normalized = y3 / max_value
y4_normalized = y4 / max_value
# 绘制函数图
plt.figure(dpi=300)
plt.plot(x1, y1_normalized,label='Original images', color='gray',linewidth=2)
plt.plot(x2, y2_normalized,label='Wavelet denoising',color='lightblue',linewidth=2)
plt.plot(x3, y3_normalized,label='Non-Local Means denoising',color='yellow',linewidth=2)
plt.plot(x4, y4_normalized,label='Segmented and denoise images',color='orange',linewidth=2)


# 添加图例
plt.legend()
plt.ylim(0,1.2)
plt.xlabel('Distance (μm)',fontweight='bold')
plt.ylabel('Normalized intensity',fontweight='bold')
plt.show()