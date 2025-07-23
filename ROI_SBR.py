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
df1 = pd.read_csv('1660微米处SBR对比/data/ori.csv',encoding=detect_encoding('1660微米处SBR对比/data/ori.csv'))
df2 = pd.read_csv('1660微米处SBR对比/data/segmented and denoise.csv',encoding=detect_encoding('1660微米处SBR对比/data/segmented and denoise.csv'))

# 提取 Distance_(μm) 和 Gray_Value 列
x1 = df1.iloc[1:, 0].values.astype(float)
y1 = df1.iloc[1:, 1].values.astype(float)
x2 = df2.iloc[1:, 0].values.astype(float)
y2 = df2.iloc[1:, 1].values.astype(float)
# 归一化 y 数据
y1_normalized = y1 / 255.0
y2_normalized = y2 / 255.0

# 绘制函数图
plt.figure(dpi=300)
plt.plot(x1, y1_normalized,label='original images', color='gray',linewidth=2)
plt.plot(x2, y2_normalized ,label='segmented and denoise images', color='orange',linewidth=2)
# 添加图例
plt.legend()
plt.ylim(0,1)
plt.xlabel('Distance_(μm)',fontweight='bold')
plt.ylabel('Normalized intensity',fontweight='bold')
plt.show()