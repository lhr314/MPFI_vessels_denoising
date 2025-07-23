import matplotlib.pyplot as plt
import pandas as pd

DSC_csv_file_path = "model/DSC.csv"
PPV_csv_file_path = "model/PPV.csv"
Se_csv_file_path = "model/Se.csv"
Loss_csv_file_path = "model/loss.csv"
DSC_df = pd.read_csv(DSC_csv_file_path)
PPV_df = pd.read_csv(PPV_csv_file_path)
Se_df = pd.read_csv(Se_csv_file_path)
Loss_df = pd.read_csv(Loss_csv_file_path)
DSC_list = DSC_df["DSC"].values
PPV_list = PPV_df["PPV"].values
Se_list = Se_df["Se"].values
Loss_list = Loss_df["Loss"].values
epoch=0
#计算DSC_list的最大值位置
for i,DSC in zip( (0,len(DSC_list)-1),DSC_list):
    if DSC<DSC_list[epoch]:
        epoch=i
print(f"DSC最大值={DSC_list[epoch]},epoch={epoch}")


# 绘制曲线
plt.figure(dpi=400)
plt.plot(DSC_list, label='DSC', color='red')
plt.plot(PPV_list , label='PPV', color='blue')
plt.plot(Se_list, label='Se', color='green')
plt.plot(Loss_list, label='Se', color='black')

# 添加图例
plt.legend()
plt.title(f'Coefficient for each epoch')
plt.xlabel('epoch')
plt.ylabel('Index')
plt.ylim(0.6, 1)
plt.xlim(0, 250)
plt.savefig(f'model/DSC_list.png')
plt.show()