import matplotlib.pyplot as plt
import pandas as pd

Dice_csv_file_path = "model_metrics/Dice.csv"
Loss_csv_file_path = "model_metrics/loss.csv"
Dice_df = pd.read_csv(Dice_csv_file_path)
Loss_df = pd.read_csv(Loss_csv_file_path)
Dice_list = Dice_df["Dice"].values
Loss_list = Loss_df["Loss"].values
max_epoch=0
i=0
#计算DSC_list的最大值位置
for Dice in Dice_list:
    if Dice>Dice_list[max_epoch]:
        max_epoch=i
    i+=1
print(f"Dice最大值={Dice_list[max_epoch]},epoch={max_epoch}")


# 绘制曲线
plt.figure(dpi=400)
plt.plot(Dice_list, label='Dice', color='black')

# 添加图例
plt.legend()
plt.title(f'Average Dice in test set for each epoch')
plt.xlabel('epoch')
plt.ylabel('Value of Index')
plt.ylim(0.75, 0.85)
plt.xlim(0, 250)
plt.savefig(f'model_metrics/Dice_list.png')

# plt.plot(Loss_list, label='Dice Loss', color='black')
# plt.title(f'Average Dice Loss for each epoch')
# plt.xlabel('epoch')
# plt.ylabel('Value of Index')
# plt.ylim(0, 1)
# plt.xlim(0, 250)
# plt.savefig(f'model_metrics/Dice_Loss_list.png')