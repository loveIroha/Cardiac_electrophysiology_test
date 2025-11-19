import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件（请替换为你的实际文件路径）
df = pd.read_csv('/mnt/large2/gjh/Small_tissue_ele/build1/gpb_CN_output.csv')

# 提取x轴（时间）和y轴（V值）数据
x = df['time_ms']
y = df['Vm_mV']

# 创建图形并设置大小
plt.figure(figsize=(10, 6))

# 绘制曲线
plt.plot(x, y, marker='o', linestyle='-', color='b', label='V (stats)')

# 添加标题和坐标轴标签
plt.title('Time vs V (stats)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('V (stats)', fontsize=12)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend()

# 调整布局，避免标签被截断
plt.tight_layout()

# 保存为PNG文件（分辨率300dpi，可根据需要调整）
# 文件名可自定义，如'time_voltage_plot.png'
plt.savefig('time_vs_v_cell.png', dpi=300, bbox_inches='tight')
