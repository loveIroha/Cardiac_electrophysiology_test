import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('/mnt/large2/gjh/Small_tissue_ele/Vm_tissue.csv')

# 提取x轴(时间)和y轴(sum(u)值)数据
x = df['Time']
y = df['sum(u)'] * 1000  # 乘以1000转换单位为mV

# 创建图形并设置大小
plt.figure(figsize=(12, 6))

# 绘制曲线
plt.plot(x, y, linestyle='-', color='b', linewidth=2, label='Potential')

# 添加标题和坐标轴标签
plt.title('Time vs Potential', fontsize=16, fontweight='bold')
plt.xlabel('Time t (ms)', fontsize=14)
plt.ylabel('Potential (mV)', fontsize=14)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5)

# 添加图例
plt.legend(fontsize=12)

# 调整布局,避免标签被截断
plt.tight_layout()

# 保存为PNG文件(分辨率300dpi)
plt.savefig('/mnt/large2/gjh/Small_tissue_ele/time_vs_sum_u.png', dpi=300, bbox_inches='tight')

print("图像已保存为: time_vs_sum_u.png")
print(f"数据点数量: {len(x)}")
print(f"时间范围: {x.min()} - {x.max()}")
print(f"sum(u) 范围: {y.min():.6f} - {y.max():.6f}")

# 显示图形(可选)
# plt.show()
