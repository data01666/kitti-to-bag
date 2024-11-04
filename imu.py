import datetime as dt
import os
import numpy as np
import matplotlib.pyplot as plt

data_path = r"D:\study\dataset\kitti\raw_data\00\2011_10_03\2011_10_03_drive_0027_sync"  # 修改为自己的文件路径

def load_timestamps(data='oxts'):
    """Load timestamps from file to detect discontinuities and reverse order issues."""
    timestamp_file = os.path.join(data_path, data, 'timestamps.txt')

    # Read and parse the timestamps
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f.readlines():
            # 截断到微秒精度
            t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            t = t.timestamp()
            timestamps.append(t)

    return np.array(timestamps)

# 加载时间戳
timestamps = load_timestamps()

# 计算相邻时间差
time_diffs = np.diff(timestamps)
x = np.arange(1, len(timestamps))  # 差值的x轴索引应从1开始

# 设置检测阈值
max_gap = 0.015  # 断续阈值（秒）

# 检测逆序和断续
reverse_order_indices = np.where(time_diffs < 0)[0]  # 检测逆序
discontinuity_indices = np.where(time_diffs > max_gap)[0]  # 检测断续

# 输出检测结果
print("逆序问题检测（负时间差）:")
for idx in reverse_order_indices:
    print(f"Index {idx} -> {timestamps[idx]} 与 Index {idx+1} -> {timestamps[idx+1]}")

print("\n断续问题检测（大于阈值的时间差）:")
for idx in discontinuity_indices:
    print(f"Index {idx} -> {timestamps[idx]} 与 Index {idx+1} -> {timestamps[idx+1]}，差值: {time_diffs[idx]} 秒")

# 绘制异常时间差点
plt.figure(figsize=(12, 6))
# 绘制逆序的点
plt.scatter(x[reverse_order_indices], time_diffs[reverse_order_indices], color='red', label='Reverse Order (< 0)')
# 绘制断续的点
plt.scatter(x[discontinuity_indices], time_diffs[discontinuity_indices], color='green', label=f'Discontinuity (> {max_gap} sec)')
plt.axhline(y=0, color='r', linestyle='-', label='Reverse Threshold (0)')
plt.axhline(y=max_gap, color='g', linestyle='-', label=f'Discontinuity Threshold ({max_gap} sec)')
plt.xlabel('Index')
plt.ylabel('Time Difference (sec)')
plt.legend()
plt.title('Detected Discontinuities and Reverse Orders in Time Differences')
plt.show()
