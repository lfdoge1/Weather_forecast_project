import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ 读取 CSV 数据
file_path = r"C:\Users\Glen Lee\Downloads\uk_tot_samsungtv-wired.csv"  # 正确的本地路径
df = pd.read_csv(file_path)

# 2️⃣ 确保数据列数足够
if df.shape[1] < 5:
    print("错误: 数据列数不足，无法绘制 Ports 和 IP 地址")
else:
    # 3️⃣ 选择第 4 列（索引 3）和第 5 列（索引 4）
    ports = df.iloc[:, 3]  # 端口
    ip_addresses = df.iloc[:, 4]  # IP 地址

    # 4️⃣ 创建图形giut
    plt.figure(figsize=(10, 5))

    # 📌 端口分布
    plt.subplot(1, 2, 1)
    plt.hist(ports, bins=20, alpha=0.7, edgecolor="black")
    plt.xlabel("Ports")
    plt.ylabel("Frequency")
    plt.title("Ports Distribution")

    # 📌 IP 地址分布
    plt.subplot(1, 2, 2)
    plt.hist(ip_addresses, bins=20, alpha=0.7, edgecolor="black")
    plt.xlabel("IP Addresses")
    plt.ylabel("Frequency")
    plt.title("IP Addresses Distribution")

    # 5️⃣ 显示图形
    plt.tight_layout()
    plt.show()
    dips = df.iloc[:, 5]
