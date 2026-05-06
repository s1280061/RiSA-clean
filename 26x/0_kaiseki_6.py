import pandas as pd, numpy as np

CSV = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p20_f45_s5_norm360x240_with_speed_train.csv"
H_PAST = 30  # ←あなたの実際の過去長に合わせて。20なら20

df = pd.read_csv(CSV, nrows=20000)  # 十分
cols = [f"past_speed{i+1}" for i in range(H_PAST)]
s = df[cols].to_numpy().astype(np.float32).ravel()

print("speed stats:")
print(" min =", float(np.nanmin(s)))
print(" max =", float(np.nanmax(s)))
print(" mean=", float(np.nanmean(s)))
print(" std =", float(np.nanstd(s)))
