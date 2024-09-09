import pandas as pd
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('sample', type=float)
parser.add_argument('seednum', type=int)
args = parser.parse_args()

df_list = []

for i in range(1, args.seednum + 1):
    path = f"sheridan-cs\\{args.dataset} {args.sample:.2f}\\{args.dataset} {args.sample:.2f} {i}.csv"
    one_df = pd.read_csv(path)
    df_list.append(one_df)

df = pd.concat(df_list).groupby(level=0).mean()

fdp_nominals = df['fdp_nominals']

powers_15_tv = df['powers_15_tv']
fdps_15_tv = df['fdps_15_tv']
pcers_15_tv = df['pcers_15_tv']

powers_15_rb = df['powers_15_rb']
fdps_15_rb = df['fdps_15_rb']
pcers_15_rb = df['pcers_15_rb']

powers_15_rp = df['powers_15_rp']
fdps_15_rp = df['fdps_15_rp']
pcers_15_rp = df['pcers_15_rp']

powers_cs = df['powers_cs']
fdps_cs = df['fdps_cs']
pcers_cs = df['pcers_cs']

# plot treevar

fig, axs = plt.subplots(ncols=3, figsize=(20, 6))

fig.suptitle(f"Power, FDP and PCER vs nominal FDP level in Sheridan (2015), tree variance, {args.dataset} dataset, averaged over {args.seednum} times")
axs[0].plot(fdp_nominals, powers_15_tv)
axs[0].set_xlabel("nominal FDP level")
axs[0].set_ylabel("power")
axs[1].plot(fdp_nominals, fdps_15_tv)
axs[1].set_xlabel("nominal FDP level")
axs[1].set_ylabel("FDP")
axs[2].plot(fdp_nominals, pcers_15_tv)
axs[1].plot([0, 1], [0, 1], color='green', linestyle='-.', alpha=0.5)
axs[2].set_xlabel("nominal FDP level")
axs[2].set_ylabel("PCER")
plt.savefig(f'sheridan-cs-pic\\{args.dataset} {args.sample:.2f}\\Power, FDP and PCER (15, treevar).png')

# plot RMSE bin

fig, axs = plt.subplots(ncols=3, figsize=(20, 6))
fig.suptitle(f"Power, FDP and PCER vs nominal FDP level in Sheridan (2015), RMSE binning, {args.dataset} dataset, averaged over {args.seednum} times")
axs[0].plot(fdp_nominals, powers_15_rb)
axs[0].set_xlabel("nominal FDP level")
axs[0].set_ylabel("power")
axs[1].plot(fdp_nominals, fdps_15_rb)
axs[1].set_xlabel("nominal FDP level")
axs[1].set_ylabel("FDP")
axs[2].plot(fdp_nominals, pcers_15_rb)
axs[1].plot([0, 1], [0, 1], color='green', linestyle='-.', alpha=0.5)
axs[2].set_xlabel("nominal FDP level")
axs[2].set_ylabel("PCER")
plt.savefig(f'sheridan-cs-pic\\{args.dataset} {args.sample:.2f}\\Power, FDP and PCER (15, RMSEbin).png')

# plot RMSE pred

fig, axs = plt.subplots(ncols=3, figsize=(20, 6))
fig.suptitle(f"Power, FDP and PCER vs nominal FDP level in Sheridan (2015), RMSE prediction, {args.dataset} dataset, averaged over {args.seednum} times")
axs[0].plot(fdp_nominals, powers_15_rp)
axs[0].set_xlabel("nominal FDP level")
axs[0].set_ylabel("power")
axs[1].plot(fdp_nominals, fdps_15_rp)
axs[1].set_xlabel("nominal FDP level")
axs[1].set_ylabel("FDP")
axs[2].plot(fdp_nominals, pcers_15_rp)
axs[1].plot([0, 1], [0, 1], color='green', linestyle='-.', alpha=0.5)
axs[2].set_xlabel("nominal FDP level")
axs[2].set_ylabel("PCER")
plt.savefig(f'sheridan-cs-pic\\{args.dataset} {args.sample:.2f}\\Power, FDP and PCER (15, RMSEpred).png')

# plot conformal

fig, axs = plt.subplots(ncols=3, figsize=(20, 6))

fig.suptitle(f"Power, FDP and PCER vs nominal level of conformal selection, {args.dataset} dataset, averaged over {args.seednum} times")
axs[0].plot(fdp_nominals, powers_cs)
axs[0].set_xlabel("nominal level")
axs[0].set_ylabel("power")
axs[1].plot(fdp_nominals, fdps_cs)
axs[1].plot([0, 1], [0, 1], color='green', linestyle='-.', alpha=0.5)
axs[1].set_xlabel("nominal level")
axs[1].set_ylabel("FDP")
axs[2].plot(fdp_nominals, pcers_cs)
axs[2].set_xlabel("nominal level")
axs[2].set_ylabel("PCER")
plt.savefig(f'sheridan-cs-pic\\{args.dataset} {args.sample:.2f}\\Power, FDP and PCER (cs).png')

# plot power comparison

fig, axs = plt.subplots(figsize=(8, 6))

fig.suptitle(f"Power vs FDP for Sheridan's method and Conformal Selection method, \n {args.dataset} dataset, averaged over {args.seednum} times")
# axs.plot(fdps_12, powers_12, label='Sheridan (2012)')
# axs.plot(fdps_04, powers_04, label='Sheridan (2004)')
axs.plot(fdps_15_tv, powers_15_tv, label='Sheridan (2015), tv')
axs.plot(fdps_15_rb, powers_15_rb, label='Sheridan (2015), rb')
axs.plot(fdps_15_rp, powers_15_rp, label='Sheridan (2015), rp')
axs.plot(fdps_cs, powers_cs, label='Conformal Selection')
axs.set_xlabel("FDP")
axs.set_ylabel("Power")
plt.legend()
plt.savefig(f'sheridan-cs-pic\\{args.dataset} {args.sample:.2f}\\Power vs FDP.png')