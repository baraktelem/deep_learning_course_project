import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# --- Data Setup ---
bcy = np.array([10, 50 ,100, 200, 300, 400, 500, 1000, 4000])

# Accuracy Data
a = np.array([25.6, 40.5, 48.68, 56.56, 67.44, 75.61, 76.9, 83.93, 93.84]) # Baseline
b = np.array([28.53, 47.62, 58.49, 66.06, 70.49, 73.13, 74.24, 80.63, 92.42]) # Parametric Concat
c = np.array([27.420, 47.250, 54.87, 67.320, 71.68, 75.85, 77.69, 83.28, 93.12]) # Parametric Max
d = np.array([27.45, 46.64, 57.76, 64.24, 71.92, 74.68, 76.44, 83.13, 93.22]) # Fixed Scat

# --- Plot Setup ---
fig, ax1 = plt.subplots(figsize=(12, 7))

# --- Left Axis: Test Accuracy ---
l1, = ax1.plot(bcy, a, marker='o', color='#1f77b4', linewidth=2.5, markersize=8, label='Baseline (ResNet18)')
l2, = ax1.plot(bcy, d, marker='s', color='#ff7f0e', linewidth=2.5, markersize=8, label='Fixed Scat. (Concat)')
l3, = ax1.plot(bcy, b, marker='^', color='#2ca02c', linewidth=2.5, markersize=8, label='Param. Scat. (Concat)')
l4, = ax1.plot(bcy, c, marker='D', color='#d62728', linewidth=2.5, markersize=8, label='Param. Scat. (Max)')

ax1.set_xscale('log')
ax1.set_xlabel('Training Samples per Class', fontsize=14, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
ax1.grid(True, which="both", ls="-", alpha=0.3)

# Force X-axis to show integer numbers
ax1.set_xticks(bcy)
ax1.get_xaxis().set_major_formatter(ScalarFormatter())
ax1.set_xticklabels(bcy, fontsize=11, fontweight='bold')

# --- Legend ---
lines = [l1, l2, l3, l4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower right', fontsize=11, framealpha=0.9)

plt.title('Test Accuracy vs. Sample Size', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
