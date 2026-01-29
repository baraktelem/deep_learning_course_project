import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# --- Data Setup ---
# Removed 4000 from the list
bcy = np.array([10, 50 ,100, 200, 300, 400, 500, 1000])

# Baseline (Sliced to remove the last point corresponding to n=4000)
a = np.array([25.6, 40.5, 48.68, 56.56, 67.44, 75.61, 76.9, 83.93, 93.84])[:-1]

# The Layers (Sliced L1 and L3 to match L5's length)
l1 = np.array([27.420, 47.250, 54.87, 67.320, 71.68, 75.85, 77.69, 83.28, 93.12])[:-1]
l3 = np.array([29.26, 49.6, 58.550, 68.440, 73.62, 76.5, 77.29, 84.81, 93.73])[:-1]

# Layer 5 (Already ended at n=1000, so we use it as is)
l5 = np.array([25.25, 46.27, 58.46, 68.98, 72.71, 76.12, 77.97, 85.560])

# --- Plot Setup ---
fig, ax = plt.subplots(figsize=(12, 7))

# 1. Baseline (Reference)
ax.plot(bcy, a, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')

# 2. Layer 1 (Green)
ax.plot(bcy, l1, marker='o', color='#2ca02c', linewidth=2.5, markersize=8, label='Layer 1')

# 3. Layer 3 (Blue)
ax.plot(bcy, l3, marker='D', color='#1f77b4', linewidth=3, markersize=8, label='Layers 1-3')

# 4. Layer 5 (Red)
ax.plot(bcy, l5, marker='X', color='#d62728', linewidth=2.5, linestyle='-', alpha=0.8, label='Layers 1-5')

# --- Styling ---
ax.set_xscale('log')
ax.set_xlabel('Training Samples per Class', fontsize=14, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Integration Depth vs. Sample Size', fontsize=16, fontweight='bold', pad=20)

# Ticks
ax.set_xticks(bcy)
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.set_xticklabels(bcy, fontsize=11, fontweight='bold')

# Grid & Legend
ax.grid(True, which="both", ls="-", alpha=0.3)
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)

# --- Annotation ---
# Highlighting the final point where L5 wins
#ax.annotate('Deep Integration Wins\n(High Data)', xy=(1000, 85.5), xytext=(500, 88),
#            arrowprops=dict(facecolor='#d62728', shrink=0.05), fontsize=11, fontweight='bold', color='#d62728')

plt.tight_layout()
plt.show()
