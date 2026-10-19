import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Updated Data from Thesis Table 4.1 (with the corrected 0.01s LQR runtime)
controllers = ['Stanley', 'LQR', 'Real MPC', 'Neural MPC']
lap_times = [38.34, 41.07, 31.23, 32.22]
runtimes = [0.03, 0.01, 7.89, 0.25] 

# Colors matching your screenshot
colors = ['#6DA4BB', '#B67397', '#EFAC4D', '#D4735E']

# Setup the figure and the primary axis (Lap Time)
fig, ax1 = plt.subplots(figsize=(11, 7))

x = np.arange(len(controllers))
width = 0.35  # the width of the bars

# 1. Plot Lap Completion Time (Solid Bars) on left axis
rects1 = ax1.bar(x - width/2, lap_times, width, 
                 color=colors, edgecolor='black', linewidth=1, alpha=0.9)

ax1.set_ylabel('Lap Completion Time [s]', fontsize=12)
ax1.set_ylim(0, 45)  # Set limit slightly above max lap time
ax1.set_xticks(x)
ax1.set_xticklabels(controllers, rotation=10, fontsize=10)

# 2. Setup the secondary axis (Runtime)
ax2 = ax1.twinx()

# Plot Operating Runtime (Hatched Bars) on right axis
rects2 = ax2.bar(x + width/2, runtimes, width, 
                 color=colors, edgecolor='black', linewidth=1, hatch='//', alpha=0.9)

ax2.set_ylabel('Operating Runtime [s] (Log Scale)', fontsize=12)
ax2.set_yscale('log')
# Set the lower limit to slightly below 0.01 so the new LQR bar is clearly visible
ax2.set_ylim(0.005, 20) 

# 3. Add Data Labels on top of the bars
for rect in rects1:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., height + 0.5,
             f'{height:.1f}s', ha='center', va='bottom', fontsize=9)

for rect in rects2:
    height = rect.get_height()
    # For log scale, multiply by a small factor to position text just above the bar
    ax2.text(rect.get_x() + rect.get_width()/2., height * 1.1,
             f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

# 4. Custom Legend (Gray dummy patches to explain the fill styles)
legend_elements = [
    Patch(facecolor='gray', edgecolor='black', label='Lap Completion Time'),
    Patch(facecolor='gray', edgecolor='black', hatch='//', label='Operating Runtime (Log Scale)')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Title and layout
plt.title('Controller Efficiency: Performance vs. Cost', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the figure
filename = 'corrected_figure_4_4_barchart.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved '{filename}' successfully! Ready to drop into your thesis.")