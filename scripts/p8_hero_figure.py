"""Hero Figure: 3-panel synthesis for NC submission
(a) KE velocity trend dipole map with jet axis shift + GS inset
(b) Three-method time series
(c) Causal chain schematic with quantitative rates
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import pandas as pd
import json

FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")

# Load pre-made panel images
fig_velocity = mpimg.imread(str(FIG / "fig6_velocity_analysis.png"))
fig_three = mpimg.imread(str(FIG / "fig_three_methods.png"))

# Create hero figure
fig = plt.figure(figsize=(18, 16))

# ── Panel (a): Velocity dipole (use existing fig6 panel a) ──
ax_a = fig.add_axes([0.02, 0.55, 0.55, 0.42])
# Crop fig6 to just panel (a) — top third
h = fig_velocity.shape[0]
panel_a = fig_velocity[:h//3, :, :]
ax_a.imshow(panel_a)
ax_a.axis('off')
ax_a.set_title('(a) KE Geostrophic Speed Trend: North Acceleration / South Deceleration',
               fontsize=12, fontweight='bold', pad=10)

# ── GS inset ──
fig_gs = mpimg.imread(str(FIG / "fig7_gulf_stream_velocity.png"))
ax_gs = fig.add_axes([0.58, 0.55, 0.40, 0.42])
panel_gs_a = fig_gs[:fig_gs.shape[0]//3, :, :]
ax_gs.imshow(panel_gs_a)
ax_gs.axis('off')
ax_gs.set_title('Gulf Stream: No Dipole (Control)', fontsize=11, fontweight='bold', pad=10)

# ── Panel (b): Three-method time series ──
ax_b = fig.add_axes([0.05, 0.28, 0.90, 0.22])
ax_b.imshow(fig_three)
ax_b.axis('off')
ax_b.set_title('(b) Three Tracking Methods on Same Data: Opposite Conclusions',
               fontsize=12, fontweight='bold', pad=10)

# ── Panel (c): Causal chain schematic ──
ax_c = fig.add_axes([0.05, 0.02, 0.90, 0.22])
ax_c.set_xlim(0, 10)
ax_c.set_ylim(0, 3)
ax_c.axis('off')
ax_c.set_title('(c) Physical Mechanism: From Hadley Widening to Self-Concealing Bias',
               fontsize=12, fontweight='bold', pad=10)

# Draw causal chain boxes and arrows
boxes = [
    (0.3, 1.8, 'Hadley Cell\nWidening', '~0.5°/dec', '#FFE0B2'),
    (2.3, 1.8, 'WSC Zero Line\nPoleward Shift', '0.58°/dec\n(Wu 2018)', '#C8E6C9'),
    (4.5, 1.8, 'KE Jet\nDisplacement', '0.14°/dec\n(centroid)', '#BBDEFB'),
    (4.5, 0.5, 'KE Jet\nIntensification', '0.37°/dec\n(asymmetric)', '#E1BEE7'),
    (7.0, 1.8, 'Differential\nSea Level Rise', '+1.9 cm/dec', '#FFCDD2'),
    (7.0, 0.5, 'SLA Methods\nBiased', '−0.14°/dec\n(reversed!)', '#F5F5F5'),
]

for x, y, label, rate, color in boxes:
    rect = mpatches.FancyBboxPatch((x, y), 1.6, 0.9, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
    ax_c.add_patch(rect)
    ax_c.text(x+0.8, y+0.55, label, ha='center', va='center', fontsize=8, fontweight='bold')
    ax_c.text(x+0.8, y+0.15, rate, ha='center', va='center', fontsize=7, color='#333333')

# Arrows
arrow_style = dict(arrowstyle='->', color='black', lw=2, connectionstyle='arc3,rad=0')
arrows = [
    ((1.9, 2.25), (2.3, 2.25)),   # Hadley → WSC
    ((3.9, 2.25), (4.5, 2.25)),   # WSC → displacement
    ((5.3, 1.8), (5.3, 1.4)),     # displacement → intensification (vertical)
    ((6.1, 2.25), (7.0, 2.25)),   # displacement → diff SLR
    ((7.8, 1.8), (7.8, 1.4)),     # diff SLR → SLA bias
]
for (x1,y1), (x2,y2) in arrows:
    ax_c.annotate('', xy=(x2,y2), xytext=(x1,y1),
                  arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Feedback arrow (red, curved)
ax_c.annotate('', xy=(7.8, 0.5), xytext=(9.2, 1.8),
              arrowprops=dict(arrowstyle='->', color='red', lw=2,
                              connectionstyle='arc3,rad=-0.5'))
ax_c.text(9.3, 1.2, 'Self-\nconcealing', color='red', fontsize=8, fontweight='bold',
          ha='center', va='center')

# STMW feedback
ax_c.text(9.3, 0.3, 'STMW ↓\nCO₂ sink ↓\n→ warming ↑',
          ha='center', va='center', fontsize=7, color='#B71C1C',
          bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#B71C1C'))

plt.savefig(FIG / "fig_hero.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(FIG / "fig_hero.pdf", dpi=300, bbox_inches='tight', facecolor='white')
print(f"Hero figure 保存: {FIG / 'fig_hero.png'}")
plt.close()

import shutil
shutil.copy(FIG / "fig_hero.pdf", "/Users/zhulin/aitest/黑潮延伸体/manuscript/fig_hero.pdf")
print("已复制到 manuscript/fig_hero.pdf")
