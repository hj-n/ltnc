"""
SENSITIVITY ANALYSIS (Section 5.1) - DRAWING FIGURE
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

sys.path.append("../libs")
import sensitivity_helpers as sh

X_AXIS_LABELS = [
	"Replacement prob.", "Angle btw two discs", "Dist. to the origin",
	"Number of PCs", "Replacement prob.", "Dist. to the origin", "Starting index of PC"
]
INVERT_AXIS   = [False, True, True, True, False, True, False]
XTICK_INTS    = [False, False, False, True, False, False, True]
SHOW_Y_LABELS = [True, False, False, False, False, False, False]

sns.set_style("whitegrid")
fig, axs = plt.subplots(3, 7, figsize=(20, 10), sharey="row")

### plot linecharts
for col_idx, exp_idx in enumerate(["A", "B_1", "B_2", "C", "D", "E", "F"]):
	results = pd.read_csv(f"./results/02_sensitivity_{exp_idx}.csv")
	sh.lineplot_ax(results, axs[0][col_idx], (0, 4)  , X_AXIS_LABELS[col_idx], "score", f"Exp {exp_idx}", INVERT_AXIS[col_idx], False, SHOW_Y_LABELS[col_idx], True, XTICK_INTS[col_idx])
	sh.lineplot_ax(results, axs[1][col_idx], (4, 12) , X_AXIS_LABELS[col_idx], "score", f"Exp {exp_idx}", INVERT_AXIS[col_idx], False, SHOW_Y_LABELS[col_idx], False, XTICK_INTS[col_idx])
	sh.lineplot_ax(results, axs[2][col_idx], (12, 16), X_AXIS_LABELS[col_idx], "score", f"Exp {exp_idx}", INVERT_AXIS[col_idx], True , SHOW_Y_LABELS[col_idx], False, XTICK_INTS[col_idx])
	axs[0, col_idx].margins(x=0)
	axs[1, col_idx].margins(x=0)
	axs[2, col_idx].margins(x=0)


axs[0, 0].text(-0.41, 0.05, "Ours (Label-T&C)", rotation="vertical", fontsize=13, transform=axs[0, 0].transAxes,c="red")
axs[1, 0].set_title("Measures w/o Labels", rotation="vertical", x=-0.38, y=-0.07, fontsize=13, c="blue")
axs[2, 0].set_title("Measures w/ Labels", rotation="vertical", x=-0.38, y=-0.03, fontsize=13, c="purple")


### plot legends
sh.legend_ax(bbox_to_anchor=(0.4, -0.16), ncol=4, fontsize=11, ax= axs[0, 3], index=(0,4))
sh.legend_ax(bbox_to_anchor=(0.4, -0.16), ncol=8, fontsize=11, ax= axs[1, 3], index=(4,12))
sh.legend_ax(bbox_to_anchor=(0.4, -0.26), ncol=4, fontsize=11, ax= axs[2, 3], index=(12,16))

plt.subplots_adjust(bottom=0.2, hspace=0.48)

plt.savefig(f"./plot/02_sensitivity.png", dpi=300)
plt.savefig(f"./plot/02_sensitivity.pdf", dpi=300)