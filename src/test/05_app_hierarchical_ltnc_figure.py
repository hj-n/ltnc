
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


GRANULARITY = 20

with open(f"./results/05_app_hierarchical_ltnc_{GRANULARITY}_btw_ch.json", "r") as f:
	btw_ch_data = json.load(f)

with open(f"./results/05_app_hierarchical_ltnc_{GRANULARITY}_dsc.json", "r") as f:
	dsc_data = json.load(f)

def _reformat(nestedData: pd.DataFrame, column_name: str):
	decouples = nestedData[column_name].apply(pd.Series)
	decouples = pd.melt(decouples, value_vars=decouples.columns, var_name='level', value_name='score')
	decouples = decouples.loc[~decouples['score'].isna(), :]
	decouples['metric'] = column_name
	return decouples

btw_ch_stats = []
for frame in btw_ch_data:
	data = pd.DataFrame(frame.get('result', None))
	data_ls = _reformat(data, 'ls')
	data_lc = _reformat(data, 'lc')
	data = pd.concat([data_ls, data_lc], ignore_index=True)
	data['type'] = frame.get('dr_type', '')
	btw_ch_stats.append(data)

dsc_stats = []
for frame in dsc_data:
	data = pd.DataFrame(frame.get('result', None))
	data_ls = _reformat(data, 'ls')
	data_lc = _reformat(data, 'lc')
	data = pd.concat([data_ls, data_lc], ignore_index=True)
	data['type'] = frame.get('dr_type', '')
	dsc_stats.append(data)

btw_ch_stats = pd.concat(btw_ch_stats, ignore_index=True)
dsc_stats = pd.concat(dsc_stats, ignore_index=True)

lc_btw_ch_df = btw_ch_stats.loc[btw_ch_stats['metric'] == 'lc']
ls_btw_ch_df = btw_ch_stats.loc[btw_ch_stats['metric'] == 'ls']
lc_dsc_df = dsc_stats.loc[dsc_stats['metric'] == 'lc']
ls_dsc_df = dsc_stats.loc[dsc_stats['metric'] == 'ls']

lc_btw_ch_df = lc_btw_ch_df[lc_btw_ch_df['score'] <= 1]
ls_btw_ch_df = ls_btw_ch_df[ls_btw_ch_df['score'] <= 1]
lc_dsc_df = lc_dsc_df[lc_dsc_df['score'] <= 1]
ls_dsc_df = ls_dsc_df[ls_dsc_df['score'] <= 1]

lc_btw_ch_df = lc_btw_ch_df[lc_btw_ch_df['score'] >= 0]
ls_btw_ch_df = ls_btw_ch_df[ls_btw_ch_df['score'] >= 0]
lc_dsc_df = lc_dsc_df[lc_dsc_df['score'] >= 0]
ls_dsc_df = ls_dsc_df[ls_dsc_df['score'] >= 0]

sns.set_style("whitegrid")

fig, ax = plt.subplots(1, 4, figsize=(17, 2.5))

sns.lineplot(x = 'level', y = 'score', hue = 'type', data = lc_btw_ch_df, ax = ax[0], ci=0, legend=False)
ax[0].set_title('(A) Label-Trustworhiness [$CH_{btwn}$]')

sns.lineplot(x = 'level', y = 'score', hue = 'type', data = ls_btw_ch_df, ax = ax[1], ci=0, legend=False)
ax[1].set_title('(B) Label-Continuity [$CH_{btwn}$]')
ax[1].set_ylabel('')


sns.lineplot(x = 'level', y = 'score', hue = 'type', data = lc_dsc_df, ax = ax[2], ci=0)
ax[2].set_title('(C) Label-Trustworhiness [DSC]')
ax[2].set_ylabel('')

## place the legned outside the plot (bottom)
ax[2].legend(loc='upper center', bbox_to_anchor=(-0.4, -0.3), ncol=6)
## change the texts in the legend to [t-SNE, UMAP, PCA, Isomap, LLE, Densmap]
handles, labels = ax[2].get_legend_handles_labels()
ax[2].legend(handles, ['t-SNE', 'UMAP', 'PCA', 'Isomap', 'LLE', 'Densmap'], loc='upper center', bbox_to_anchor=(-0.1, -0.22), ncol=6)



sns.lineplot(x = 'level', y = 'score', hue = 'type', data = ls_dsc_df, ax = ax[3], ci=0, legend=False)
ax[3].set_title('(D) Label-Continuity [DSC]')
ax[3].set_ylabel('')

ax[0].set_xlabel("Cluster granularity")
ax[1].set_xlabel("Cluster granularity")
ax[2].set_xlabel("Cluster granularity")
ax[3].set_xlabel("Cluster granularity")


## add text to the plot
ax[0].text(-1, 0.403, "($fine$)", fontsize=10, ha="left")
ax[0].text(20, 0.403, "($coarse$)", fontsize=10, ha="right")
ax[1].text(-1, 0.798, "($fine$)", fontsize=10, ha="left")
ax[1].text(20, 0.798, "($coarse$)", fontsize=10, ha="right")
ax[2].text(-1, 0.554, "($fine$)", fontsize=10, ha="left")
ax[2].text(20, 0.554, "($coarse$)", fontsize=10, ha="right")
ax[3].text(-1, 0.9917, "($fine$)", fontsize=10, ha="left")
ax[3].text(20, 0.9917, "($coarse$)", fontsize=10, ha="right")
# sns.lineplot(x = 'level', y = 'score', hue = 'type', data = lc_btw_ch_df, ax = ax[4])



plt.savefig(f"./plot/05_app_hierarchical_ltnc.pdf", bbox_inches='tight', dpi=300)
plt.savefig(f"./plot/05_app_hierarchical_ltnc.png", bbox_inches='tight', dpi=300)