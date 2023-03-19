
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




GRANULARITY = 20

with open(f"./shc_data/data_granularity_{GRANULARITY}_btw_ch.json") as f:
	btw_ch_data = json.load(f)

with open(f"./shc_data/data_granularity_{GRANULARITY}_dsc.json") as f:
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

fig, ax = plt.subplots(1, 5, figsize=(21, 5))

sns.lineplot(x = 'level', y = 'score', hue = 'type', data = lc_btw_ch_df, ax = ax[0], ci=0, legend=False)
ax[0].set_title('Label-Trustworhiness ($CH_{btwn}$)')

sns.lineplot(x = 'level', y = 'score', hue = 'type', data = ls_btw_ch_df, ax = ax[1], ci=0, legend=False)
ax[1].set_title('Label-Continuity ($CH_{btwn}$)')

ax[0].get_shared_y_axes().join(ax[0], ax[1])
ax[1].set_ylabel('')
ax[1].set_yticks([])


sns.lineplot(x = 'level', y = 'score', hue = 'type', data = lc_dsc_df, ax = ax[2], ci=0)
ax[2].set_title('Label-Trustworhiness ($DSC$)')
ax[2].set_ylabel('')

## place the legned outside the plot (bottom)
ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=6)
## change the texts in the legend to [t-SNE, UMAP, PCA, Isomap, LLE, Densmap]
handles, labels = ax[2].get_legend_handles_labels()
ax[2].legend(handles, ['t-SNE', 'UMAP', 'PCA', 'Isomap', 'LLE', 'Densmap'], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

sns.lineplot(x = 'level', y = 'score', hue = 'type', data = ls_dsc_df, ax = ax[3], ci=0, legend=False)
ax[3].set_title('Label-Continuity ($DSC$)')

ax[2].get_shared_y_axes().join(ax[2], ax[3])
ax[3].set_ylabel('')
ax[3].set_yticks([])

sns.lineplot(x = 'level', y = 'score', hue = 'type', data = ls_dsc_df, ax = ax[4], ci=0, legend=False)
ax[4].set_title('Label-Continuity ($DSC$) / Zoomed')

ax[4].set_ylabel('')




# sns.lineplot(x = 'level', y = 'score', hue = 'type', data = lc_btw_ch_df, ax = ax[4])



plt.savefig(f"./shc_data/results_{GRANULARITY}_figure.png", bbox_inches='tight', dpi=300)
plt.savefig(f"./shc_data/results_{GRANULARITY}_figure.pdf", bbox_inches='tight', dpi=300)