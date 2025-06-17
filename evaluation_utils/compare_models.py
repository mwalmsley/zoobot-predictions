import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

import evaluate_model



def show_metrics(df: pd.DataFrame, save_loc=None, errorbar=lambda x: (-1.96*evaluate_model.standard_error(x)+x.mean(), 1.96*evaluate_model.standard_error(x)+x.mean())):# lambda x: (x.min(), x.max())


    # plotting setup

    sns.set_theme('paper')

    SMALL_SIZE = 9
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 10

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.rc('font', family='Nimbus Roman')


    # fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(20/3, 10), sharey=True)
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(22/3, 8.2), sharey=True)

    # df = df[df['trained'] != 'DR8 only']  # don
    # hue_order = ['DR5 only', 'DR12 + DR5', 'DR12 + DR5 + DR8']

    print(df['trained'].unique())

    # hue_order = ['GZD-5 only', 'GZD-1/2 + GZD-5 + GZD-8']


    hue_order = ['GZD-8 only', 'GZD-5 only', 'GZD-1/2 + GZD-5', 'GZD-1/2 + GZD-5 + GZD-8']
    palette = sns.color_palette()  # default
    # palette = ['#46E5EB','#00979D', '#00686B', 'k']

    for hue in hue_order:
        if hue not in df['trained'].unique():
            raise ValueError(hue)

    df = df[df['trained'].isin(hue_order)]

    # baseline_mae_per_q_df = df.query('trained == "GZD-5 only"').groupby('answer').agg({'mean_absolute_error': 'mean'})  # GZD-5 baseline
    baseline_mae_per_q_df = df.query('trained == "GZD-8 only"').groupby('answer').agg({'mean_absolute_error': 'mean'})   # GZD-8 baseline
    assert len(baseline_mae_per_q_df) > 0
    baseline_mae_per_q = dict(zip(baseline_mae_per_q_df.index, baseline_mae_per_q_df['mean_absolute_error'].values))

    df['change_in_mae'] = df.apply(lambda x: x['mean_absolute_error'] - baseline_mae_per_q[x['answer']], axis=1)
    # print(df['change_in_mae'])

    df['change_in_mae_pc'] = df.apply(lambda x: x['change_in_mae'] / baseline_mae_per_q[x['answer']], axis=1)
    # print(df['change_in_mae_pc'])


    # no need to show more than one answer to binary question
    df = df[df['answer_clean'] != 'Disk Edge On: No']
    df = df[df['answer_clean'] != 'Has Spiral Arms: No']
    # still too much info - restrict to featured branch
    df = df[df['answer_clean'] != 'How Rounded: Round']
    df = df[df['answer_clean'] != 'How Rounded: In Between']
    df = df[df['answer_clean'] != 'How Rounded: Cigar Shaped']
    df = df[df['answer_clean'] != 'Edge On Bulge: Boxy']
    df = df[df['answer_clean'] != 'Edge On Bulge: None']
    df = df[df['answer_clean'] != 'Edge On Bulge: Rounded']

    # print(df[['trained', 'answer_clean', 'mean_absolute_error', 'change_in_mae_pc']])
 
    sns.barplot(data=df, y='answer_clean', x='mean_absolute_error', hue='trained', ax=ax0, errwidth=0.5, errorbar=errorbar, hue_order=hue_order, palette=palette)
    ax0.set_xlabel('Mean Vote Frac. Error')
    ax0.set_ylabel('')
    ax0.set_xlim([0, 0.2])
    ax0.legend(title=None, loc=(-1., 1.02))  # ncol=2

    print('Average performance improvement of All vs GZD-8 only: ', df.query('trained == "GZD-1/2 + GZD-5 + GZD-8"')['change_in_mae_pc'].mean())

    # df[df['trained'] != 'DR5 only']
    sns.barplot(data=df, y='answer_clean', x='change_in_mae_pc', hue='trained', errwidth=0.5, errorbar=errorbar, hue_order=hue_order, palette=palette, ax=ax1)
    ax1.set_xlabel('Change vs. GZD-8 only')
    ax1.set_ylabel('')
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.))
    ax1.set_xlim([-0.6, 0.6])
    ax1.axvline(0., color='k')
    ax1.get_legend().remove()

    fig.tight_layout()
    if save_loc:
        plt.savefig(save_loc)
    else:
        plt.show()
    # plt.show()


# custom for paper
def compare_dr5_direct_vs_dr8_retrained():
    base_dir = '/home/walml/repos/gz-decals-classifiers/results/campaign_comparison'

    # dr8only_checkpoint = '_desi_pytorch_v5_posthp_train_dr8_test_dr8_m*'
    dr8only_checkpoint = '_desi_pytorch_v5_hpv2_train_dr8_test_dr8_m*'

    # dr5_only_checkpoint = '_desi_pytorch_v5_posthp_train_dr5_test_dr8_m*'   # bad hparams
    dr5_only_checkpoint = '_desi_pytorch_v5_posthp_train_dr5_test_dr8_decals_hparams_m*'

    # dr12_dr5_checkpoint = '_desi_pytorch_v5_posthp_train_dr12dr5_test_dr8_m*'  # bad hparams
    dr12_dr5_checkpoint = '_desi_pytorch_v5_hpv2_train_dr12dr5_test_dr8_m*'

    # dr8_checkpoint = '_desi_pytorch_v5_posthp_train_all_test_dr8_m*'
    dr8_checkpoint = '_desi_pytorch_v5_posthp_train_all_test_dr8_decals_hparams_m*'

    # yes - old hparams do work slightly better, when training on all
    # yes - old hparams also better when training only on dr5

    

    dr8only_trained_locs = glob.glob(os.path.join(base_dir, dr8only_checkpoint, 'regression_errors.csv'))
    # dr8only_trained_locs = glob.glob(os.path.join(base_dir, dr8only_checkpoint, 'regression_errors_dr8_galaxies_with_dr5_head_for_dr8.csv'))
    
    # trained on dr5 only, predicting on dr8 with dr5 head
    dr8_from_dr5_locs = glob.glob(os.path.join(base_dir, dr5_only_checkpoint, 'regression_errors_dr8_galaxies_with_dr5_head_for_dr8.csv'))
    # trained on dr12 and dr5, predicting on dr8 with dr5 head
    dr8_from_dr12_dr5_locs = glob.glob(os.path.join(base_dir, dr12_dr5_checkpoint, 'regression_errors_dr8_galaxies_with_dr5_head_for_dr8.csv'))
    
    # trained on all (dr12, dr5, dr8), predicting on dr8 with dr8 head (this is okay now)
    dr8_trained_locs = glob.glob(os.path.join(base_dir, dr8_checkpoint, 'regression_errors.csv'))
    # dr8_trained_locs = glob.glob(os.path.join(base_dir, dr8_checkpoint, 'regression_errors_dr8_galaxies_with_dr5_head_for_dr8.csv'))  # this would use the dr5 head



    print(len(dr8_from_dr5_locs), len(dr8_from_dr12_dr5_locs), len(dr8_trained_locs), len(dr8only_trained_locs))


    dr8_from_dr5 = pd.concat([pd.read_csv(loc) for loc in dr8_from_dr5_locs], axis=0)
    dr8_from_dr12_dr5 = pd.concat([pd.read_csv(loc) for loc in dr8_from_dr12_dr5_locs], axis=0)
    dr8_trained = pd.concat([pd.read_csv(loc) for loc in dr8_trained_locs], axis=0)
    dr8only_trained = pd.concat([pd.read_csv(loc) for loc in dr8only_trained_locs], axis=0)

    assert not dr8_from_dr5.empty
    assert not dr8_from_dr12_dr5.empty
    assert not dr8_trained.empty
    assert not dr8only_trained.empty
   
    dr8_from_dr5 = tidy_up(dr8_from_dr5)
    dr8_from_dr12_dr5 = tidy_up(dr8_from_dr12_dr5)
    dr8_trained = tidy_up(dr8_trained)
    dr8only_trained = tidy_up(dr8only_trained)

    dr8_from_dr5['trained'] = 'GZD-5 only'
    dr8_from_dr12_dr5['trained'] = 'GZD-1/2 + GZD-5'
    dr8only_trained['trained'] = 'GZD-8 only'
    dr8_trained['trained'] = 'GZD-1/2 + GZD-5 + GZD-8'

    # df = pd.concat([dr8_from_dr5, dr8_from_dr12_dr5, dr8_trained], axis=0).reset_index(drop=True)
    df = pd.concat([dr8_from_dr5, dr8_from_dr12_dr5, dr8_trained, dr8only_trained], axis=0).reset_index(drop=True)

    show_metrics(df, 'comparison_v5.pdf')



def tidy_up(df):
    df['data_release'] = df['answer'].apply(lambda x: x.split('_')[0].split('-')[-1])
    df = df.query('data_release == "dr8"').reset_index(drop=True)
    df['answer_clean'] = df['answer'].apply(lambda x: evaluate_model.clean_text(x.replace('-dr8', ': ')))

        # only comparing the dr8 columns
    for col in df.columns.values:
        if ('dr12' in col) or ('dr5' in col):
            del df[col]

    return df




if __name__ == '__main__':

    compare_dr5_direct_vs_dr8_retrained()

    # base_dir = '/home/walml/repos/gz-decals-classifiers/results/campaign_comparison'
    # # shards = ['dr5']
    # checkpoints = [
    #     # 'all_campaigns_ortho_with_dr8',
    #     # 'all_campaigns_ortho_with_dr8_but_only_train_dr12_dr5',
    #     # 'all_campaigns_ortho_with_dr8_but_only_train_dr5',
    #     'all_campaigns_ortho_with_dr8_nl',
    #     'all_campaigns_ortho_with_dr8_nl_but_train_only_dr5'
    #     ]

    # model_indices = ['m0', 'm1', 'm2']
    # eval_dirs = []
    # # for shard in shards:
    # for checkpoint in checkpoints:
    #     for model_index in model_indices:
    #         # eval_dirs += ['shard_'+ shard + '_checkpoint_' + checkpoint + '_' + model_index]
    #         eval_dirs += ['checkpoint_' + checkpoint + '_' + model_index]

    # # print(eval_dirs)

    # eval_dirs = [os.path.join(base_dir, x) for x in eval_dirs]

    # eval_dirs = [d for d in eval_dirs if os.path.isdir(d)]
    # print(len(eval_dirs))

    # # eval_dirs = [os.path.join(base_dir, x) for x in ['dr5_pretrained_ortho', 'dr5_pretrained_ortho_dr5_only']]
    # # eval_dirs = [os.path.join(base_dir, x) for x in ['shard_dr5_checkpoint__m0', 'shard_dr5_checkpoint_all_campaigns_ortho_with_dr8_but_only_train_dr5_m0', 'shard_dr5_checkpoint_all_campaigns_ortho_with_dr8_but_only_train_dr12_dr5_m0']]
    # # compare_metric_by_question(eval_dirs, metric='mean_loss', save_loc=None)
