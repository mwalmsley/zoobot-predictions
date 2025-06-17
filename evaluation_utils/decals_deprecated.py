
def match_predictions_to_catalog(predictions_hdf5_locs: List, catalog: pd.DataFrame, save_loc=None):

    # predictions will potentially not be the same galaxies as the catalog
    # need to carefully line them up 

    galaxy_id_df, concentrations, _ = load_predictions.load_hdf5s(predictions_hdf5_locs)
    # logging.info(len(galaxy_id_df))
    # logging.info(concentrations.shape)
    # exit()

    assert not any(galaxy_id_df.duplicated(subset=['id_str']))
    assert not any(catalog.duplicated(subset=['id_str']))

    # which concentrations are safely in the catalog?
    catalog_locs = set(catalog['id_str'])

    loc_is_in_catalog = galaxy_id_df['id_str'].isin(catalog_locs) 
    galaxy_id_df = galaxy_id_df[loc_is_in_catalog].reset_index(drop=True)  # now safe to left join on catalog
    concentrations_in_catalog = concentrations[loc_is_in_catalog]  # drop the concentrations corresponding to the dropped rows (galaxies with preds not in the catalog)

    # join catalog labels to df of galaxy ids to create label_df

    print(catalog.iloc[0]['id_str'])
    label_df = pd.merge(galaxy_id_df, catalog, how='left', on='id_str')  # will not change the galaxy_id_df index so will still match the prediction_in_catalog index
    print(label_df['hdf5_loc'].value_counts())
    assert len(galaxy_id_df) == len(label_df)
    assert len(label_df) > 0

    print('Predictions: {}'.format(len(label_df)))
    # print('Galaxies from each hdf5:')

    if save_loc:
        label_df.to_parquet(save_loc, index=False)

    return label_df, concentrations_in_catalog


def multi_catalog_tweaks(label_df: pd.DataFrame):
    smooth_featured_cols = [col for col in label_df.columns.values if ('smooth-or-featured' in col) and ('total-votes') in col]
    # print(smooth_featured_cols)
    match_with_dr = {
        'smooth-or-featured-dr12_total-votes': 'has_dr12_votes',
        'smooth-or-featured-dr5_total-votes': 'has_dr5_votes',
        'smooth-or-featured-dr8_total-votes': 'has_dr8_votes',
    }

    for col in smooth_featured_cols:
        matched_col = match_with_dr[col]
        label_df[matched_col] = label_df[col] > 0
        print('Test galaxies with {} non-zero votes: {}'.format(col, label_df[matched_col].sum()))

    return label_df


"""Plotting"""


# copied from trust_the_model.ipynb
def show_galaxies(df, scale=3, nrows=6, ncols=6):
    fig = plt.gcf()

    # plt.figure(figsize=(scale * nrows * 1.505, scale * ncols / 2.59))
    plt.figure(figsize=(scale * nrows * 1., scale * ncols))
    gs1 = gridspec.GridSpec(nrows, ncols)
    gs1.update(wspace=0.0, hspace=0.0)
    galaxy_n = 0
    for row_n in range(nrows):
        for col_n in range(ncols):
            galaxy = df.iloc[galaxy_n]
            image = Image.open(galaxy['file_loc'])
            ax = plt.subplot(gs1[row_n, col_n])
            ax.imshow(image)

            ax.text(35, 40, 'smooth: V={:.2f}, ML={:.2f}'.format(galaxy['smooth-or-featured_smooth_true_fraction'], galaxy['smooth-or-featured_smooth_predicted_fraction']), fontsize=12, color='r')
            if galaxy['smooth-or-featured_smooth_true_fraction'] < 0.5:
                ax.text(35, 100, 'arms: V={:.2f}, ML={:.2f}'.format(galaxy['has-spiral-arms_yes_true_fraction'], galaxy['has-spiral-arms_yes_predicted_fraction']), fontsize=12, color='r')


                # ax.text(35, 50, 'Vol: 2={:.2f}, ?={:.2f}'.format(galaxy['spiral-arm-count_2_fraction'], galaxy['spiral-arm-count_cant-tell_fraction']), fontsize=12, color='r')
                # ax.text(35, 100, 'ML: 2={:.2f}, ?={:.2f}'.format(galaxy['spiral-arm-count_2_ml_fraction'], galaxy['spiral-arm-count_cant-tell_ml_fraction']), fontsize=12, color='r')
    #             ax.text(10, 50, r'$\rho = {:.2f}$, Var ${:.3f}$'.format(galaxy['median_prediction'], 3*galaxy['predictions_var']), fontsize=12, color='r')
    #             ax.text(10, 80, '$L = {:.2f}$'.format(galaxy['bcnn_likelihood']), fontsize=12, color='r')
            ax.axis('off')
            galaxy_n += 1
    #     print('Mean L: {:.2f}'.format(df[:nrows * ncols]['bcnn_likelihood'].mean()))
    fig = plt.gcf()
    fig.tight_layout()
    return fig





def main(model_n):

    question_answer_pairs = label_metadata.decals_all_campaigns_ortho_pairs
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    # logging.info('Schema: {}'.format(schema))

    """Pick which model's predictions to load"""

    model_index = f'm{model_n}'

    # always uses (all randomly selected): N% from DR8, and then potentially N% from dr5 and N% from dr12
    # so in practice, always uses DR5, always uses DR8 (as test only potentially), sometimes also uses DR12
    # hence DR8 and DR12DR5 has 80k (DR5+dr12+DR8) but DR5 has 63k (missing DR12 test set)

    # v4 catalog, new (bad) hparams
    # experiment_dir = f'_desi_pytorch_v4_posthp_train_all_test_dr8_{model_index}' # 1 2 3   #80k
    # experiment_dir = f'_desi_pytorch_v4_posthp_train_dr12dr5_test_dr8_{model_index}' # 1 2 3  # 80k
    # experiment_dir = f'_desi_pytorch_v4_posthp_train_dr5_test_dr8_{model_index}' # 1 2 3  # 63k
    # experiment_dir = f'_desi_pytorch_v4_posthp_train_dr8_test_dr8_{model_index}' # 1 2 3   #20k, DR8 only

    # v5 catalog, new (bad) hparams except where specified
    # experiment_dir = f'_desi_pytorch_v5_posthp_train_all_test_dr8_{model_index}' # 1 2 3 4 5   #80k
    experiment_dir = f'_desi_pytorch_v5_posthp_train_all_test_dr8_decals_hparams_{model_index}' # 1 2 3 4 5   # USED in paper, equiv to hpv2
    # experiment_dir = f'_desi_pytorch_v5_posthp_train_dr12dr5_test_dr8_{model_index}' # 1 2 3 4 5  # 80k
    # experiment_dir = f'_desi_pytorch_v5_posthp_train_dr5_test_dr8_{model_index}' # 1 2 3 4 5  # 63k
    # experiment_dir = f'_desi_pytorch_v5_posthp_train_dr8_test_dr8_{model_index}' # 1 2 3 4 5   #20k, DR8 only
    # experiment_dir = f'_desi_pytorch_v5_posthp_train_dr5_test_dr8_decals_hparams_{model_index}'  # USED in paper, equiv to hpv2

    # v5 catalog, back to decals hparams always
    # experiment_dir = f'_desi_pytorch_v5_hpv2_train_all_test_dr8_{model_index}' # 1 2 3 4 5   # does not exist, used above
    # experiment_dir = f'_desi_pytorch_v5_hpv2_train_dr12dr5_test_dr8_{model_index}' # 1 2 3 4 5 
    # experiment_dir = f'_desi_pytorch_v5_hpv2_train_dr5_test_dr8_{model_index}' # 1 2 3 4 5  # does not exist, used above
    # experiment_dir = f'_desi_pytorch_v5_hpv2_train_dr8_test_dr8_{model_index}' # 1 2 3 4 5 

    # rsync -avz walml@galahad.ast.man.ac.uk:/share/nas2/walml/repos/gz-decals-classifiers/results/pytorch/desi /nvme1/scratch/walml/repos/gz-decals-classifiers/results/pytorch

    predictions_hdf5_locs = [f'/home/walml/repos/gz-decals-classifiers/results/pytorch/desi/{experiment_dir}/test_predictions.hdf5']
    # print(predictions_hdf5_locs)
    predictions_hdf5_locs = [loc for loc in predictions_hdf5_locs if os.path.exists(loc)]
    assert len(predictions_hdf5_locs) > 0
    logging.info('Num. prediction .hdf5 to load: {}'.format(len(predictions_hdf5_locs)))

    """Specify some details for saving"""
    save_dir = f'/home/walml/repos/gz-decals-classifiers/results/campaign_comparison/{experiment_dir}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # normalize_cm_matrices = 'true'
    normalize_cm_matrices = None


    """Load volunteer catalogs and match to predictions"""

    catalog_dr12 = pd.read_parquet(f'/home/walml/repos/gz-decals-classifiers/data/catalogs/training_catalogs/dr12_ortho_v5_labelled_catalog.parquet')
    catalog_dr5 = pd.read_parquet(f'/home/walml/repos/gz-decals-classifiers/data/catalogs/training_catalogs/dr5_ortho_v5_labelled_catalog.parquet')
    catalog_dr8 = pd.read_parquet(f'/home/walml/repos/gz-decals-classifiers/data/catalogs/training_catalogs/dr8_ortho_v5_labelled_catalog.parquet')
    catalog = pd.concat([catalog_dr12, catalog_dr5, catalog_dr8], axis=0).reset_index()

    # recalculate total votes as dr5 catalog has dodgy totals
    for question in schema.questions:
        catalog[question.text + '_total-votes'] = catalog[[a.text for a in question.answers]].sum(axis=1)

    # possibly, catalogs don't include _fraction cols?! 
    # for question in schema.questions:
    #     for answer in question.answers:
    #         catalog[answer.text + '_fraction'] = catalog[answer.text].astype(float) / catalog[question.text + '_total-votes'].astype(float)

    all_labels, all_concentrations = match_predictions_to_catalog(predictions_hdf5_locs, catalog, save_loc=None)
    # print(len(all_labels))
    # print(len(all_concentrations))
    # print(all_labels.head())
    # exit()

    all_labels = multi_catalog_tweaks(all_labels)

    # not actually used currently
    all_fractions = stats.expected_value_of_dirichlet_mixture(all_concentrations, schema)

    # plt.hist(all_labels['smooth-or-featured-dr12_total-votes'], alpha=.5, label='dr12', range=(0, 80))  # 7.5k retired
    # plt.hist(all_labels['smooth-or-featured_total-votes'], alpha=.5, label='dr5', range=(0, 80))  # 2.2k retired
    # plt.hist(all_labels['smooth-or-featured-dr8_total-votes'], alpha=.5, label='dr8', range=(0, 80))  # half have almost no votes, 4.6k retired
    # plt.show()
    # # exit()

    votes_for_retired = 34  # ALSO CHANGE IN FILTER_TO_SENSIBLE ETC
    all_labels['is_retired_in_dr12'] = all_labels['smooth-or-featured-dr12_total-votes'] > votes_for_retired
    all_labels['is_retired_in_dr5'] = all_labels['smooth-or-featured-dr5_total-votes'] > votes_for_retired
    all_labels['is_retired_in_dr8'] = all_labels['smooth-or-featured-dr8_total-votes'] > votes_for_retired
    all_labels['is_retired_in_any_dr'] = all_labels['is_retired_in_dr12'] | all_labels['is_retired_in_dr5'] | all_labels['is_retired_in_dr8']
    # all_labels['is_retired_in_multiple_dr'] = all_labels['is_retired_in_dr12'].astype(int) + all_labels['is_retired_in_dr5'].astype(int) + all_labels['is_retired_in_dr8'].astype(int)
    logging.info('Retired by DR: DR12={}, DR5={}, DR8={}'.format(all_labels['is_retired_in_dr12'].sum(), all_labels['is_retired_in_dr5'].sum(), all_labels['is_retired_in_dr8'].sum()))
    # exit()

    # dr8
    # DR12=12450, DR5=11077, DR8=10056
    # DR12=12431, DR5=11160, DR8=9923
    # dr12dr5
    # DR12=12439, DR5=11073, DR8=9971
    # DR12=12428, DR5=11015, DR8=9957
    # dr5
    # DR12=0, DR5=10959, DR8=9906
    # DR12=0, DR5=10921, DR8=10122
    # dr8 only
    # DR12=0, DR5=0, DR8=9924

    retired_concentrations = all_concentrations[all_labels['is_retired_in_any_dr']]
    retired_labels = all_labels.query('is_retired_in_any_dr')
    retired_fractions = stats.expected_value_of_dirichlet_mixture(retired_concentrations, schema)

    logging.info('All concentrations: {}'.format(all_concentrations.shape))
    logging.info('Retired concentrations: {}'.format(retired_concentrations.shape))

    # print(all_labels['is_retired_in_dr5'].sum())
    # exit()

    """Now we're ready to calculate some metrics"""

    # # temp - some spiral checks
    # spirals = all_labels['is_retired_in_dr8'] & (all_labels['spiral-arm-count-dr8_total-votes'] > (all_labels['smooth-or-featured-dr8_total-votes'] / 2))

    # spiral_labels = all_labels[spirals]
    # spiral_concentrations = all_concentrations[spirals]
    # spiral_q = schema.get_question('spiral-arm-count-dr8')

    # first_q = schema.get_question('smooth-or-featured-dr8')
    # first_answers = ['smooth-or-featured-dr8_smooth', 'smooth-or-featured-dr8_featured-or-disk', 'smooth-or-featured-dr8_artifact']
    # print(spiral_labels[first_answers][:4])
    # print(spiral_concentrations[:4, first_q.start_index:first_q.end_index+1])
    
    # dr8_spiral_answers = ['spiral-arm-count-dr8_{}'.format(x) for x in ['1','2', '3', '4','more-than-4','cant-tell']]
    # # print(spiral_labels[dr8_spiral_answers][:4])

    # dr5_spiral_answers = ['spiral-arm-count-dr5_{}'.format(x) for x in ['1','2', '3', '4','more-than-4','cant-tell']]

    # print(spiral_concentrations[:4, spiral_q.start_index:spiral_q.end_index+1])
    # print(spiral_concentrations[:4, spiral_q.start_index:spiral_q.end_index+1])
    # # print(spiral_labels[])
    # exit()

    # create_paper_metric_tables(retired_labels, retired_fractions, schema)

    # # the least interpretable but maybe most ml-meaningful metric
    # # unlike cm and regression, does not only include retired (i.e. high N) galaxies
    # val_loss, loss_by_q_df = get_loss(all_labels, all_concentrations, schema=schema, save_loc=os.path.join(save_dir, 'val_loss_by_q.csv'))
    # print('Mean val loss: {:.3f}'.format(val_loss.mean()))
    confusion_matrices_split_by_confidence(retired_labels, retired_fractions, schema, save_dir, normalize=normalize_cm_matrices, cm_name='cm')

    # print((retired_labels['smooth-or-featured-dr12_total-votes'] > 20).sum())

    get_regression_errors(
        retired=retired_labels,
        predicted_fractions=retired_fractions,
        schema=schema,
        df_save_loc=os.path.join(save_dir, 'regression_errors.csv'),
        fig_save_loc=os.path.join(save_dir, 'regression_errors_bar_plot.pdf')
    )


    """And we can repeat the process, but using the DR5 predictions for the DR8 answer columns"""
    logging.info('Repeating with DR5 predictions for DR8 galaxies')

    # pick the rows with dr8 galaxies (dr8_)
    dr8_galaxies = all_labels['has_dr8_votes'].values
    dr8_labels = all_labels[dr8_galaxies]
    dr8_concentrations = all_concentrations[dr8_galaxies]
    dr8_fractions = all_fractions[dr8_galaxies]
    # convert the predictions for dr8 answers to use the dr5 answers instead
    dr8_fractions_with_dr5_head = replace_dr8_cols_with_dr5_cols(dr8_fractions, schema)
    dr8_concentrations_with_dr5_head = replace_dr8_cols_with_dr5_cols(dr8_concentrations, schema)

    # calculate loss on all dr8 galaxies, using dr5 head answers
    val_loss, loss_by_q_df = get_loss(dr8_labels, dr8_concentrations_with_dr5_head, schema=schema, save_loc=os.path.join(save_dir, 'val_loss_by_q_dr8_galaxies_with_dr5_head_for_dr8.csv'))
    logging.info('Mean val loss for DR8 galaxies using DR5 head for DR8: {:.3f}'.format(val_loss.mean()))
    
    # select the retired dr8 galaxies by row
    dr8_retired_galaxies = dr8_labels['is_retired_in_dr8'].values
    dr8_retired_labels = dr8_labels[dr8_retired_galaxies]
    # dr8_retired_concentrations_with_dr5_head = dr8_concentrations_with_dr5_head[dr8_retired_galaxies] (not used)
    dr8_retired_fractions_with_dr5_head = dr8_fractions_with_dr5_head[dr8_retired_galaxies]

    # predict on dr8 galaxies
    # using dr5 head (renamed to -dr8 questions)
    # will give a bunch of warnings as we've only selected dr8 galaxies hence dr12 and dr5 will not have enough votes and give empty confusion matrices
    # e.g. -smooth-or-featured-smooth-dr5 cm will be empty because no dr8 galaxies have enough (any) smooth-dr5 labels
    confusion_matrices_split_by_confidence(
        dr8_retired_labels,
        dr8_retired_fractions_with_dr5_head,
        schema,
        save_dir,
        normalize=normalize_cm_matrices,
        cm_name='cm_dr8_galaxies_with_dr5_head_for_dr8'
    )

    # similarly will throw dr12 and dr5 warnings
    get_regression_errors(
        retired=dr8_retired_labels,
        predicted_fractions=dr8_retired_fractions_with_dr5_head,
        schema=schema,
        df_save_loc=os.path.join(save_dir, 'regression_errors_dr8_galaxies_with_dr5_head_for_dr8.csv'),
        fig_save_loc=os.path.join(save_dir, 'regression_errors_dr8_galaxies_with_dr5_head_for_dr8_bar_plot.pdf')
    )


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    for model_n in range(1, 6):
        main(model_n)

