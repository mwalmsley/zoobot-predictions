import os
import logging
from typing import List

import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import seaborn as sns
# from PIL import Image
import numpy as np
import pandas as pd
from sklearn import metrics

from zoobot.shared import schemas
# from galaxy_datasets.shared import label_metadata
# from zoobot.tensorflow.training import losses
from zoobot.pytorch.training import losses
from zoobot.shared import stats

"""Text utils"""

def get_label(text, question):
    return clean_text(text.replace(question.text, '').title())


def clean_text(text):
    return text.replace('-', ' ').replace('_', '').title()


def standard_error(x):
    return np.std(x)/len(x)



"""Discrete Metrics"""


def print_metrics(question, label_df, predicted_fractions, schema, min_responses:int, style='human'):

    y_true, y_pred = get_integer_responses(question, label_df, predicted_fractions, schema, min_responses)

    # print(pd.value_counts(y_true, sort=False))

    # how to handle multi-class metrics - see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
    # average = 'micro'  # "Calculate metrics globally by counting the total true positives, false negatives and false positives.""
    average = 'weighted'  # "Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).""
    
    if style == 'human':
        print('Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f} <- {}'.format(
            metrics.accuracy_score(y_true, y_pred),
            metrics.precision_score(y_true, y_pred, average=average),
            metrics.recall_score(y_true, y_pred, average=average),
            metrics.f1_score(y_true, y_pred, average=average),
            question.text
        ))
    elif style == 'latex':
        # questions count accuracy precision recall f1
        print('{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\'.format(
            question.text.replace('-', ' ').replace('_', ' ').title(),
            len(y_true),
            metrics.accuracy_score(y_true, y_pred),
            # 0,
            metrics.precision_score(y_true, y_pred, average=average, zero_division=np.nan),
            metrics.recall_score(y_true, y_pred, average=average, zero_division=np.nan),
            # 0,
            metrics.f1_score(y_true, y_pred, average=average, zero_division=np.nan)
        ))


def get_integer_responses(question: schemas.Question, label_df: pd.DataFrame, predicted_fractions: np.array, schema: schemas.Schema, min_responses: int):
    """
    Turn actual/predicted answer fractions into actual/predicted binary responses.

    For all predictions, pick only ones with at least min_responses and > 50% of humans answering,
    and return an integer (axis=1 index in predicted fractions) denoting the most likely response.

    Args:
        question (schemas.Question): _description_
        label_df (pd.DataFrame): _description_
        predicted_fractions (np.array): _description_
        schema (schemas.Schema): _description_
        min_responses (int): _description_

    Returns:
        _type_: _description_
    """

    # print(label_df['has_dr12_votes'].sum(), label_df['has_dr5_votes'].sum(), label_df['has_dr8_votes'].sum())

    # at least 1 human vote, at least half of humans answered the question (in this DR). And from earlier: retired anywhere (i.e. >34 total votes anywhere)
    # what if >34 were asked in *another DR*, and half of humans answered question in THIS DR - few votes, then?
    valid_labels, valid_predictions = filter_to_sensible(label_df, predicted_fractions, question, schema, min_responses=min_responses)

    cols = [answer.text + '_fraction' for answer in question.answers]
    # most likely answer, might be less than .5 though
    y_true = np.argmax(valid_labels[cols].values, axis=1)
    y_pred = np.argmax(valid_predictions[:, question.start_index:question.end_index+1], axis=1)
    return y_true, y_pred


def show_confusion_matrix(question, label_df, predicted_fractions, schema, min_responses:int, ax=None, blank_yticks=False, add_title=False, normalize=None):
    y_true, y_pred = get_integer_responses(question, label_df, predicted_fractions, schema, min_responses)
    
    labels = range(len(question.answers))

    # TODO add more summary metrics here
    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize).transpose()  # I think I need to transpose to get correct true/pred arrangement
    
    ticklabels = [get_label(a.text, question) for a in question.answers]
    
    # manual tweaks
    for n in range(len(ticklabels)):
        if ticklabels[n] == 'Featured Or Disk':
            ticklabels[n] = 'Feat./Disk'
        elif ticklabels[n] == 'Cigar Shaped':
            ticklabels[n] = 'Cigar'
        elif ticklabels[n] == 'More Than 4':
            ticklabels[n] = r'$>4$'
        elif ticklabels[n] == 'Cant Tell':
            ticklabels[n] = '?'
        elif ticklabels[n] == 'Minor Disturbance':
            ticklabels[n] = 'Minor Dist.'
        elif ticklabels[n] == 'Major Disturbance':
            ticklabels[n] = 'Major Dist.'
            
    if ax is None:
        _, ax = plt.subplots(figsize=(4., 4.))
    
    if add_title:
        ax.set_title(clean_text(question.text))

    if blank_yticks:
    #         yticklabels = ['' for _ in ticklabels]
            yticklabels = [''.join([' '] * len(s)) for s in ticklabels]
    else:
        yticklabels = ticklabels

    if normalize == 'true':
        fmt = '.3f'
    else:
        fmt = 'd'
    return sns.heatmap(
        cm,
        annot=True,
        cmap='Blues',
        fmt=fmt,
        xticklabels=ticklabels,
        yticklabels=yticklabels,
        cbar=False,
    #         annot_kws={"size": 14},
        ax=ax,
        square=True,
        robust=True
    ), acc


def confusion_matrices_split_by_confidence(label_df: pd.DataFrame, predicted_fractions: np.ndarray, schema: schemas.Schema, min_responses:int, save_dir: str, cm_name: str, normalize=None):
    logging.info('Making confusion matrices')
    for question in schema.questions:
        
        fig = plt.figure(constrained_layout=True, figsize=(4, 2.))
        gs = fig.add_gridspec(8, 10)

        ax0 = fig.add_subplot(gs[:6, 0])
        ax1 = fig.add_subplot(gs[:6, 1:5])
        ax2 = fig.add_subplot(gs[:6, 6:])

        ax3 = fig.add_subplot(gs[6:, 1:5])
        ax4 = fig.add_subplot(gs[6:, 6:])

        ax5 = fig.add_subplot(gs[:6, 5:6])
        
    #     fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 10))
        fig, acc = show_confusion_matrix(question, label_df, predicted_fractions, schema, min_responses, ax=ax1, normalize=normalize)

        fractions = np.array([label_df[answer.text + '_fraction'] for answer in question.answers])
        high_confidence = np.any(fractions > 0.8, axis=0)
        fig, acc_high_conf = show_confusion_matrix(question, label_df[high_confidence], predicted_fractions[high_confidence], schema, min_responses, ax=ax2, blank_yticks=False, normalize=normalize)
        
        label_size = 10
        ax3.text(0.5, 0.75, 'True', horizontalalignment='center', verticalalignment='center', fontsize=label_size)
        ax4.text(0.5, 0.75, 'True', horizontalalignment='center', verticalalignment='center', fontsize=label_size)

        ax0.text(0.8, 0.5, 'Predicted', rotation='vertical', horizontalalignment='center', verticalalignment='center', fontsize=label_size)
        ax0.text(-0.5, 0.5, clean_text(question.text).replace('Dr8', ''), rotation='vertical', horizontalalignment='center', verticalalignment='center', fontsize=label_size, weight='bold')

        for ax in [ax0, ax3, ax4, ax5]:
            ax.grid('off')
            ax.axis('off')

        if normalize == 'true':
            norm_text = 'normalized'
        else:
            norm_text = ''

        name = f'{save_dir}/{cm_name}_{norm_text}_{question.text}'

        plt.savefig(f'{name}.pdf')   # pdf is too big? we'll see
        # plt.clf()  
        plt.close()

        metric_df = pd.DataFrame([{'accuracy': acc, 'accuracy_high_conf': acc_high_conf, 'question': question.text}])
        metric_df.to_csv(f'{name}.csv', index=False)




def filter_to_sensible(label_df: pd.DataFrame, predictions: np.array, question: schemas.Question, schema: schemas.Schema, min_responses: int):
    """
    Select galaxies with:
    - at least 1 human vote (not ML-predicted human votes, actual human votes) for that question
    - at least 50% of humans answered this q (product of previous actual vote fractions must be > 0.5)
    
    Filter label_df and predictions to only those galaxies (and return)

    Args:
        label_df (_type_): _description_
        predictions (_type_): anything indexable like label_df
        question (_type_): _description_
        schema (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert len(label_df) == len(predictions)

    sensible_indices = indices_of_sensible_labels(label_df, question, schema, min_responses=min_responses)

    return label_df[sensible_indices], predictions[sensible_indices]


def indices_of_sensible_labels(label_df: pd.DataFrame, question: schemas.Question, schema: schemas.Schema, min_responses: int):
    """
    Get indices with at least min_responses and a leaf probability for question of at least 0.5

    Args:
        label_df (pd.DataFrame): _description_
        question (schemas.Question): _description_
        schema (schemas.Schema): _description_
        min_responses (int): _description_

    Returns:
        _type_: _description_
    """

    # label_df must include _total-votes columns plus vote cols matching schema

    # new - explicitly require min responses. Requiring e.g. 34 total votes, plus half of people asked this q, would work, but can become confused about 34 total votes in *another* DR
    total_votes_for_q = label_df[question.text + '_total-votes']
    meets_min_responses = total_votes_for_q >= min_responses

    # might be better called "actual fraction of people answering this q"
    # human fractions imply you'd get this many votes for this question


    # this bit effectively recalculates proportion_asked, but that's fine

    # 0 if no votes, 1 otherwise (i.e. assume the base q has 1 vote, below)
    base_q_votes = (total_votes_for_q > 0).values.astype(int)

    expected_votes = stats.get_expected_votes_human(label_df, question, base_q_votes, schema, round_votes=False)  
    if not isinstance(expected_votes, np.ndarray):
        expected_votes = expected_votes.numpy()  # hack, should fix properly...

    at_least_half_of_humans_would_answer = expected_votes > 0.5
    if at_least_half_of_humans_would_answer.sum() == 0:
        logging.warning('No galaxies with more than half of humans answering: {}, {} candidates'.format(question, len(label_df)))

    sensible_indices = meets_min_responses & at_least_half_of_humans_would_answer

    if sensible_indices.sum() == 0:
        logging.warning('No galaxies with more than half of humans answering: {}, {} candidates'.format(question, len(label_df)))
        raise ValueError(question.text, schema, min_responses, len((label_df)))


    return sensible_indices



def print_paper_style_metric_tables(label_df, predicted_fractions, schema, min_responses: int, style='human'):

    print('Metrics on all galaxies:')
    print('Question & Count & Accuracy & Precision & Recall & F1 \\')  # need to add second slash back manually
    print('\hline \hline')
    for question in schema.questions:
        print_metrics(question, label_df, predicted_fractions, schema, min_responses, style=style)

    print('Metrics on (retired) high confidence galaxies:')
    print(r'Question & Count & Accuracy & Precision & Recall & F1 \\')
    print('\hline \hline')
    for question in schema.questions:

        answers = question.answers
        fractions = np.array([label_df[answer.text + '_fraction'] for answer in answers])
        high_confidence = np.any(fractions > 0.8, axis=0)
        print('High conf galaxies: {}'.format(high_confidence.sum()))
        print_metrics(question, label_df[high_confidence], predicted_fractions[high_confidence], schema, min_responses, style=style)



def get_regression_errors(label_df: pd.DataFrame, predicted_fractions: np.ndarray, schema: schemas.Schema, min_responses, df_save_loc=None, fig_save_loc=None):

    logging.info('Getting regression errors')
    errors = []
    for question_n, question in enumerate(schema.questions):
        valid_labels, valid_predictions = filter_to_sensible(label_df, predicted_fractions, question, schema, min_responses=min_responses)
        if len(valid_labels) == 0:
            logging.warning('Skipping regression - no valid labels/predictions for answer {}, {} candidates'.format(question.text, len(label_df)))
        else:
            valid_labels = valid_labels.reset_index(drop=True)
            enough_votes = valid_labels[question.text + '_total-votes'] >= min_responses  # not quite the same as filter_to_sensible
            # print(enough_votes.mean())
            if not any(enough_votes):
                logging.warning('Skipping regression - valid labels/predictions but none with more than {} votes - {}'.format(min_responses, answer.text))
            else:
                for answer in question.answers:
                    y_true = valid_labels.loc[enough_votes, answer.text + '_fraction']
                    y_pred = valid_predictions[enough_votes, answer.index]
                    assert not pd.isna(y_true).any()
                    assert not pd.isna(y_pred).any()
                    # print(len(y_true), len(y_pred))


                    # get the metrics
                    absolute = metrics.mean_absolute_error(y_true, y_pred)
                    mse = metrics.mean_squared_error(y_true, y_pred)

                    # also bootstrap for errors
                    # select a random 80%
                    # recalculate metrics
                    bootstrap_absolute = []
                    bootstrap_mse = []
                    for _ in range(50):
                        random_indices = np.random.choice(range(len(y_true)), int(0.8 * len(y_true)), replace=False)
                        y_true_bootstrap = y_true[random_indices]
                        y_pred_bootstrap = y_pred[random_indices]
                        bootstrap_absolute.append(metrics.mean_absolute_error(y_true_bootstrap, y_pred_bootstrap))
                        bootstrap_mse.append(metrics.mean_absolute_error(y_true_bootstrap, y_pred_bootstrap))
                        

                    # acc = metrics.accuracy_score(y_true > 0.5, y_pred)
                    # is_high_confidence = np.any(y_true > 0.8, axis=0)
                    # acc_high_conf = metrics.accuracy_score(y_true[is_high_confidence], y_pred[is_high_confidence])
                    errors.append({
                        'answer': answer.text,
                        'rmse': np.sqrt(mse),
                        'mean_absolute_error': absolute,

                        'mean_absolute_error_bootstrap_95pc': np.percentile(bootstrap_absolute, 95),
                        'mean_absolute_error_bootstrap_5pc': np.percentile(bootstrap_absolute, 5),

                        'mse_bootstrap_95pc': np.percentile(bootstrap_mse, 95),
                        'mse_bootstrap_5pc': np.percentile(bootstrap_mse, 5),

                        # 'acc': acc,
                        # 'acc_high_conf': acc_high_conf,
                        'question_n': question_n,
                        'total_galaxies': len(y_true),
                        'mean_vol_answer': y_true.mean(),
                        'mean_ml_answer': y_pred.mean(),

                    })

    assert len(errors) > 0
    regression_errors = pd.DataFrame(errors)

    if df_save_loc:
        regression_errors.to_csv(df_save_loc, index=False)

    if fig_save_loc:
        # sns.set_style('whitegrid', {'axes.edgecolor': '0.2'})
        # sns.set_context('notebook')
        # sns.set_palette(repeating_palette)
        # fig, ax = plt.subplots(figsize=(6, 20))
        fig, ax = plt.subplots(figsize=(5, 8))

        # temporarily ignore lowest options
        exclude = ['clumps', 'problem', 'artifact']
        for e in exclude:
            regression_errors = regression_errors[~regression_errors['answer'].str.startswith(e)]

        sns.barplot(data=regression_errors, y='answer', x='mean_absolute_error', ax=ax)
        plt.xlabel('Vote Fraction Mean Deviation')
        plt.ylabel('')
        fig.tight_layout()
        plt.savefig(fig_save_loc)

    return regression_errors


def regression_scatterplot(retired, predicted_fractions, answer_text, schema):

    # question = schema.get_question('has-spiral-arms')
    answer = schema.get_answer(answer_text)
    question = answer.question
    # answer = question.answers[0]
    valid_labels, valid_predictions = filter_to_sensible(retired, predicted_fractions, question, schema)
    sns.scatterplot(valid_labels[answer.text + '_fraction'], valid_predictions[:, answer.index], alpha=.1)

# kind of a hassle, not currently bothering
# def replace_dr12_preds_with_dr5_preds(predictions, schema):
#     predictions = predictions.copy()  # do not modify inplace
#     # predictions uses ortho schema

#     # shared questions are any q in the non-ortho dr5/dr12 schema that don't have "dr12" in them
#     # (decals_dr12_questions uses dr5 columns and col-dr12 for changed columns)
#     shared_questions = [q for q in label_metadata.decals_dr12_questions if "dr12" not in q.text]
#     # these will have the wrong indices for predictions/schema, which is ortho - just use for text

#     for q in shared_questions:
#         dr5_ortho_q = schema.get_question(q.text) + '-dr5' # will appear in the ortho schema (would have -dr5, but is implicit in this case)
#         dr12_ortho_q = schema.get_question(q.text + '-dr12')

#         # now replace the dr5 predictions with the dr12 values, both using the ortho schema
#         for answer_n in range(q.answers):
#             dr5_ortho_answer = dr5_ortho_q.answers[answer_n]
#             dr12_ortho_answer = dr12_ortho_q.answers[answer_n]
#             predictions[dr12_ortho_answer.index] = predictions[dr5_ortho_answer.index]

#     return predictions


def get_loss(label_df, concentrations, schema, save_loc=None):

    answers = [a.text for a in schema.answers]
    import torch
    labels = torch.Tensor(label_df[answers].values)
    concentrations = torch.Tensor(concentrations)

    # multiquestion_loss = losses.get_multiquestion_loss(schema.question_index_groups, sum_over_questions=False)
    # loss = lambda x, y: multiquestion_loss(x, y) / batch_size  

    # tf_labels = tf.constant(labels.astype(np.float32))
    # tf_preds = tf.constant(concentrations.astype(np.float32))[:, :, 0]  # TODO just picking first dropout pass for now
    # print(tf_labels.shape, tf_preds.shape)

    loss = losses.calculate_multiquestion_loss(labels, concentrations, schema.question_index_groups).numpy()  # this version doesn't reduce
    print(loss.shape)

    mean_loss_by_q = loss.mean(axis=0).squeeze()
    rows = [{'question': question.text, 'mean_loss': l} for question, l in zip(schema.questions, mean_loss_by_q)]
    
    loss_by_q_df = pd.DataFrame(data=rows)

    if save_loc:
        loss_by_q_df.to_csv(save_loc, index=False)

    return loss, loss_by_q_df



# TODO move
def replace_dr8_cols_with_dr5_cols(predicted_fracs: np.ndarray, schema: schemas.Schema):
    predicted_fracs = predicted_fracs.copy()
    # this is easier than the above as the columns match exactly
    dr5_questions = [q for q in schema.questions if 'dr5' in q.text]
    dr8_questions = [q for q in schema.questions if 'dr8' in q.text]
    # will line up

    for dr5_q, dr8_q in zip(dr5_questions, dr8_questions):
        for dr5_a, dr8_a in zip(dr5_q.answers, dr8_q.answers):
            predicted_fracs[:, dr8_a.index] = predicted_fracs[:, dr5_a.index]

    return predicted_fracs

