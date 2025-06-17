from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from PIL import Image
# from zoobot.shared import schemas

# lightly adapted from MER_Morphology validation checks
def show_qualitative_grid(df):
    fig = plt.figure(figsize=(25., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(5, 8),  # creates 2x2 grid of axes
                     axes_pad=(0.1, .1),  # pad between axes in inch.
                     share_all=True
                     )

    for ax_n, ax in enumerate(grid):
        row_n = ax_n // 2
        try:
            row = df.iloc[row_n]

            if ax_n % 2 == 0:
                # reconstruct the thumbnail that Zoobot would have seen
                im = Image.open(row['jpg_loc_composite'])
            else:
                description = '\n'.join(get_description(row))
                # print(description)
                ax.text(0., .98, description, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', fontsize=9.)
        except:
            ## This means that I have less than 36 entries in my sample
            if ax_n % 2 == 0:
                im = np.zeros((50,50))
                ax.imshow(im.squeeze(), cmap='gray')
            else:
                description = 'Empty cutout'
                ax.text(0., .98, description, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', fontsize=9.)

        ax.axis('off')
    
    return fig


def get_description(row):

    descriptions = []

    ## Adding the existence condition since this is still not in Euclid
    if 'smooth-or-featured_problem_fraction' in row and row['smooth-or-featured_problem_fraction'] > 0.5:
        descriptions.append('Artifact ({:.0f}%)'.format(100*row['smooth-or-featured_problem_fraction']))
    else:
        feat = row['smooth-or-featured_featured-or-disk_fraction']
        smooth = row['smooth-or-featured_smooth_fraction']
        if feat > smooth:
            descriptions.append('Featured ({:.0f}%)'.format(100*row['smooth-or-featured_featured-or-disk_fraction']))
            descriptions += get_featured_description(row)
        else:
            descriptions.append('Smooth ({:.0f}%)'.format(100*row['smooth-or-featured_smooth_fraction']))

    if len(descriptions) > 0:
        descriptions += ['']  # i.e. add \n once we get to \n.join(). Needs '' else .join will skip.
    if 'merging_merger_fraction' in row.keys():
        descriptions += get_merger_description(row)

    return descriptions


def get_featured_description(row):
    descriptions = []
    
    if row['disk-edge-on_yes'] > .5:
        descriptions.append('edge-on disk ({:.0f}%)'.format(100 * row['disk-edge-on_yes']))
    
    descriptions += get_spiral_description(row)
    descriptions += get_bar_description(row)

    return descriptions


def get_spiral_description(row, schema):
    descriptions = []

    if row['has-spiral-arms_fraction'] > 0.6:
        descriptions.append('\nSpiral ({:.0f}%), with'.format(100*row['has-spiral-arms_fraction']))

        # question = 'spiral-arm-count'
        # answers = [k for k in ZOOBOT_OUTPUT_COLUMNS_MER_FORMAT if question in k]
        # frac_cols = [answer + '_fraction' for answer in answers]
        # answers = EUCLID_QUESTION_ANSWER_PAIRS[question]
        # frac_cols = [question + answer.text + '_fraction' for answer in schema.get_question(question).answers]\
        
        spiral_winding_answers = ['_tight', '_medium', '_loose'],
        spiral_arm_count_answers = ['_1', '_2', '_3', '_4', '_more_than_4', '_cant_tell'],


        frac_cols = ['spiral-arm-count_' + a for a in spiral_arm_count_answers]
        most = frac_cols[np.argmax(row[frac_cols])]
        most_str = '{} arms ({:.0f}%),'.format(most.split('_')[-2].lower().replace('-', ' ').replace('tell', 'unclear'), row[most]*100)
        descriptions.append(most_str)

        frac_cols = ['spiral-winding_' + a for a in spiral_winding_answers]
        # answers = [k for k in ZOOBOT_OUTPUT_COLUMNS_MER_FORMAT if question in k]
        # frac_cols = [answer + '_fraction' for answer in answers]
        # answers = EUCLID_QUESTION_ANSWER_PAIRS[question]
        # frac_cols = [question + answer + '_fraction' for answer in answers]
        # frac_cols = [question + answer.text + '_fraction' for answer in schema.get_question(question).answers]
        most = frac_cols[np.argmax(row[frac_cols])]
        most_str = '{} winding ({:.0f}%)'.format(most.split('_')[-2].lower().replace('-', ' '), row[most]*100)
        descriptions.append(most_str)

    return descriptions
        

def get_bar_description(row):
    descriptions = []
    if row['bar_no_fraction'] < .3:
        bar_weak = float(row['bar_weak_fraction'])
        bar_strong = float(row['bar_strong_fraction'])
        if bar_weak > bar_strong:
            descriptions.append('\nWeak ({:.0f}%) bar ({:.0f}%)'.format(bar_weak * 100., 100*(bar_weak + bar_strong)))
        else:
            descriptions.append('\nStrong ({:.0f}%) bar ({:.0f}%)'.format(bar_strong * 100., 100*(bar_weak + bar_strong)))

        # descriptions.append('')
    return descriptions


def get_merger_description(row):
    description = []
    merger = float(row['merging_merger_fraction'])
    major = float(row['merging_major-disturbance_fraction'])
    minor = float(row['merging_minor-disturbance_fraction'])
    any_merger = merger + major + minor 
    # no = row['merging_none_fraction']
    if any_merger > 0.3:
        description.append('Disturbed ({:.0f}%)'.format(any_merger * 100))
    return description
