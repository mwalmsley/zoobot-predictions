import glob
import os

import numpy as np

import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import PIL
import torch

# basically a copy of view_mae_predictions

if __name__ == '__main__':
        
    torch.serialization.add_safe_globals([PIL.Image.Image])
    
    on_datalabs = os.path.isdir('/media/home/my_workspace')
    if on_datalabs:
        # max_images = 128
        prediction_dir = '/media/home/team_workspaces/Galaxy-Zoo-Euclid/huggingface/predictions/mwalmsley/euclid_q1/euclid-rr2-mae-lightning'
    else:
        max_images = None
        prediction_dir = '/home/walml/repos/zoobot-foundation/results/tmp_predictions'

    batch_locs = glob.glob(f'{prediction_dir}/0/*.pt')
    assert batch_locs, f'No predictions found in {prediction_dir}'

    bad_images = []
    bad_masks = []
    bad_reconstructions = []

    save_loc = os.path.join(prediction_dir, '0', '_bad_reconstructions.jpg')

    for batch_loc in tqdm.tqdm(batch_locs):
        # batch_index = os.path.basename(batch_loc).replace('.pt', '')  # 0, 1...
 
        # if os.path.isfile(save_loc):
        #     continue

        batch_preds = torch.load(batch_loc)

        images = batch_preds['images']
        masked = batch_preds['masked']
        reconstructed = batch_preds['reconstructed']
        reconstruction_error = batch_preds['reconstruction_error'].cpu().numpy()
        # print(np.percentile(reconstruction_error, [99, 99.5, 99.9, 100]))
        # exit()
        min_error = 0.007

        # worst_indices = np.argsort(-reconstruction_error)  # descending
        worst_indices = [i for i in range(len(images)) if reconstruction_error[i] > min_error]
        if not worst_indices:
            continue

        for i in worst_indices:
            bad_images.append(images[i])
            bad_masks.append(masked[i])
            bad_reconstructions.append(reconstructed[i])

    n_images = len(bad_images)

    fig = plt.figure(figsize=(5., n_images))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(n_images, 3),  # creates 2x2 grid of Axes
                    axes_pad=0.05,  # pad between Axes in inch.
                    )

    def get_rows(grid):
        n_ax = len(grid) 
        grid = iter(grid)
        for _ in range(n_ax // 3):
            yield [next(grid) for _ in range(3)]

    for i, row in enumerate(get_rows(grid)):

        # put i in the lower corner
        row[0].text(0.15, 0.1, str(i), ha='center', va='center', transform=row[0].transAxes, color='white', fontsize=8)

        row[0].imshow(bad_images[i])
        row[0].axis('off')

        row[1].imshow(bad_masks[i])
        row[1].axis('off')

        row[2].imshow(bad_reconstructions[i])
        row[2].axis('off')

    fig.savefig(save_loc)
    plt.close(fig)
