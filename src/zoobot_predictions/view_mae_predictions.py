import glob
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import PIL
import torch



if __name__ == '__main__':
        
    torch.serialization.add_safe_globals([PIL.Image.Image])

    
    on_datalabs = os.path.isdir('/media/home/my_workspace')
    if on_datalabs:
        prediction_dir = '/media/home/team_workspaces/Galaxy-Zoo-Euclid/huggingface/predictions/mwalmsley/euclid_q1/euclid-rr2-mae-lightning/0'
    else:
        prediction_dir = '/home/walml/repos/zoobot-foundation/results/tmp_predictions'

    batch_locs = glob.glob(f'{prediction_dir}/0/*.pt')
    for batch_loc in batch_locs:
        batch_index = os.path.basename(batch_loc).replace('.pt', '')  # 0, 1...

        batch_preds = torch.load(batch_loc)
        # print(batch_preds)
        # for k, v in batch_preds.items():
        #     print(k)
        #     if isinstance(v, list):
        #         print(len(v))
        #     else:
        #         print(k, v.shape)



        images = batch_preds['images']
        masked = batch_preds['masked']
        reconstructed = batch_preds['reconstructed']
        reconstruction_error = batch_preds['reconstruction_error']


        n_images = len(images)

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

            row[0].imshow(images[i])
            # row[0].set_title('Original Image')
            row[0].axis('off')

            row[1].imshow(masked[i])
            # row[1].set_title('Masked Image')
            row[1].axis('off')

            row[2].imshow(reconstructed[i])
            # row[2].set_title(f'Reconstructed Image\nError: {reconstruction_error[i]:.4f}')
            row[2].axis('off')

        fig.savefig(f'/home/walml/repos/zoobot-foundation/foundation/experiments/hybrid/{batch_index}_images.jpg')
