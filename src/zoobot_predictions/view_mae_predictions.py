
import matplotlib.pyplot as plt
import PIL
import torch

if __name__ == '__main__':
        
    torch.serialization.add_safe_globals([PIL.Image.Image])

    batch_preds = torch.load('/home/walml/repos/zoobot-foundation/results/tmp_predictions/0/0.pt')
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

    # n_images = images.shape[0]
    n_images = 16

    for i in range(len(images)):

    fig, rows = plt.subplots(n_images, 3, figsize=(6, 3 * n_images))
    for i in range(n_images):
        rows[i, 0].imshow(images[i])
        rows[i, 0].set_title('Original Image')
        rows[i, 0].axis('off')

        rows[i, 1].imshow(masked[i])
        rows[i, 1].set_title('Masked Image')
        rows[i, 1].axis('off')

        rows[i, 2].imshow(reconstructed[i])
        rows[i, 2].set_title(f'Reconstructed Image\nError: {reconstruction_error[i]:.4f}')
        rows[i, 2].axis('off')

    fig.savefig('/home/walml/repos/zoobot-foundation/foundation/experiments/hybrid/load_preds.png')