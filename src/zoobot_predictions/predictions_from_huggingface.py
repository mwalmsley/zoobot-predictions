

import json
import os
import hydra

import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter

import torch

from representations_from_huggingface import create_datamodule

# zoobot foundation
from foundation.experiments.hybrid import pretrain

class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir: str, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        torch.save(prediction, os.path.join(self.output_dir, str(dataloader_idx), f"{batch_idx}.pt"))




@hydra.main(version_base=None, config_name="config")
def main(cfg):

    L.seed_everything(42)

    on_datalabs = os.path.isdir('/media/home/my_workspace')

    # datalabs only, for token
    if on_datalabs:
        with open('/media/home/my_workspace/_credentials/secrets.txt') as f:
            token = json.load(f)['token']
    else:
        token = None


    # this bit is different
    model = pretrain.BaseHybridLearner.load_from_checkpoint(cfg.model.model_name)
    model.to('cuda')
    model.eval()

    datamodule = create_datamodule(cfg, split='train', token=token)

    pred_dir = '/home/walml/repos/zoobot-foundation/results/tmp_predictions'
    pred_writer = CustomWriter(output_dir=pred_dir, write_interval="batch")

    trainer = L.Trainer(
        devices=1,
        callbacks=[pred_writer]
    )
    datamodule.setup('test')
    trainer.predict(model, dataloaders=datamodule.test_dataloader(), return_predictions=False)


if __name__ == '__main__':
    main()

    """
    python zoobot_predictions/predictions_from_huggingface.py +dataset=debug +model=mae +hardware=home ++model.model_name="/home/walml/Dropbox (The University of Manchester)/euclid/euclid_morphology/models/u11vazbb/checkpoints/model.ckpt"
    """