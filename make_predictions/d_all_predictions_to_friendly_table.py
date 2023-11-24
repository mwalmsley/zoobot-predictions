import argparse
import logging
import os

import hydra
from omegaconf import DictConfig

from zoobot.shared import schemas, load_predictions


@hydra.main(version_base=None, config_path="../conf", config_name='default')
def main(config: DictConfig):

    """
    
    this should only be run for hdf5 locs from a single model
    other models might make predictions in different orders and it will be a real pain to group them afterwards
    instead, run this individually for each model

    every model will predict on all galaxies, in exactly the same order (but hopefully not a necessary assumption)
    this will produce output prediction hdf5s for each snippet
    each hdf5 has the 3D (galaxy, question, forward pass) concentrations and the id_str

    Args:
        hdf5_loc (_type_): _description_
        save_loc (_type_): _description_
        debug (_type_): _description_
    """
    debug_trigger_dirs = [
        '/nvme1/scratch/walml',
        '/User/user/',
        '/home/walml/programs'
    ]
    if any([os.path.isdir(trigger_dir) for trigger_dir in debug_trigger_dirs]):
        logging.warning('Debug system detected - forcing debug mode')
        debug = True  # always debug locally
    else:
        debug = config.aggregation.debug

    # schema = getattr(schemas, config.model.schema_name)\
    from zoobot.shared import schemas
    schema = schemas.decals_dr8_ortho_schema
    
    grouped_across_models_loc = os.path.join(config.predictions_dir, 'grouped_across_models.hdf5')
    
    final_predictions_loc = os.path.join(config.predictions_dir, 'predictions.parquet')  # will be renamed to _friendly.parquet, _advanced.paruet
    load_predictions.prediction_hdf5_to_summary_parquet(grouped_across_models_loc, final_predictions_loc, schema, debug)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='predict by slice')
    # parser.add_argument('--hdf5-loc', dest='hdf5_loc', type=str, default='/Users/user/repos/zoobot-predictions/example/predictions/evo_py_co_vittiny_224_all.hdf5')
    # parser.add_argument('--save-loc', dest='save_loc', type=str, default='/Users/user/repos/zoobot-predictions/example/evo_py_co_vittiny_224.parquet')
    # parser.add_argument('--debug', default=False, action='store_true')
    # args = parser.parse_args()

    # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    # main(args.hdf5_loc, args.save_loc, args.debug)

    # logging.warning('Exiting gracefully')

    main()

    """
    python make_predictions/d_all_predictions_to_friendly_table.py \
        --hdf5-loc /Users/user/repos/zoobot-predictions/example/predictions/evo_py_co_vittiny_224_all.hdf5 \
        --save-loc /Users/user/repos/zoobot-predictions/example/evo_py_co_vittiny_224.parquet
    """
