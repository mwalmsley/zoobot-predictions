# zoobot-predictions
Make arbitrary-scale (100k+ galaxies) predictions (inc. representations) from Zoobot.

Predictions are simple enough with lightning and galaxy-datasets, but it's fiddly to make predictions on very large numbers of galaxies distributed across a cluster. There's also some further aggregation to ensemble together the multiple forward passes from one model and then the multiple models, and create nice neat tables.

The approach is:

- Divide the galaxy catalog into ordered subsets (snippets), each with relatively few galaxies (not crucial, say 10k). See `make_hsc_snippets.py`
- Assign each node to make single-model predictions on a few of those snippets. This is done by using a slurm task array to provide each node with different snippet indexes (e.g. snippets 0-5, snippets 6-10, etc) to make predictions on.  See `a_make_bulk_catalog_predictions.sh`
- Predictions by one model on each snippet are saved as hdf5 arrays, with shape (galaxy_in_snippet, question, forward pass). See `a_make_bulk_catalog_predictions.py`
- Group all the predictions by one model on every snippet into a single hdf5 file with shape (galaxy, question, forward pass). See `b_group_hdf5_from_a_model.py`
- Group those single hdf5 files across models. Simply stack them for a single hdf5 array of shape (galaxy, question, model, forward pass). See `c_group_hdf5_across_models.py`
- Take that final (potentially quite large, you may need high mem node) hdf5 file and apply some formatting and statistics to create final output tables of astronomer-friendly predictions. See `d_all_predictions_to_friendly_table.py`.

To manage the configuration (paths, model options, etc) I use [hydra](https://hydra.cc/docs/intro/). This lets you write yaml config files which can then be mixed-and-matched. See the example below. You will want to make new config files under the `conf` subfolders. 

For a test example with GZCD that runs (set download=True on first run, and ask me for the required hidden `internal_urls.py` file):

    python example/make_snippets.py 

    python make_predictions/a_make_bulk_catalog_predictions.py +cluster=local_debug +galaxies=hsc_local_debug +model=maxvit_tiny_evo

    python make_predictions/b_group_hdf5_from_a_model.py +cluster=local_debug +galaxies=hsc_local_debug +model=maxvit_tiny_evo +aggregation=local_debug.yaml

    python make_predictions/c_group_hdf5_across_models.py +cluster=local_debug +galaxies=hsc_local_debug +model=maxvit_tiny_evo +aggregation=local_debug.yaml

    python make_predictions/d_all_predictions_to_friendly_table.py +cluster=local_debug +galaxies=hsc_local_debug +model=maxvit_tiny_evo +aggregation=local_debug.yaml

Apart from the conda.yaml requirements, you will need the very latest `zoobot` and `galaxy-datasets` packages from github (or pip, though that may be outdated quickly).
