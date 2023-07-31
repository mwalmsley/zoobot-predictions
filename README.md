# zoobot-predictions
Make arbitrary-scale (100k+ galaxies) predictions (inc. representations) from Zoobot


For a test example with GZCD (set download=True on first run)

    python example/make_snippets.py 

    python make_predictions/a_make_bulk_catalog_predictions.py +cluster=local_debug +galaxies=hsc_local_debug +model=maxvit_tiny_evo

    python make_predictions/b_group_hdf5_from_a_model.py +cluster=local_debug +galaxies=hsc_local_debug +model=maxvit_tiny_evo +aggregation=local_debug.yaml

    python make_predictions/c_group_hdf5_across_models.py +cluster=local_debug +galaxies=hsc_local_debug +model=maxvit_tiny_evo +aggregation=local_debug.yaml

    python make_predictions/d_all_predictions_to_friendly_table.py +cluster=local_debug +galaxies=hsc_local_debug +model=maxvit_tiny_evo +aggregation=local_debug.yaml