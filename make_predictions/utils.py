import os

def get_model_name(checkpoint_loc):
    return os.path.basename(os.path.dirname(os.path.dirname(checkpoint_loc)))

