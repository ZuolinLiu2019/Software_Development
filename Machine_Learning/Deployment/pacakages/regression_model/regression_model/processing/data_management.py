import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from regression_model.config import config
from regression_model import __version__ as _version

import logging
import typing as t
import os


_logger = logging.getLogger(__name__)

def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return _data

def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline.

    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # prepare versioned save file name
    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = os.path.join(config.TRAINED_MODEL_DIR, save_file_name)

    remove_old_pipeline(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f'saved pipeline: {save_file_name}')

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persistedd pipeline."""

    file_path = os.path.join(config.TRAINED_MODEL_DIR, file_name)
    trained_model = joblib.load(filename=file_path)
    return trained_model

def remove_old_pipeline(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipeline.

    This is to ensure there is a simple one-to_one
    mapping between the package version and the model
    version to be imported and used by other applications.
    However, we do also include the immediate previous
    pipeline version differential testing purpose.
    """

    do_not_delete = files_to_keep + ['__init__.py']
    # print(config.TRAINED_MODEL_DIR)
    for model_file in os.listdir(config.TRAINED_MODEL_DIR):
        # print(model_file)
        if model_file not in do_not_delete:
            model_file.unlink()
        # pass
