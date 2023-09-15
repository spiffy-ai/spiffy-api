import os
from spiffy.api import (
    acreate_model,
    aupload_training_data,
    atrain,
    aget_train_status,
    aget_available_user_models,
    agenerate,
    TrainingStatus,
    agenerate_streamed,
)

__version__ = '0.1.0'

via_blobr = True
api_key = os.environ.get("SPIFFY_API_KEY")
api_base_train = "https://api.spiffy.ai/api"
api_base_infer = "https://prod.svcs.spiffy.ai/"


__all__ = (
    'api_key',
    'api_base_train',
    'api_base_infer',
    'acreate_model',
    'aupload_training_data',
    'atrain',
    'aget_train_status',
    'aget_available_user_models',
    'agenerate',
    'TrainingStatus',
    'agenerate_streamed'
)
