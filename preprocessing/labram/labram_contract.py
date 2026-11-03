"""Shared HDF5 and model-input contract; importing the reader does not load Lance."""

import json

from preprocessing.common import preprocess_emovid_rules as prep
from preprocessing.labram.preprocess_lwh_labram import LABRAM_CHANNELS, POSITION

LAYOUT = 'emovid-labram-windows-v1'
DEFAULT_LABRAM_URI = 's3://personalized-fm/processed_labram'
PATCH_SAMPLES = 200
AMPLITUDE_DIVISOR = 100.0
ORDER = sorted(range(len(prep.CHANNELS)), key=lambda i: POSITION[prep.CHANNELS[i]])
CHANNEL_NAMES = [LABRAM_CHANNELS[POSITION[prep.CHANNELS[i]]] for i in ORDER]
INPUT_CHANS = [0] + [POSITION[name.upper()]+1 for name in CHANNEL_NAMES]



def validate_hdf5(dataset, *, n_windows=None, signature=None):
    if (dataset.attrs.get('layout') != LAYOUT or dataset.attrs.get('processing_status') != 'complete'
            or dataset.attrs.get('unit') != 'uV' or dataset.attrs.get('sampling_rate_hz') != 200
            or dataset.attrs.get('amplitude_divisor_at_model_input') != AMPLITUDE_DIVISOR
            or dataset.attrs.get('patch_samples') != PATCH_SAMPLES):
        raise ValueError('Invalid LaBraM HDF5 input contract')
    shape = dataset['data'].shape
    if len(shape) != 3 or shape[1:] != (len(CHANNEL_NAMES), prep.WINDOW_SAMPLES) or not shape[0]:
        raise ValueError('Invalid LaBraM waveform shape')
    if n_windows is not None and shape[0] != n_windows:
        raise ValueError('LaBraM window count disagrees with summary')
    if dataset['channel_names'].asstr()[:].tolist() != CHANNEL_NAMES or dataset['input_chans'][:].tolist() != INPUT_CHANS:
        raise ValueError('Invalid LaBraM channel position mapping')
    for key in ['labels', 'emotion_labels', 'qc_pass', 'sample_ids', 'subject_ids', 'session_ids', 'segment_type', 'rows_json']:
        if dataset[key].shape != (shape[0],):
            raise ValueError(f'Invalid LaBraM row field: {key}')
    metadata = json.loads(dataset.attrs['metadata_json'])
    if signature is not None and metadata.get('signature') != signature:
        raise ValueError('LaBraM provenance signature mismatch')
    return metadata
