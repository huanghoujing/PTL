from copy import deepcopy
from .datasets.market1501 import Market1501
from .datasets.cuhk03_np_detected_jpg import CUHK03NpDetectedJpg
from .datasets.duke import DukeMTMCreID
from .datasets.peta import PETA
from .datasets.rap import RAP
from .datasets.pa100k import PA100K


__factory = {
        'market1501': Market1501,
        'cuhk03_np_detected_jpg': CUHK03NpDetectedJpg,
        'duke': DukeMTMCreID,
        'peta': PETA,
        'rap': RAP,
        'pa100k': PA100K,
    }


dataset_shortcut = {
    'market1501': 'M',
    'cuhk03_np_detected_jpg': 'C',
    'duke': 'D',
    'rap': 'R',
    'peta': 'PETA',
    'pa100k': 'PA100K',
}


def create_dataset(cfg, samples=None):
    return __factory[cfg.name](cfg, samples=samples)
