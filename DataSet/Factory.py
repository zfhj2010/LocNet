from Config.Settings import cfg
from DataSet.Coco import coco

_Sets = {}


def get_imdb(mode, dataset):
    if dataset.startswith('coco'):
        if dataset not in _Sets:
            _Sets[dataset] = coco(mode, dataset)
    elif dataset.startswith('pascal_voc'):
        pass
    else:
        raise NotImplementedError
    return _Sets[dataset]


def get_roidb(imdb):
    if cfg.TRAIN.USE_FLIPPED:
        imdb.append_flipped_images()

