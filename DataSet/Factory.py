from Config.Settings import cfg
from DataSet.Coco import coco
from RoiLayer.Roidb import prepare_roidb

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
        print('append horizontally-flipped training data')
        imdb.append_flipped_images()
        print('append done')
    print('prepare training data')
    prepare_roidb(imdb)
    print('prepare done')
    return imdb.roidb

