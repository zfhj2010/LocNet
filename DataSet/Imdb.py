import os
import PIL.Image
import os.path as osp
from Config.Settings import cfg


class imdb(object):
    def __init__(self, mode, dataset):
        self._dataset = dataset
        self._classes = []
        self._image_index = []
        self._roidb = None
        self._roidb_handler = self.default_roidb

    @property
    def dataset(self):
        return self._dataset

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def num_images(self):
        return len(self._image_index)

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb(self):
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def image_path_at(self, i):
        raise NotImplementedError

    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in range(self.num_images)]

    def append_flipped_images(self):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

