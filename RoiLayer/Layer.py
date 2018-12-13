import numpy as np
from Config.Settings import cfg
from RoiLayer.MiniBatch import get_minibatch


class roidatalayer(object):
    def __init__(self, roidb, num_classes):
        self._roidb = roidb
        self._num_classes = num_classes
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def forward(self):
        blobs = self._get_next_minibatch()
        return blobs

    def _get_next_minibatch(self):
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes)

    def _get_next_minibatch_inds(self):
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH

        return db_inds
