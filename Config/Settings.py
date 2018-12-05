import os.path as osp
from easydict import EasyDict as edict

__C = edict()

cfg = __C
__C.TRAIN = edict()
__C.TRAIN.USE_FLIPPED = True

__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

