import os
import numpy as np
import os.path as osp
from easydict import EasyDict as edict

__C = edict()

cfg = __C
__C.TRAIN = edict()
__C.TRAIN.USE_FLIPPED = True

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0001

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False
__C.TRAIN.TRUNCATED = False

__C.TRAIN.LEARNING_RATE = 0.001
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.DOUBLE_BIAS = True

__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)

__C.TRAIN.FG_FRACTION = 0.25
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
__C.TRAIN.RPN_NMS_THRESH = 0.7

__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
__C.TRAIN.RPN_FG_FRACTION = 0.5
__C.TRAIN.RPN_BATCHSIZE = 256

__C.TRAIN.BATCH_SIZE = 128
__C.TRAIN.IMS_PER_BATCH = 1

__C.TRAIN.SCALES = (600,)
__C.TRAIN.MAX_SIZE = 1000

__C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'
__C.TRAIN.STEPSIZE = [30000]
__C.TRAIN.DISPLAY = 10

__C.TRAIN.GAMMA = 0.1
__C.TRAIN.PROPOSAL_METHOD = 'gt'
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

__C.TRAIN.USE_GT = False
__C.TRAIN.USE_ALL_GT = True

__C.TRAIN.SUMMARY_INTERVAL = 180
__C.TRAIN.RPN_CLOBBER_POSITIVES = False

__C.TRAIN.SNAPSHOT_ITERS = 5000
__C.TRAIN.SNAPSHOT_KEPT = 3

__C.TEST = edict()
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
__C.TEST.RPN_POST_NMS_TOP_N = 300
__C.TEST.RPN_NMS_THRESH = 0.7
__C.TEST.MODE = 'nms'
__C.TEST.SCALES = (600,)
__C.TEST.MAX_SIZE = 1000
__C.TEST.BBOX_REG = True

__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

__C.RPN_CHANNELS = 512
__C.USE_E2E_TF = True
# __C.ANCHOR_SCALES = [8, 16, 32]
__C.ANCHOR_SCALES = [4, 8, 16, 32]
__C.ANCHOR_RATIOS = [0.5, 1, 2]

__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.RNG_SEED = 3
__C.POOLING_MODE = 'crop'
__C.POOLING_SIZE = 7

__C.USE_GPU_NMS = True

__C.RESNET = edict()
__C.RESNET.MAX_POOL = False
__C.RESNET.FIXED_BLOCKS = 1


def get_output_and_tb_dir(net_name, dataset_name):
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', net_name, dataset_name))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    tbdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', net_name, dataset_name))
    if not os.path.exists(tbdir):
        os.makedirs(tbdir)
    return outdir, tbdir


