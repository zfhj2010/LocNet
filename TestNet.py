import os
import cv2
import numpy as np
import tensorflow as tf
import os.path as osp
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from Config.Settings import cfg, get_output_and_tb_dir
from NetWorks.Vgg16 import vgg16
from NetWorks.ResNet_V1 import resnetv1
from Utils.Blob import im_list_to_blob
from Utils.NMSWrapper import nms
from Utils.MyNMS import zsoft_nms
from Utils.BBoxTransform import clip_boxes, bbox_transform_inv


def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def _get_blobs(im):
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def im_detect(sess, net, im):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

    _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

    boxes = rois[:, 1:5] / im_scales[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    return scores, pred_boxes


def test_process(mode, net_name, dataset, pretrained_model, image_path):
    output_dir, _ = get_output_and_tb_dir(net_name, dataset)
    model_file = osp.join(output_dir, pretrained_model)
    if not os.path.isfile(model_file + '.meta'):
        raise IOError('{:s} not found.\n'.format(model_file + '.meta'))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if net_name == 'vgg16':
        net = vgg16()
    elif net_name == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError

    # load data annotations
    fin_classes = []
    if dataset.startswith('coco'):
        ann_path = osp.join(cfg.DATA_DIR, 'coco', 'annotations', 'instances_val2017.json')
        _COCO = COCO(ann_path)
        cats = _COCO.loadCats(_COCO.getCatIds())
        coco_classes = tuple(['__background__'] + [c['name'] for c in cats])
        fin_classes = coco_classes
        net.create_architecture("TEST", len(coco_classes), tag='default', anchor_scales=[4, 8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    im = cv2.imread(image_path)
    scores, boxes = im_detect(sess, net, im)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    ind = [ind for ind, cls in enumerate(fin_classes) if cls == 'clock']
    if len(ind) > 0:
        cls_ind = ind[0]
    else:
        raise NotImplementedError
    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    # keep = nms(dets, NMS_THRESH)
    keep = zsoft_nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    real_inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    if len(real_inds) == 0:
        return []
    else:
        real_boxes = [dets[real_ind] for real_ind in real_inds]
    vis_detections(im, 'Meter', dets, CONF_THRESH)
    plt.show()
    return real_boxes


def vis_detections(im, class_name, dets, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

