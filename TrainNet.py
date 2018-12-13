import os
import glob
import numpy as np
import tensorflow as tf
from NetWorks.Vgg16 import vgg16
from NetWorks.ResNet_V1 import resnetv1
from RoiLayer.Layer import roidatalayer
from DataSet.Factory import get_imdb, get_roidb
from Config.Settings import cfg, get_output_and_tb_dir
from tensorflow.python import pywrap_tensorflow
from Utils.Timer import Timer

try:
    import cPickle as pickle
except ImportError:
    import pickle
import time


class SolverWrapper(object):
    def __init__(self, network, imdb, roidb, output_dir, tb_dir, pretrained_model):
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.tb_dir = tb_dir
        self.pretrained_model = os.path.join(cfg.DATA_DIR, 'weights', pretrained_model)

    def initialize(self, sess):
        # Initial file lists are empty
        np_paths = []
        ss_paths = []
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(self.pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
        # Get the variables to restore, ignoring the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, self.pretrained_model)
        print('Loaded.')
        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
        # fully connected weights
        self.net.fix_variables(sess, self.pretrained_model)
        print('Fixed.')
        last_snapshot_iter = 0
        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = list(cfg.TRAIN.STEPSIZE)

        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def restore(self, sess, sfile, nfile):
        # Get the most recent snapshot and restore
        np_paths = [nfile]
        ss_paths = [sfile]
        # Restore model from snapshots
        last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
        # Set the learning rate
        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            if last_snapshot_iter > stepsize:
                rate *= cfg.TRAIN.GAMMA
            else:
                stepsizes.append(stepsize)

        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def snapshot(self, sess, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()
        # current position in the database
        cur = self.data_layer._cur
        # current shuffled indexes of the database
        perm = self.data_layer._perm

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def from_snapshot(self, sess, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess, sfile)
        print('Restored.')
        # Needs to restore the other hyper-parameters/states for training
        # I have tried my best to find the random states so that it can be recovered exactly
        # However the Tensorflow state is currently not available
        with open(nfile, 'rb') as fid:
            st0 = pickle.load(fid)
            cur = pickle.load(fid)
            perm = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)

            np.random.set_state(st0)
            self.data_layer._cur = cur
            self.data_layer._perm = perm

        return last_snapshot_iter

    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)

        to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)

    def train_model(self, sess, max_iters):
        self.data_layer = roidatalayer(self.roidb, self.imdb.num_classes)
        lr, train_op = self.construct_graph(sess)
        lsf, nfiles, sfiles = self.find_previous()

        # Initialize the variables or restore them from the last snapshot
        if lsf == 0:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
        else:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess,
                                                                                   str(sfiles[-1]),
                                                                                   str(nfiles[-1]))
        timer = Timer()
        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        # Make sure the lists are not empty
        stepsizes.append(max_iters)
        stepsizes.reverse()
        next_stepsize = stepsizes.pop()
        while iter < max_iters + 1:
            # Learning rate
            if iter == next_stepsize + 1:
                # Add snapshot here before reducing the learning rate
                self.snapshot(sess, iter)
                rate *= cfg.TRAIN.GAMMA
                sess.run(tf.assign(lr, rate))
                next_stepsize = stepsizes.pop()

            timer.tic()
            # Get training data, one batch at a time
            blobs = self.data_layer.forward()

            now = time.time()
            if iter == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
                # Compute the graph with summary
                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
                    self.net.train_step_with_summary(sess, blobs, train_op)
                self.writer.add_summary(summary, float(iter))
                last_summary_time = now
            else:
                # Compute the graph without summary
                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
                    self.net.train_step(sess, blobs, train_op)
            timer.toc()

            # Display training information
            if iter % cfg.TRAIN.DISPLAY == 0:
                print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                      '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
                      (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval()))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            # Snapshotting
            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                ss_path, np_path = self.snapshot(sess, iter)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths, ss_paths)

            iter += 1

        if last_snapshot_iter != iter - 1:
            self.snapshot(sess, iter - 1)

        self.writer.close()

    def construct_graph(self, sess):
        with sess.graph.as_default():
            tf.set_random_seed(cfg.RNG_SEED)
            layers = self.net.create_architecture('TRAIN', self.imdb.num_classes, tag='default',
                                                  anchor_scales=cfg.ANCHOR_SCALES,
                                                  anchor_ratios=cfg.ANCHOR_RATIOS)
            loss = layers['total_loss']
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
            gvs = self.optimizer.compute_gradients(loss)
            if cfg.TRAIN.DOUBLE_BIAS:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult') as scope:
                    for grad, var in gvs:
                        scale = 1.
                        if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = self.optimizer.apply_gradients(final_gvs)
            else:
                train_op = self.optimizer.apply_gradients(gvs)

            self.saver = tf.train.Saver(max_to_keep=100000)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tb_dir, sess.graph)
        return lr, train_op

    def find_previous(self):
        sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redfiles = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            redfiles.append(os.path.join(self.output_dir,
                                         cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize + 1)))
        sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles


def gen_train_imdb_roidb(mode, dataset):
    print('load dataset {:s} for training'.format(dataset))
    imdb = get_imdb(mode, dataset)
    roidb = get_roidb(imdb)
    return imdb, roidb


def _is_valid_entry(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid


def filter_roidb(roidb):
    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if _is_valid_entry(entry)]
    return filtered_roidb


def train_net(net, imdb, roidb, output_dir, tb_dir, pretrained_mode=None, max_iters=35000):
    roidb = filter_roidb(roidb)
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(net, imdb, roidb, output_dir, tb_dir, pretrained_mode)
        print('training begin')
        sw.train_model(sess, max_iters)
        print('training done')


def train_process(mode, net_name, dataset, weight, iters):
    # prepare data for training
    imdb, roidb = gen_train_imdb_roidb(mode, dataset)
    # prepare dir for training
    output_dir, tb_dir = get_output_and_tb_dir(net_name, dataset)
    # create net architecture
    if net_name == 'vgg16':
        net = vgg16()
    elif net_name == 'res50':
        net = resnetv1(num_layers=50)
    elif net_name == 'res101':
        net = resnetv1(num_layers=101)
    elif net_name == 'res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError
    train_net(net, imdb, roidb, output_dir, tb_dir, weight, iters)
