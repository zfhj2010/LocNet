import argparse
from DataSet.Factory import get_imdb, get_roidb

NETS = ('vgg16', 'res101')
DATASETS = ('coco2017', 'pascal_voc2007')


def parse_args():
    parser = argparse.ArgumentParser(description='Working on LocNet ')
    parser.add_argument('--net', dest='net', default='res101', type=str)
    parser.add_argument('--mode', dest='mode', default='train', type=str)
    parser.add_argument('--dataset', dest='dataset', default='coco2017', type=str)
    parser.add_argument('--weight', dest='weight', default=None, type=str)
    parser.add_argument('--iter', dest='iter', default=35000, type=int)
    args = parser.parse_args()
    return args


def gen_imdb_roidb(mode, dataset):
    imdb = get_imdb(mode, dataset)
    get_roidb(imdb)


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        gen_imdb_roidb(args.mode, args.dataset)



