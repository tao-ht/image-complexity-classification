import tensorflow as tf
from ResNet import ResNet
import argparse
import os

"""configuration"""
def parse_args():
    desc = "Tensorflow implementation of ResNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='myself', help='myself')

    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')

    return parser.parse_args()

"""main"""
def main():

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        cnn = ResNet(sess, args)
        cnn.build_model()
        cnn.test()

if __name__ == '__main__':
    main()