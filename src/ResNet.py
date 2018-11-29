from ops import *
from utils import *
import os

class ResNet(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset

        if self.dataset_name == 'myself':
            test_image_dir = ["test1.png", "test2.png", "test3.png"]
            """测试使用照片,test_img_dir 存放在列表中"""
            self.test_x = load_myself(test_image_dir)
            self.img_size = 200
            self.c_dim = 1
            self.label_dim = 3

        self.checkpoint_dir = args.checkpoint_dir
        self.res_n = 18
        self.batch_size = len(self.test_x)
        self.init_lr = args.lr


    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):
            residual_block = resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]):
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]):
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]):
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]):
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################

            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            x = fully_conneted(x, units=self.label_dim, scope='logit')

            return x


    def build_model(self):

        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.test_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='test_inputs')
        self.predict_labels = tf.placeholder(tf.float32, self.batch_size, name='predict_labels')
        self.probability = tf.placeholder(tf.float32, self.batch_size, name='probability')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits = self.network(self.train_inptus)
        self.test_logits = self.network(self.test_inptus, is_training=False, reuse=True)

        self.predict_labels, self.probability= classification_loss(logit=self.test_logits)

    def load(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.load(self.checkpoint_dir)

        test_feed_dict = {
            self.test_inptus: self.test_x,
        }

        predict_labels, probability = self.sess.run([self.predict_labels, self.probability], feed_dict=test_feed_dict)

        for i in range(len(predict_labels)):
            print("image", i+1, ": predict_labels: ", predict_labels[i], ",probability :%.4f"% (max(probability[i])))






