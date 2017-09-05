import tensorflow as tf
import sudoku
from tensorflow.contrib import layers as tfl
from concurrent.futures import ThreadPoolExecutor
import numpy as np

tf.flags.DEFINE_integer('iterations', 30, 'number of iterations')
tf.flags.DEFINE_integer('batch_size', 100, 'batch size')
tf.flags.DEFINE_float('discount', .99, 'discount factor')
tf.flags.DEFINE_float('lr', 3e-4, 'learning rate')

tf.flags.DEFINE_integer('rep', 0, 'dummy repetition')
tf.flags.DEFINE_integer('empty', 1, 'number of empty fields')

flags = tf.flags.FLAGS

game_shape = (9, 9)
action_shape = (9, 9, 9)


def q(s):
    net = s
    net = tfl.conv2d(net, 100, 3)
    net = tfl.conv2d(net, 100, 3)

    net = tfl.conv2d(net, 9, 3, activation_fn=None)

    return net

def get_single_sample(i):
    s = sudoku.generate()
    s1 = sudoku.make_empty(s, flags.empty)
    s2, (x, y) = sudoku.random_move(s1)
    if sudoku.check_consistent(s2, (x, y)):
        r = 0.
        if sudoku.check_all_filled(s2):
            r = 1.
    else:
        r = -1.
    return s1, s2, r


threadPool = ThreadPoolExecutor()

def get_sample(batch_size):
    batch = threadPool.map(get_single_sample, range(batch_size))
    return list(zip(*batch))


def main():
    
    s1, s2 = (tf.placeholder(tf.int32, (None,) + game_shape) for _ in range(2))
    reward = tf.placeholder(tf.float32, (None,))
    final_step = tf.placeholder(tf.bool)
    s1_oh, s2_oh = tf.one_hot(s1, 10), tf.one_hot(s2, 10)
    s1_action_mask = (s2_oh - s1_oh)[:, :, :, 1:]
    s2_action_mask = tf.tile(tf.cast(s2_oh[:, :, :, :1] > 1e-9, tf.float32), (1, 1, 1, 9))

    q1, q2 = q(s1_oh), q(s2_oh)
    q1 = tf.stop_gradient(q1)
    q1, q2 = tfl.flatten(q1 * s1_action_mask), tfl.flatten(q2 * s2_action_mask)
    q1 = tf.reduce_sum(q1, axis=1)
    q2_max = tf.reduce_max(q2, axis=1)
    y = tf.cond(final_step, lambda: reward, lambda: reward + flags.discount * q2_max)
    q_loss = tf.reduce_mean((q1 - y)**2.)

    opt = tf.train.RMSPropOptimizer(flags.lr)
    train_op = opt.minimize(q_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(flags.iterations):
            batch_s1, batch_s2, batch_r = get_sample(flags.batch_size)
            feed_dict = {s1: batch_s1, s2: batch_s2, reward: batch_r, final_step: flags.empty == 1}
            _, ql = sess.run([train_op, q_loss], feed_dict=feed_dict)
            print(ql)
    

if __name__ == '__main__':
    main()
