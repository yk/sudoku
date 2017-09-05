import tensorflow as tf
import sudoku
from tensorflow.contrib import layers as tfl
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import data

tf.flags.DEFINE_integer('iterations', 30, 'number of iterations')
tf.flags.DEFINE_integer('batch_size', 100, 'batch size')
tf.flags.DEFINE_float('discount', .99, 'discount factor')
tf.flags.DEFINE_float('lr', 3e-4, 'learning rate')
tf.flags.DEFINE_float('epsilon', .1, 'exploration epsilon')

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
    s_diff_action_mask = (s2_oh - s1_oh)[:, :, :, 1:]
    s1_action_mask = tf.tile(tf.cast(s1_oh[:, :, :, :1] > 1e-9, tf.float32), (1, 1, 1, 9))
    s2_action_mask = tf.tile(tf.cast(s2_oh[:, :, :, :1] > 1e-9, tf.float32), (1, 1, 1, 9))

    q1, q2 = q(s1_oh), q(s2_oh)
    q1 = tf.stop_gradient(q1)
    q1_masked, q2 = tfl.flatten(q1 * s_diff_action_mask), tfl.flatten(q2 * s2_action_mask)
    q1_masked = tf.reduce_sum(q1_masked, axis=1)
    q2_max = tf.reduce_max(q2, axis=1)
    y = tf.cond(final_step, lambda: reward, lambda: reward + flags.discount * q2_max)
    q_loss = tf.reduce_mean((q1_masked - y)**2.)

    opt = tf.train.RMSPropOptimizer(flags.lr)
    train_op = opt.minimize(q_loss)

    dataset = data.load()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(flags.iterations):
            begin_batch = (i * flags.batch_size) % len(dataset)
            end_batch = ((i+1) * flags.batch_size) % len(dataset)
            if end_batch < begin_batch:
                np.random.shuffle(dataset)
                begin_batch = 0
                end_batch = flags.batch_size
            batch = dataset[begin_batch : end_batch]

            batch = np.asarray([sudoku.make_empty(b, flags.empty) for b in batch])

            for e_step in range(flags.empty):
                q1_val, = sess.run([q1], feed_dict={s1: batch})

                batch_next = []
                rewards = []

                for b, qv in zip(batch, q1_val):
                    qv += np.min(qv)
                    qv = qv * (b > 0)
                    arg_max_q = np.argmax(qv.ravel()) // 9
                    if np.random.rand() < flags.epsilon:
                        arg_max_q = np.random.randint(81)
                    x, y = arg_max_q // 9, arg_max_q % 9
                    b = np.copy(b)
                    b[x, y] = np.argmax(qv[x, y]) + 1
                    batch_next.append(b)
                    if sudoku.check_consistent(b, (x, y)):
                        r = 0.
                        if sudoku.check_all_filled(b):
                            r = 1.
                    else:
                        r = -1.
                    rewards.append(r)

                batch_next = np.asarray(batch_next)
                rewards = np.asarray(rewards)

                _, ql = sess.run([train_op, q_loss], feed_dict={s1: batch, s2: batch_next, reward: rewards, final_step: e_step == flags.empty - 1})
                print(ql)

                batch = batch_next
    

if __name__ == '__main__':
    main()
