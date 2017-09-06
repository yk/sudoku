import tensorflow as tf
import sudoku
from tensorflow.contrib import layers as tfl
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import data
import time
import os
from sudoku import SIDE, SIDE_ROOT

tf.flags.DEFINE_integer('iterations', 30, 'number of iterations')
tf.flags.DEFINE_integer('batch_size', 100, 'batch size')
tf.flags.DEFINE_float('discount', .99, 'discount factor')
tf.flags.DEFINE_float('lr', 3e-4, 'learning rate')
tf.flags.DEFINE_float('epsilon', .1, 'exploration epsilon')

tf.flags.DEFINE_integer('rep', 0, 'dummy repetition')
tf.flags.DEFINE_integer('empty', 1, 'number of empty fields')
tf.flags.DEFINE_integer('channels_l1', 50, 'number of channels in first layer')
tf.flags.DEFINE_integer('copy_critic_steps', 1, 'number of steps before critic is updated')
tf.flags.DEFINE_integer('test_after', 100, 'number of iterations before test')

flags = tf.flags.FLAGS

game_shape = (SIDE, SIDE)
action_shape = (SIDE, SIDE, SIDE)


def q(s):
    net = s
    blocks = []
    blocks.append(tfl.conv2d(net, flags.channels_l1, [SIDE_ROOT, SIDE_ROOT], stride=SIDE_ROOT, padding='VALID'))
    blocks.append(tfl.conv2d(net, flags.channels_l1, [1, SIDE], padding='VALID'))
    blocks.append(tfl.conv2d(net, flags.channels_l1, [SIDE, 1], padding='VALID'))
    # blocks.append(tfl.conv2d(net, flags.channels_l1, 5))
    net = tf.concat([tfl.flatten(b) for b in blocks], -1)

    net = tfl.fully_connected(net, SIDE*SIDE*SIDE)

    net = tf.reshape(net, (-1, SIDE, SIDE, SIDE))

    return net


def main():
    s1, s2 = (tf.placeholder(tf.int32, (None,) + game_shape) for _ in range(2))

    reward = tf.placeholder(tf.float32, (None,))
    e_step_var = tf.placeholder(tf.int32)

    for e in range(1, flags.empty + 1):
        tf.summary.scalar('empty_{}_num_samples'.format(e), tf.shape(s1)[0], collections=['summary_empty_{}'.format(e)])

    s1_oh, s2_oh = tf.one_hot(s1, SIDE+1), tf.one_hot(s2, SIDE+1)
    s_diff_action_mask = (s2_oh - s1_oh)[:, :, :, 1:]
    s2_action_mask = tf.tile(tf.cast(s2_oh[:, :, :, :1] > 1e-9, tf.float32), (1, 1, 1, SIDE))
    for e in range(1, flags.empty + 1):
        tf.summary.scalar('empty_{}_mean_reward'.format(e), tf.reduce_mean(reward), collections=['summary_empty_{}'.format(e)])

    with tf.variable_scope('q1'):
        q1 = q(s1_oh)
    with tf.variable_scope('q2'):
        q2 = q(s2_oh)
    q2 = tf.stop_gradient(q2)
    q1_masked, q2 = tfl.flatten(q1 * s_diff_action_mask), tfl.flatten(q2 * s2_action_mask)
    q1_masked = tf.reduce_sum(q1_masked, axis=1)
    q2_max = tf.reduce_max(q2, axis=1)
    y = reward + tf.cast(tf.not_equal(e_step_var, tf.constant(1, tf.int32)), tf.float32) * flags.discount * q2_max
    q_loss = tf.reduce_mean((q1_masked - y)**2.)
    for e in range(1, flags.empty + 1):
        tf.summary.scalar('empty_{}_q_loss'.format(e), q_loss, collections=['summary_empty_{}'.format(e)])

    q1_vars = [v for v in tf.trainable_variables() if v.name.startswith('q1/')]
    q2_vars = [v for v in tf.trainable_variables() if v.name.startswith('q2/')]

    copy_critic_op = tf.group(*[tf.assign(q2v, q1v) for q2v, q1v in zip(q2_vars, q1_vars)])

    opt = tf.train.RMSPropOptimizer(flags.lr)
    train_op = opt.minimize(q_loss, var_list=q1_vars)
    summary_ops = [tf.summary.merge_all('summary_empty_{}'.format(e)) for e in range(1, flags.empty + 1)]

    dataset = data.load()

    with tf.Session() as sess:
        log_dir = 'logs/{}'.format(int(time.time()))
        summary_writer = tf.summary.FileWriter(log_dir)
        saver = tf.train.Saver(max_to_keep=1)
        sess.run(tf.global_variables_initializer())
        global_step = 0
        for empty in range(1, flags.empty + 1):
            print('empty', empty)
            np.random.shuffle(dataset)
            for i in range(flags.iterations):
                if i % 10 == 0:
                    print('iteration', i)
                begin_batch = (i * flags.batch_size) % len(dataset)
                end_batch = ((i+1) * flags.batch_size) % len(dataset)
                if end_batch <= begin_batch:
                    np.random.shuffle(dataset)
                    begin_batch = 0
                    end_batch = flags.batch_size
                batch = dataset[begin_batch : end_batch]

                batch = np.asarray([sudoku.make_empty(b, empty) for b in batch])

                for e_step in range(empty):
                    e_step = empty - e_step
                    if global_step % flags.copy_critic_steps == 0:
                        sess.run([copy_critic_op])
                    q1_val, = sess.run([q1], feed_dict={s1: batch})

                    batch_prev = []
                    batch_next = []
                    rewards = []

                    for b, qv in zip(batch, q1_val):
                        if not sudoku.check_consistent(b):
                            continue
                        batch_prev.append(b)

                        explore = np.random.rand() < flags.epsilon
                        if explore:
                            qv = np.random.rand(*(qv.shape))
                        qv += np.min(qv)
                        qv = qv * (b > 0)
                        arg_max_q = np.argmax(qv.ravel()) // SIDE
                        x, y = arg_max_q // SIDE, arg_max_q % SIDE
                        b = np.copy(b)
                        new_b_val = np.argmax(qv[x, y]) + 1
                        if explore:
                            new_b_val = np.random.randint(SIDE) + 1
                        b[x, y] = new_b_val
                        batch_next.append(b)
                        if sudoku.check_consistent(b, (x, y)):
                            r = 0.
                            if sudoku.check_all_filled(b):
                                r = 1.
                        else:
                            r = -1.
                        rewards.append(r)

                    if len(batch_prev) == 0:
                        continue

                    batch_prev = np.asarray(batch_prev)
                    batch_next = np.asarray(batch_next)
                    rewards = np.asarray(rewards)

                    ops_to_run = [train_op]
                    summary_op = summary_ops[e_step - 1]
                    ops_to_run.append(summary_op)

                    run_res = sess.run(ops_to_run, feed_dict={s1: batch_prev, s2: batch_next, reward: rewards, e_step_var: e_step})
                    if summary_op in ops_to_run:
                        summ_str = run_res[1]
                        summary_writer.add_summary(summ_str, global_step)
                    global_step += 1

                    batch = batch_next
        
        saver.save(sess, os.path.join(log_dir, 'model.ckpt'))
    

if __name__ == '__main__':
    main()
