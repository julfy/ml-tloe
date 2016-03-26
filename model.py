import tensorflow.python.platform
import tensorflow as tf
import numpy as np
import math

LOGDIR = '/home/bogdan/work/repos/ml-tloe/run/summary'

# NUM_CLASSES = 10
# num_inputs = 31

training_epochs = 750
display_step = 100

num_inputs = 41
NUM_CLASSES = 1

P1 = 10

with tf.Graph().as_default():

    def create (H1, H2):
        global x, y, h11w, h12w, h13w, h14w, h15w, h1b, h2b, h3b, h4b, h5b, \
            p11w, p12w, p13w, p14w, p15w, p1b, p2b, p3b, p4b, p5b, h2w, h2b, outw, smb
        x = tf.placeholder("float", [None, num_inputs])
        y = tf.placeholder("float", [None, NUM_CLASSES])

        h11w = tf.Variable(tf.truncated_normal([num_inputs, H1], stddev = 1.0 / math.sqrt(float(num_inputs))), name="h11w")
        h1b = tf.Variable(tf.zeros([H1]), name='h1b')

        h2w = tf.Variable(tf.truncated_normal([H1,H2], stddev = 1.0 / math.sqrt(float(H1))), name="h2w")
        h2b = tf.Variable(tf.zeros([H2]), name='h2b')

        outw = tf.Variable(tf.truncated_normal([H2, NUM_CLASSES], stddev = 1.0 / math.sqrt(float(H2))), name="outw")
        smb = tf.Variable(tf.zeros([NUM_CLASSES]), name='outb')

    def train (dataset, learning_rate, batch_size, lmbda, ermul, threshold, save=0, name='saved', restore=False, train = True):

        saver = tf.train.Saver()

        # GRAPH

        def feed(_x):
            h11 = tf.nn.relu (tf.matmul(_x, h11w) + h1b, name='h11')
            h2 = tf.nn.relu (tf.matmul(h11, h2w) + h2b, name = 'h2')
            out = tf.sigmoid (tf.matmul(h2, outw) + smb, name = 'out')
            return out

        normed = tf.nn.l2_normalize(x,1)
        pred = feed(normed)
        sqr_dif = tf.square(y - pred)
        sda = sqr_dif * (1 + y * (ermul-1 if ermul > 1 else 0))
        cost = tf.reduce_sum(sda) # + (tf.reduce_sum(tf.square(h1w)) +
                                  #    tf.reduce_sum(tf.square(h2w)) +
                                  #    tf.reduce_sum(tf.square(outw))) * lmbda / 2 / dataset.train.num_examples

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        tf.scalar_summary('cost', cost)
        # pollbias = tf.reduce_sum(p1b)
        # tf.scalar_summary('biases', pollbias)

        init = tf.initialize_all_variables()

        merged_summary_op = tf.merge_all_summaries()

        #execution
        with tf.Session() as sess:
            sess.run(init)
            if restore:
                print "Restoring ", tf.train.latest_checkpoint('.')
                saver.restore(sess, tf.train.latest_checkpoint('.'))
            summary_writer = tf.train.SummaryWriter(LOGDIR, graph_def=sess.graph_def)

            #training
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = dataset.train.num_batch(batch_size)

                for i in range(total_batch):
                    batch_xs, batch_ys = dataset.train.next_batch(batch_size)
                    e, c , _ = sess.run([normed, cost, optimizer], feed_dict={x: batch_xs, y: batch_ys})
                    # print e
                    avg_cost += c/total_batch

                if epoch % display_step == 0:
                    print "Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost)

                summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
                summary_writer.add_summary(summary_str, epoch*total_batch + i)
                if save > 0 and epoch % save == 0:
                    saver.save(sess, name, global_step = epoch)
            saver.save(sess, name, global_step = epoch)
            print "Optimization Finished!"

            correct_prediction_p = tf.logical_and(tf.less_equal(pred, threshold), tf.less_equal(y, threshold))
            correct_prediction_n = tf.logical_and(tf.greater(pred, threshold), tf.greater(y, threshold))
            accuracy_p = tf.reduce_sum(tf.cast(correct_prediction_p, 'float')) / tf.reduce_sum(tf.cast(tf.less_equal(y, threshold), 'float'))
            accuracy_n = tf.reduce_sum(tf.cast(correct_prediction_n, 'float')) / tf.reduce_sum(tf.cast(tf.greater(y, threshold), 'float'))

            t_acc_p = accuracy_p.eval({x: dataset.train.inputs, y: dataset.train.labels})
            t_acc_n = accuracy_n.eval({x: dataset.train.inputs, y: dataset.train.labels})
            print "Train Accuracy:", t_acc_p, " ", t_acc_n

            v_acc_p = accuracy_p.eval({x: dataset.validation.inputs, y: dataset.validation.labels})
            v_acc_n = accuracy_n.eval({x: dataset.validation.inputs, y: dataset.validation.labels})
            print "Validation Accuracy:", v_acc_p, " ", v_acc_n
            return float(1.0 - v_acc_n * v_acc_p)

    def run (dataset, threshold):
        saver = tf.train.Saver()

        # GRAPH

        def feed(_x):
            h11 = tf.nn.relu (tf.matmul(_x, h11w) + h1b, name='h11')
            h2 = tf.nn.relu (tf.matmul(h11, h2w) + h2b, name = 'h2')
            out = tf.sigmoid (tf.matmul(h2, outw) + smb, name = 'out')
            return out

        normed = tf.nn.l2_normalize(x,1)
        pred = feed(normed)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            print "Restoring ", tf.train.latest_checkpoint('.')
            saver.restore(sess, tf.train.latest_checkpoint('.'))

            correct_prediction_p = tf.logical_and(tf.less_equal(pred, threshold), tf.less_equal(y, threshold))
            correct_prediction_n = tf.logical_and(tf.greater(pred, threshold), tf.greater(y, threshold))
            accuracy_p = tf.reduce_sum(tf.cast(correct_prediction_p, 'float')) / tf.reduce_sum(tf.cast(tf.less_equal(y, threshold), 'float'))
            accuracy_n = tf.reduce_sum(tf.cast(correct_prediction_n, 'float')) / tf.reduce_sum(tf.cast(tf.greater(y, threshold), 'float'))

            t_acc_p = accuracy_p.eval({x: dataset.train.inputs, y: dataset.train.labels})
            t_acc_n = accuracy_n.eval({x: dataset.train.inputs, y: dataset.train.labels})
            print "Train Accuracy:", t_acc_p, " ", t_acc_n

            v_acc_p = accuracy_p.eval({x: dataset.validation.inputs, y: dataset.validation.labels})
            v_acc_n = accuracy_n.eval({x: dataset.validation.inputs, y: dataset.validation.labels})
            print "Validation Accuracy:", v_acc_p, " ", v_acc_n
            return float(1.0 - v_acc_n * v_acc_p)
