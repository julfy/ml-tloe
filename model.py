import tensorflow.python.platform
import tensorflow as tf
import numpy as np
import math

# NUM_CLASSES = 10
# num_inputs = 31

training_epochs = 750
display_step = 749

num_inputs = 31
NUM_CLASSES = 1

def train_once (dataset,learning_rate, batch_size, lmbda, ermul, H1, H2):
    
    # description
    with tf.Graph().as_default():
        x = tf.placeholder("float", [None, num_inputs])
        y = tf.placeholder("float", [None, NUM_CLASSES])
        
        h1w = tf.Variable(tf.truncated_normal([num_inputs, H1], stddev = 1.0 / math.sqrt(float(num_inputs))), name="h1w")
        h2w = tf.Variable(tf.truncated_normal([H1,H2], stddev = 1.0 / math.sqrt(float(H1))), name="h2w")
        outw = tf.Variable(tf.truncated_normal([H2, NUM_CLASSES], stddev = 1.0 / math.sqrt(float(H2))), name="softmax")
        h1b = tf.Variable(tf.zeros([H1]), name='h1b')
        h2b = tf.Variable(tf.zeros([H2]), name='h2b')
        smb = tf.Variable(tf.zeros([NUM_CLASSES]), name='smb')
    
        def feed(_x):
            h1 = tf.nn.relu (tf.matmul(_x, h1w) + h1b)
            h2 = tf.nn.relu (tf.matmul(h1, h2w) + h2b)
            out = tf.sigmoid (tf.matmul(h2, outw) + smb)
            return out

        normed = tf.nn.l2_normalize(x,1)
        pred = feed(normed)
        sqr_dif = tf.square(y - pred)
        sda = sqr_dif * (1 + y * (ermul-1 if ermul > 1 else 0))
        cost = tf.reduce_sum(sda) + (tf.reduce_sum(tf.square(h1w)) +
                                     tf.reduce_sum(tf.square(h2w)) +
                                     tf.reduce_sum(tf.square(outw))) * lmbda / 2 / dataset.train.num_examples
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        
        # tf.scalar_summary('x_entropy', cross_entropy)
        tf.scalar_summary('cost', cost)
        
        init = tf.initialize_all_variables()
    
        merged_summary_op = tf.merge_all_summaries()

        #execution
        with tf.Session() as sess:
            sess.run(init)
            summary_writer = tf.train.SummaryWriter('/home/julfy/work/ml-tloe/summary', graph_def=sess.graph_def)
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
                
            print "Optimization Finished!"
            correct_prediction_p = tf.cast(tf.equal(tf.round(pred), y),'float') * y
            correct_prediction_n = tf.cast(tf.equal(tf.round(pred), y),'float') * (1 - y)
            accuracy_p = tf.reduce_sum(tf.cast(correct_prediction_p, 'float')) / tf.reduce_sum(y)
            accuracy_n = tf.reduce_sum(tf.cast(correct_prediction_n, 'float')) / tf.reduce_sum(1 - y)
            v_acc_p = accuracy_p.eval({x: dataset.validation.inputs, y: dataset.validation.labels})
            v_acc_n = accuracy_n.eval({x: dataset.validation.inputs, y: dataset.validation.labels})

            t_acc_p = accuracy_p.eval({x: dataset.train.inputs, y: dataset.train.labels})
            t_acc_n = accuracy_n.eval({x: dataset.train.inputs, y: dataset.train.labels})
            print "Train Accuracy:", t_acc_p, " ", t_acc_n
            
            print "Accuracy:", v_acc_p, " ", v_acc_n
            return float(1.0 - v_acc_n * v_acc_p)
    
