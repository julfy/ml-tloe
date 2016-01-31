import tensorflow.python.platform
import tensorflow as tf
import input_data
import math

NUM_CLASSES = 10
num_inputs = 31

H1 = 50
H2 = 20
training_epochs = 5
display_step = 1

def train_once (dataset,learning_rate, batch_size, lmbda, H1, H2, num_inputs, NUM_CLASSES):
    # description
    x = tf.placeholder("float", [None, num_inputs])
    y = tf.placeholder("float", [None, NUM_CLASSES])

    h1w = tf.Variable(tf.truncated_normal([num_inputs, H1], stddev = 1.0 / math.sqrt(float(num_inputs))), name="h1w")
    h2w = tf.Variable(tf.truncated_normal([H1,H2], stddev = 1.0 / math.sqrt(float(H1))), name="h2w")
    outw = tf.Variable(tf.truncated_normal([H2, NUM_CLASSES], stddev = 1.0 / math.sqrt(float(H2))), name="softmax")
    h1b = tf.Variable(tf.zeros([H1]), name='h1b')
    h2b = tf.Variable(tf.zeros([H2]), name='h2b')
    smb = tf.Variable(tf.zeros([NUM_CLASSES]), name='smb')
    
    def feed(_x):
        h1 = tf.nn.softplus(tf.matmul(_x, h1w) + h1b)
        h2 = tf.nn.softplus(tf.matmul(h1, h2w) + h2b)
        out = tf.matmul(h2, outw) + smb
        return out

    pred = feed(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred, y)
    cost = tf.reduce_mean(cross_entropy) + (tf.reduce_sum(tf.square(h1w)) +
                                            tf.reduce_sum(tf.square(h2w)) +
                                            tf.reduce_sum(tf.square(outw))) * lmbda / 2 / dataset.train.num_examples
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    tf.scalar_summary('cost', cost)

    init = tf.initialize_all_variables()
    
    merged_summary_op = tf.merge_all_summaries()

    #execution
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.train.SummaryWriter('/home/julfy/work/tensorflow/mnist', graph_def=sess.graph_def)
        #training
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(dataset.train.num_examples/batch_size)
            
            for i in range(total_batch):
                batch_xs, batch_ys = dataset.train.next_batch(batch_size)
                c , _ = sess.run([cost, optimizer], feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += c/total_batch
                
            if epoch % display_step == 0:
                summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
                summary_writer.add_summary(summary_str, epoch*total_batch + i)
                print "Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost)
                
        print "Optimization Finished!"
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        tf.reduce_mean(tf.cast(correct_prediction, "float"))
        acc = accuracy.eval({x: dataset.validation.images, y: dataset.validation.labels})
        print "Accuracy:", acc
        return 1.0 - acc
    
