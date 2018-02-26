import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import random

tf.set_random_seed(42)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.01
training_epochs = 15 
batch_size = 100


def main():
    """
    Original paper: Gradient-Based Learning Applied to Document Recognition
    """
    with tf.variable_scope("input"):
        X = tf.placeholder(tf.float32, [None, 784])
        X_img = tf.reshape(X, [-1, 28, 28, 1])
        y = tf.placeholder(tf.float32, [None, 10])

    with tf.variable_scope("c1_s2"): # convolution->subsampling->squash function
        c1_w = tf.Variable(tf.random_normal([5, 5, 1, 6], stddev=0.01))
        # Convolution -> (?, 28, 28, 6)
        L1 = tf.nn.conv2d(X_img, c1_w, strides=[1, 1, 1, 1], padding="SAME")
        # Subsampling -> (?, 14, 14, 6)
        L1 = tf.nn.avg_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME") # similar to sum
        s2_w = tf.Variable(tf.random_normal([1, 1, 1, 6]))
        s2_b = tf.Variable(tf.random_normal([1, 1, 1, 6]))
        L1 = L1 * s2_w + s2_b
        # Squash
        L1 = tf.nn.sigmoid(L1) 

    with tf.variable_scope("c3_s4"):
        # Convolution -> (?, 10, 10, 16)
        c3_0 = tf.Variable(tf.random_normal([5, 5, 3, 1], stddev=0.01))
        c3_1 = tf.Variable(tf.random_normal([5, 5, 3, 1], stddev=0.01)) 
        c3_2 = tf.Variable(tf.random_normal([5, 5, 3, 1], stddev=0.01)) 
        c3_3 = tf.Variable(tf.random_normal([5, 5, 3, 1], stddev=0.01)) 
        c3_4 = tf.Variable(tf.random_normal([5, 5, 3, 1], stddev=0.01)) 
        c3_5 = tf.Variable(tf.random_normal([5, 5, 3, 1], stddev=0.01)) 
        c3_6 = tf.Variable(tf.random_normal([5, 5, 4, 1], stddev=0.01)) 
        c3_7 = tf.Variable(tf.random_normal([5, 5, 4, 1], stddev=0.01)) 
        c3_8 = tf.Variable(tf.random_normal([5, 5, 4, 1], stddev=0.01)) 
        c3_9 = tf.Variable(tf.random_normal([5, 5, 4, 1], stddev=0.01)) 
        c3_10 = tf.Variable(tf.random_normal([5, 5, 4, 1], stddev=0.01)) 
        c3_11 = tf.Variable(tf.random_normal([5, 5, 4, 1], stddev=0.01)) 
        c3_12 = tf.Variable(tf.random_normal([5, 5, 4, 1], stddev=0.01)) 
        c3_13 = tf.Variable(tf.random_normal([5, 5, 4, 1], stddev=0.01)) 
        c3_14 = tf.Variable(tf.random_normal([5, 5, 4, 1], stddev=0.01)) 
        c3_15 = tf.Variable(tf.random_normal([5, 5, 6, 1], stddev=0.01)) 
        # (?, 14, 14, 6) -> (?, 10, 10, 16) 
        L2_0 = tf.nn.conv2d(tf.boolean_mask(L1, [True, True, True, False, False, False], axis=3), c3_0, strides=[1, 1, 1, 1], padding="VALID") 
        L2_1 = tf.nn.conv2d(tf.boolean_mask(L1, [False, True, True, True, False, False], axis=3), c3_1, strides=[1, 1, 1, 1], padding="VALID")
        L2_2 = tf.nn.conv2d(tf.boolean_mask(L1, [False, False, True, True, True, False], axis=3), c3_2, strides=[1, 1, 1, 1], padding="VALID") 
        L2_3 = tf.nn.conv2d(tf.boolean_mask(L1, [False, False, False, True, True, True], axis=3), c3_3, strides=[1, 1, 1, 1], padding="VALID")
        L2_4 = tf.nn.conv2d(tf.boolean_mask(L1, [True, False, False, False, True, True], axis=3), c3_4, strides=[1, 1, 1, 1], padding="VALID") 
        L2_5 = tf.nn.conv2d(tf.boolean_mask(L1, [True, True, False, False, False, True], axis=3), c3_5, strides=[1, 1, 1, 1], padding="VALID")
        L2_6 = tf.nn.conv2d(tf.boolean_mask(L1, [True, True, True, True, False, False], axis=3), c3_6, strides=[1, 1, 1, 1], padding="VALID") 
        L2_7 = tf.nn.conv2d(tf.boolean_mask(L1, [False, True, True, True, True, False], axis=3), c3_7, strides=[1, 1, 1, 1], padding="VALID")
        L2_8 = tf.nn.conv2d(tf.boolean_mask(L1, [False, False, True, True, True, True], axis=3), c3_8, strides=[1, 1, 1, 1], padding="VALID") 
        L2_9 = tf.nn.conv2d(tf.boolean_mask(L1, [True, False, False, True, True, True], axis=3), c3_9, strides=[1, 1, 1, 1], padding="VALID")
        L2_10 = tf.nn.conv2d(tf.boolean_mask(L1, [True, True, False, False, True, True], axis=3), c3_10, strides=[1, 1, 1, 1], padding="VALID") 
        L2_11 = tf.nn.conv2d(tf.boolean_mask(L1, [True, True, True, False, False, True], axis=3), c3_11, strides=[1, 1, 1, 1], padding="VALID")
        L2_12 = tf.nn.conv2d(tf.boolean_mask(L1, [True, True, False, True, True, False], axis=3), c3_12, strides=[1, 1, 1, 1], padding="VALID") 
        L2_13 = tf.nn.conv2d(tf.boolean_mask(L1, [False, True, True, False, True, True], axis=3), c3_13, strides=[1, 1, 1, 1], padding="VALID")
        L2_14 = tf.nn.conv2d(tf.boolean_mask(L1, [True, False, True, True, False, True], axis=3), c3_14, strides=[1, 1, 1, 1], padding="VALID") 
        L2_15 = tf.nn.conv2d(tf.boolean_mask(L1, [True, True, True, True, True, True], axis=3), c3_15, strides=[1, 1, 1, 1], padding="VALID")

        L2 = tf.concat([L2_0, L2_1, L2_2, L2_3, L2_4, L2_5, L2_6, L2_7, L2_8, L2_9, L2_10, L2_11, L2_12, L2_13, L2_14, L2_15], axis=3)
        # L2 = tf.reshape(L2, [-1, 10, 10, 16]) ###???
        # Subsampling -> (?, 5, 5, 16)
        L2 = tf.nn.avg_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME") # similar to sum
        s4_w = tf.Variable(tf.random_normal([1, 1, 1, 16]))
        s4_b = tf.Variable(tf.random_normal([1, 1, 1, 16]))
        L2 = L2 * s4_w + s4_b
        # Squash
        L2 = tf.nn.sigmoid(L2)
        L2_flat = tf.reshape(L2, [-1, 16*5*5])

    with tf.variable_scope("c5_f6"):
        c5_w = tf.get_variable("c5_w", shape=[16*5*5, 84], initializer=tf.contrib.layers.xavier_initializer())
        c5_b = tf.Variable(tf.random_normal([84]))
        f6_w = tf.get_variable("f6_w", shape=[84, 10], initializer=tf.contrib.layers.xavier_initializer())
        f6_b = tf.Variable(tf.random_normal([10]))

        L3 = tf.matmul(L2_flat, c5_w) + c5_b 
        logits = tf.matmul(L3, f6_w) + f6_b

    with tf.variable_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        
    training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("Learning started.")
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                #print(i)
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # print(batch_xs.shape, batch_ys.shape)
                c, _ = sess.run([cost, training_op], feed_dict={X:batch_xs, y: batch_ys})
            # print("Epoch {}".format(epoch))
            print("Epoch {}, cost {}".format(epoch, c))

         # Test model and check accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, y:mnist.test.labels})) # 0.9713

        # Get one and predict
        r = random.randint(0, mnist.test.num_examples - 1)
        print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
        print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r+1]}))

if __name__ == "__main__":
    main()