import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 64

def main():
    tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    kb = tf.placeholder(tf.float32, [1])

    with tf.Session() as sess:
        output = convolution(x)
        loss_op = loss(output, y)
        train_op = train(loss_op)
        accuracy_op = accuracy(output, y)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(1500):
            images, labels = mnist.train.next_batch(64)
            _ = sess.run((train_op), feed_dict={x: images, y: labels, kb: [0.5]}) 
            if i % 10 == 0:
                _loss, _accuracy = sess.run((loss_op, accuracy_op), feed_dict={x: images, y: labels}) 
                print('global step: %04d, train loss: %01.7f, train accuracy %01.5f' % (i, _loss, _accuracy))
        _accuracy = sess.run(accuracy_op, feed_dict={x: mnist.test.images, y: mnist.test.labels, kb: [1.0]})
        print('Test accuracy:', _accuracy)


def convolution(images, keep_prob=0.5):
    batch_size = tf.shape(images)[0]
    images = tf.reshape(images, [batch_size, 28, 28, 1])

    output = tf.layers.conv2d(images, filters=16, kernel_size=[2, 2], strides=[2, 2], padding='SAME')
    output = tf.nn.relu(output)

    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.conv2d(images, filters=32, kernel_size=[2, 2], strides=[2, 2], padding='SAME')
    output = tf.nn.relu(output)

    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.conv2d(images, filters=32, kernel_size=[2, 2], strides=[2, 2], padding='SAME')
    output = tf.nn.relu(output)

    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.conv2d(images, filters=64, kernel_size=[2, 2], strides=[2, 2], padding='SAME')
    output = tf.nn.relu(output)

    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.conv2d(images, filters=64, kernel_size=[2, 2], strides=[2, 2], padding='SAME')
    output = tf.nn.relu(output)

    output = tf.nn.dropout(output, keep_prob=keep_prob)
    
    output = tf.contrib.layers.flatten(output)

    output = tf.layers.dense(output, 1024)
    output = tf.nn.relu(output)

    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.dense(output, 256)
    output = tf.nn.relu(output)

    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.dense(output, 10)
    output = tf.nn.softmax(output)
    return output


def loss(logits, labels):
    return tf.reduce_mean(-tf.log(logits) * labels)

def train(loss):
    return tf.train.AdamOptimizer().minimize(loss)


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == '__main__':
    main()