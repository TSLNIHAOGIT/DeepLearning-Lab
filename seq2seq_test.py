
#https://blog.csdn.net/wangyangzhizhou/article/details/77977655
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = 20
batch_size = 9


def random_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower, high=vocab_upper, size=random_length()).tolist()
            for _ in range(batch_size)
            ]
####
# length_from=2, length_to=6
# 表示序列长度从多少到多少:最小2，最大6

# vocab_lower=2, vocab_upper=9
#表示数值大小从多少到多少

#batch_size=9
# 表示一批的有多少个

'''
[list([7, 2]) list([2, 5]) list([6, 3, 8]) list([6, 6, 5, 6, 8])
 list([7, 4]) list([3, 6, 5, 3, 3])]
'''

batches = random_sequences(length_from=2, length_to=6,
                           vocab_lower=2, vocab_upper=9,
                           batch_size=batch_size)



'''
生成的随机序列的长度是不一样的，需要对短的序列用来填充，而可设为0，取最长的序列作为每个序列的长度，不足的填充，然后再转换成time major形式
'''
def make_batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)


    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element
    print('inputs_batch_major', inputs_batch_major)
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)#第0轴和第1轴元素互换
    '''
    X_in=(6steps,9batch,hidden)
    '''
    return inputs_time_major, sequence_lengths

for each in batches:
    print('each\n',each)
    print('make_batch(each)\n',make_batch(each)[0])
    break

'''
each
 [[8, 2, 5], [4, 4, 7, 6], [4, 6, 8, 2, 4], [7, 4], [8, 5, 6, 2], [7, 3, 7, 7, 7, 4], [4, 3, 5, 3], [7, 8, 4], [2, 8, 3, 8]]
inputs_batch_major [[8 2 5 0 0 0]
 [4 4 7 6 0 0]
 [4 6 8 2 4 0]
 [7 4 0 0 0 0]
 [8 5 6 2 0 0]
 [7 3 7 7 7 4]
 [4 3 5 3 0 0]
 [7 8 4 0 0 0]
 [2 8 3 8 0 0]]
make_batch(each)
 [[8 4 4 7 8 7 4 7 2]
 [2 4 6 4 5 3 3 8 8]
 [5 7 8 0 6 7 5 4 3]
 [0 6 2 0 2 7 3 0 8]
 [0 0 4 0 0 7 0 0 0]
 [0 0 0 0 0 4 0 0 0]]

'''

train_graph = tf.Graph()
with train_graph.as_default():
    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_inputs_embedded,
        dtype=tf.float32, time_major=True,
    )
    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell, decoder_inputs_embedded,
        initial_state=encoder_final_state,
        dtype=tf.float32, time_major=True, scope="plain_decoder",
    )

    decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
    decoder_prediction = tf.argmax(decoder_logits, 2)
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits,
    )
    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)

loss_track = []
epochs = 300

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        batch = next(batches)#batches是一个生成器，每次从中取一个batch
        encoder_inputs_, _ = make_batch(batch)
        decoder_targets_, _ = make_batch([(sequence) + [EOS] for sequence in batch])
        decoder_inputs_, _ = make_batch([[EOS] + (sequence) for sequence in batch])
        feed_dict = {encoder_inputs: encoder_inputs_, decoder_inputs: decoder_inputs_,
                     decoder_targets: decoder_targets_,
                     }
        _, l = sess.run([train_op, loss], feed_dict)
        loss_track.append(l)
        if epoch == 0 or epoch % 100 == 0:
            print('loss: {}'.format(sess.run(loss, feed_dict)))
            predict_ = sess.run(decoder_prediction, feed_dict)
            for i, (inp, pred) in enumerate(zip(feed_dict[encoder_inputs].T, predict_.T)):
                print('input > {}'.format(inp))
                print('predicted > {}'.format(pred))
                if i >= 20:
                    break

plt.plot(loss_track)
plt.show()
