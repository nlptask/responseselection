from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf

model = Word2Vec.load('../../output/wv.model')
word_vectors = model.wv
print(type(word_vectors))
vec_size = word_vectors.vector_size
vocab_size = len(word_vectors.vocab)
print(vec_size)
print(vocab_size)

# Create the embedding matrix where words are indexed alphabetically
embedding_mat = np.zeros(shape=(vocab_size, vec_size), dtype='float32')
for idx, word in enumerate(word_vectors.vocab):
    embedding_mat[idx] = word_vectors.get_vector(word)
    if idx == 0:
        print(word)

# Setup the embedding matrix for tensorflow
with tf.variable_scope("input_layer"):
    embedding_tf = tf.get_variable(
       "embedding", 
        [vocab_size, vec_size],
        initializer=tf.constant_initializer(embedding_mat),
        trainable=False
    )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.shape(embedding_tf)))
    print(embedding_tf[0].eval())

    # Integrate this into your model
    batch_size = 32     # just for example
    seq_length = 20
    input_data = tf.placeholder(tf.int32, shape=(batch_size, seq_length))
    inputs = tf.nn.embedding_lookup(embedding_tf, input_data)
    v = np.random.randint(vec_size, size=(batch_size, seq_length))
    for j in range(batch_size):
        for item in range(seq_length):
            print(word_vectors.index2word[v[j][item]], end=' ')
        print('\r\n')
        print(v[j,:])
    print(sess.run(inputs, feed_dict={input_data: v}))

testwords = ['金融','上','股票','跌','经济','牛奶','梁朝伟']
for i in range(len(testwords)):
    res = model.most_similar(testwords[i])
    print (testwords[i])
    print (res)
