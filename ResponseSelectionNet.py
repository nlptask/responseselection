import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import Word2Vec


class ResponseSelectionNet(object):
    def __init__(self):
        self.utterances_in_context = 87
        self.negative_response = 1
        self.words_in_sentence = 140
        self.word_embedding_size = 400
        self.rnn_units = 200
        self.total_words = 175511
        self.batch_size = 40

    def build_net(self):
        '''construct the net'''

        self.embedding_ph = tf.placeholder(
            tf.float32, shape=(self.total_words, self.word_embedding_size))
        word_embeddings = tf.get_variable(
            name='wv',
            shape=(self.total_words, self.word_embedding_size),
            dtype=tf.float32,
            trainable=False)
        self.embedding = word_embeddings.assign(self.embedding_ph)

        self.context_ph = tf.placeholder(
            tf.int32,
            shape=(None, self.utterances_in_context, self.words_in_sentence))
        context_embeddings = tf.nn.embedding_lookup(word_embeddings,
                                                    self.context_ph)
        context_embeddings = tf.unstack(
            context_embeddings, num=self.utterances_in_context, axis=1)
        self.context_len_ph = tf.placeholder(
            tf.int32, shape=(None, self.utterances_in_context))
        context_len = tf.unstack(
            self.context_len_ph, num=self.utterances_in_context, axis=1)

        self.response_ph = tf.placeholder(
            tf.int32, shape=(None, self.words_in_sentence))
        response_embeddings = tf.nn.embedding_lookup(word_embeddings,
                                                     self.response_ph)
        self.response_len = tf.placeholder(tf.int32, shape=(None, ))
        self.response_y = tf.placeholder(tf.int32, shape=(None, ))

        sentence_GRU = tf.nn.rnn_cell.GRUCell(
            self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        final_GRU = tf.nn.rnn_cell.GRUCell(
            self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        A_matrix = tf.get_variable(
            'A_matrix_v',
            shape=(self.rnn_units, self.rnn_units),
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32)
        reuse = None

        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(
            sentence_GRU,
            response_embeddings,
            sequence_length=self.response_len,
            dtype=tf.float32,
            scope='sentence_GRU')
        self.response_GRU_embedding_save = response_GRU_embeddings
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
        response_GRU_embeddings = tf.transpose(
            response_GRU_embeddings, perm=[0, 2, 1])
        matching_vectors = []

        for sentence_embeddings, sentence_len in zip(context_embeddings,
                                                     context_len):
            matrix1 = tf.matmul(sentence_embeddings, response_embeddings)
            sentence_GRU_embeddings, _ = tf.nn.dynamic_rnn(
                sentence_GRU,
                sentence_embeddings,
                sequence_length=sentence_len,
                dtype=tf.float32,
                scope='sentence_GRU')
            matrix2 = tf.einsum('aij,jk->aik', sentence_GRU_embeddings,
                                A_matrix)
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            conv_layer = tf.layers.conv2d(
                matrix,
                filters=8,
                kernel_size=(3, 3),
                padding='VALID',
                kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                activation=tf.nn.relu,
                reuse=reuse,
                name='conv')
            pooling_layer = tf.layers.max_pooling2d(
                conv_layer, (3, 3),
                strides=(3, 3),
                padding='VALID',
                name='max_pooling')
            matching_vector = tf.layers.dense(
                tf.contrib.layers.flatten(pooling_layer),
                50,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.tanh,
                reuse=reuse,
                name='matching_v')
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)

        _, last_hidden = tf.nn.dynamic_rnn(
            final_GRU,
            tf.stack(matching_vectors, axis=0, name='matching_stack'),
            dtype=tf.float32,
            time_major=True,
            scope='final_GRU')
        logits = tf.layers.dense(
            last_hidden,
            2,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='final_v')
        self.y_pred = tf.nn.softmax(logits)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.response_y, logits=logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.loss)

    def train_net(self):
        model = Word2Vec.load('../../output/wv.model')
        wordvec = model.wv
        vec_size = wordvec.vector_size
        vocab_size = len(wordvec.vocab)
        embedding_mat = np.zeros(shape=(vocab_size, vec_size), dtype='float32')
        for idx, word in enumerate(wordvec.vocab):
            embedding_mat[idx] = wordvec.get_vector(word)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(
                self.embedding, feed_dict={self.embedding_ph: embedding_mat})


if __name__ == "__main__":
    rsn = ResponseSelectionNet()
    rsn.build_net()
    rsn.train_net()
