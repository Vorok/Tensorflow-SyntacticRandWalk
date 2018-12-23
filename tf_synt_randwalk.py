from __future__ import division
from collections import Counter, defaultdict
import os
import pickle
from random import shuffle
import tensorflow as tf
import spacy
from timeit import default_timer as timer

en_nlp = spacy.load('en')

class NotTrainedError(Exception):
    pass


class NotFitToCorpusError(Exception):
    pass


class SyntRandWalkModel():
    def __init__(self, embedding_size, context_size, max_vocab_size=100000, min_occurrences=1,
                 cooccurrence_cap=100, batch_size=512, learning_rate=0.05):
        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.__words = None
        self.__word_to_id = None
        self.__cooccurrence_matrix = None
        self.__cooccurrence_tensor = None
        self.__embeddings = None
        self.T = None

    def fit_to_corpus(self, corpus):
        self.__fit_to_corpus(corpus, self.max_vocab_size, self.min_occurrences,
                             self.left_context, self.right_context)
        self.__build_graph()


    def __fit_to_corpus(self, corpus, vocab_size, min_occurrences, left_size, right_size):
        start_time = timer()
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        for region in corpus:
            word_counts.update(region)
            for l_context, word, r_context in _context_windows(region, left_size, right_size):
                for i, context_word in enumerate(l_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurrence_counts[(word, context_word)] += 1
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1
        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")
        self.__words = [word for word, count in word_counts.most_common(vocab_size)
                        if count >= min_occurrences]
        self.__word_to_id = {word: i for i, word in enumerate(self.__words)}
        self.__cooccurrence_matrix = {
            (self.__word_to_id[words[0]], self.__word_to_id[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in self.__word_to_id and words[1] in self.__word_to_id}
        self.time_rw_fit_corp = timer() - start_time

    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default(), self.__graph.device(_device_for_node):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')

            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="cooccurrence_count")

            bias_C = tf.Variable(0.5, name="C")
            focal_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                name="focal_embeddings")
            context_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                name="context_embeddings")

            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.__context_input)
            weighting_factor = tf.minimum(
                100.0,
                self.__cooccurrence_count)
            #do we need reduce_sum?
            #print(focal_embedding.shape)
            ad = tf.add(focal_embedding, context_embedding)
            #print(ad.shape)
            norm = tf.norm(ad, axis=1, keepdims=True)
            #print(norm.shape)
            embedding_norm = tf.reduce_sum(tf.square(norm))
            #print(embedding_norm.shape)
            log_cooccurrences = tf.log(tf.to_float(self.__cooccurrence_count))

            distance_expr0 = tf.square(tf.add(
                log_cooccurrences,
                tf.negative(embedding_norm)))
            distance_expr = tf.subtract(distance_expr0, bias_C)

            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.__total_loss = tf.reduce_sum(single_losses)
            tf.summary.scalar("randwalk_loss", self.__total_loss)
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.__total_loss)
            self.__summary = tf.summary.merge_all()

            self.__combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                                name="combined_embeddings")
            self.__Ctf = bias_C

    def train(self, num_epochs):
        start_time = timer()
        batches = self.__prepare_batches()
        with tf.Session(graph=self.__graph) as session:
            tf.global_variables_initializer().run()
            for epoch in range(num_epochs):
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    i_s, j_s, counts = batch
                    #print("{}, {}, {}, {}".format(batch_index, i_s, j_s, counts))
                    #print(batch_index)
                    #print(type(i_s))
                    #print(type(i_s[0]))
                    if len(counts) != self.batch_size:
                        continue
                    feed_dict = {
                        self.__focal_input: i_s,
                        self.__context_input: j_s,
                        self.__cooccurrence_count: counts}
                    session.run([self.__optimizer], feed_dict=feed_dict)
            self.__embeddings = self.__combined_embeddings.eval()
            self.C = self.__Ctf.eval()
        self.time_rw_train = timer() - start_time

    def fit_to_corpus_synt(self, corpus, save_filepath, load_from_file=False):
        if(load_from_file):
            with open(save_filepath, "rb") as file:
                self.set_cooc_tensor(pickle.load(file))
                self.time_srw_fit_corp = "None (loaded from file)"
        else:
            self.__fit_to_corpus_synt(corpus, self.max_vocab_size, self.min_occurrences,
                                self.left_context, self.right_context)
            with open(save_filepath, "wb") as file:
                pickle.dump(self.get_cooc_tensor(), file)

        print("fit done")
        self.__build_graph_synt()
        print("graph done")

    def __fit_to_corpus_synt(self, corpus, vocab_size, min_occurrences, left_size, right_size):
        start_time = timer()
        cooccurrence_counts = defaultdict(float)
        for r, region in enumerate(corpus):
            if r % 1000 == 0:
                print("Region {} done.".format(r))
            for l_context, word_a, word_b, r_context in _context_windows_pairs(region, left_size, right_size):
                for i, context_word in enumerate(l_context[::-1]):
                    cooccurrence_counts[(word_a, word_b, context_word)] += 1
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word_a, word_b, context_word)] += 1
        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")
        self.__cooccurrence_tensor = {
            (self.__word_to_id[words[0]], self.__word_to_id[words[1]], self.__word_to_id[words[2]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in self.__word_to_id and words[1] in self.__word_to_id and words[2] in self.__word_to_id}
        self.time_srw_fit_corp = timer() - start_time

    def __build_graph_synt(self):
        tf.reset_default_graph()
        self.__graph = tf.Graph()
        with self.__graph.as_default(), self.__graph.device(_device_for_node):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            self.__Ctf = tf.constant([self.C], dtype=tf.float32,name='C')
            self.__focal_input_a = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words_a")
            self.__focal_input_b = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words_b")
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="cooccurrence_count")
            bias_Ca = tf.Variable(0.5, name="Ca")
            pair_embeddings = tf.Variable(
                tf.random_uniform([self.embedding_size, self.embedding_size, self.embedding_size], 1.0, -1.0),
                name="focal_embeddings")
            
            
            self.__combined_embeddings = tf.convert_to_tensor(self.__embeddings)
            focal_embedding_a = tf.nn.embedding_lookup(self.__combined_embeddings, self.__focal_input_a)
            focal_embedding_b = tf.nn.embedding_lookup(self.__combined_embeddings, self.__focal_input_b)
            context_embedding = tf.nn.embedding_lookup(self.__combined_embeddings, self.__context_input)
            weighting_factor = tf.minimum(
                100.0,
                self.__cooccurrence_count)

            T = tf.einsum('ijk,mi,mj->mk', pair_embeddings, focal_embedding_a, focal_embedding_b)
            ad = tf.add_n([focal_embedding_a, focal_embedding_b, context_embedding, T])
            norm = tf.norm(ad, axis=1, keepdims=True)
            embedding_norm = tf.reduce_sum(tf.square(norm))
            log_cooccurrences = tf.log(tf.to_float(self.__cooccurrence_count))
            distance_expr0 = tf.square(tf.add(
                log_cooccurrences,
                tf.negative(embedding_norm)))
            distance_expr = tf.subtract(tf.subtract(distance_expr0, bias_Ca), self.__Ctf)
            #distance_expr = tf.subtract(distance_expr0, bias_Ca)
            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.__total_loss = tf.reduce_sum(single_losses)
            tf.summary.scalar("syntRandWalk_loss", self.__total_loss)
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.__total_loss)
            self.__summary = tf.summary.merge_all()
            self.T = pair_embeddings
            #self.__Catf = bias_Ca

    def train_synt(self, num_epochs):
        start_time = timer()
        batches = self.__prepare_batches_synt()
        with tf.Session(graph=self.__graph) as session:
            tf.global_variables_initializer().run()
            for epoch in range(num_epochs):
                print("Epoch:", epoch)
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    ind1, ind2, ind3, counts = batch
                    #print("{}, {}, {}, {}, {}".format(batch_index, ind1, ind2, ind3, counts))
                    #print(batch_index)
                    #print(type(ind1))
                    #print(type(ind1[0]))
                    if len(counts) != self.batch_size:
                        continue
                    feed_dict = {
                        self.__focal_input_a: ind1,
                        self.__focal_input_b: ind2,
                        self.__context_input: ind3,
                        self.__cooccurrence_count: counts}
                    session.run([self.__optimizer], feed_dict=feed_dict)
            self.__emb_T = self.T.eval()
            #self.Ca = self.__Catf.eval()
        self.time_srw_train = timer() - start_time


    def setupT(self):
        self.__emb_T = self.T.eval()

    def getT(self):
        return self.__emb_T

    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self):
        if self.__cooccurrence_matrix is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        cooccurrences = [(word_ids[0], word_ids[1], count)
                         for word_ids, count in self.__cooccurrence_matrix.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return list(_batchify(self.batch_size, i_indices, j_indices, counts))

    def __prepare_batches_synt(self):
        if self.__cooccurrence_tensor is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        cooccurrences = [(word_ids[0], word_ids[1], word_ids[2], count)
                         for word_ids, count in self.__cooccurrence_tensor.items()]
        indices_1, indices_2, indices_3, counts = zip(*cooccurrences)
        return list(_batchify(self.batch_size, indices_1, indices_2, indices_3, counts))

    def get_cooc_tensor(self):
        return self.__cooccurrence_tensor

    # if ct was saved for later
    def set_cooc_tensor(self, t):
        self.__cooccurrence_tensor = t

    @property
    def vocab_size(self):
        return len(self.__words)

    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings

    def id_for_word(self, word):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids.")
        return self.__word_to_id[word]

    def generate_tsne(self, path=None, size=(100, 100), word_count=1000, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
        labels = self.words[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)


def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _context_windows_pairs(region, left_size, right_size):
    parse = en_nlp(" ".join(region))
    for i, word in enumerate(parse):
        if (i == 0):
            continue
        if word.pos_ == "NOUN" and parse[i - 1].pos_ == "ADJ":
            start_index = i - left_size
            end_index = i + right_size
            left_context = _window(region, start_index, i - 2)
            right_context = _window(region, i + 1, end_index)
            yield (left_context, str(word), str(parse[i - 1]), right_context)


def _window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens

# works if you have CUDA supported GPU:
'''
def _device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"
'''

def _device_for_node(n):
    return "/cpu:0"

def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)


def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    if path is not None:
        figure.savefig(path)
        plt.close(figure)

"""
import os
import re
file = open("data/enwik8.txt", "r")
doclist = [line for line in file]
docstr = ''.join(doclist)
sentences = re.split(r'[.!?]', docstr)
sentences = [sentence.split() for sentence in sentences if len(sentence) > 1]

model = SyntRandWalkModel(embedding_size=30, context_size=10, min_occurrences=200,
                            learning_rate=0.05, batch_size=256)
model.fit_to_corpus(sentences)
print('fit done')
model.train(num_epochs=10, log_dir="log/example", summary_batch_interval=1000)
print('train done')
print(model.embedding_for("ussr"))
"""