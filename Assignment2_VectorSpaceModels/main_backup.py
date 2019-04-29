import os
import csv
import subprocess
import re
import random
import numpy as np
import time
import math


def read_in_shakespeare():
    '''Reads in the Shakespeare dataset processesit into a list of tuples.
      Also reads in the vocab and play name lists from files.

    Each tuple consists of
    tuple[0]: The name of the play
    tuple[1] A line from the play as a list of tokenized words.

    Returns:
        tuples: A list of tuples in the above format.
        document_names: A list of the plays present in the corpus.
        vocab: A list of all tokens in the vocabulary.
    '''

    tuples = []

    with open('will_play_text.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            play_name = row[1]
            line = row[5]
            line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((play_name, line_tokens))

    with open('vocab.txt') as f:
        vocab = [line.strip() for line in f]

    with open('play_names.txt') as f:
        document_names = [line.strip() for line in f]

    return tuples, document_names, vocab


def get_row_vector(matrix, row_id):
    return matrix[row_id, :]


def get_column_vector(matrix, col_id):
    return matrix[:, col_id]


def create_term_document_matrix(line_tuples, document_names, vocab):
    '''Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
        line_tuples: A list of tuples, containing the name of the document and
        a tokenized line from that document.
        document_names: A list of the document names
        vocab: A list of the tokens in the vocabulary
    
    Let m = len(vocab) and n = len(document_names).

    Returns:
        td_matrix: A mxn numpy array where the number of rows is the number of words
         and each column corresponds to a document. A_ij contains the
            frequency with which word i occurs in document j.
    '''

    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    docname_to_id = dict(zip(document_names, range(0, len(document_names))))

    matrix = np.zeros((len(vocab), len(document_names)))

    for play, line in line_tuples:
        for word in line:
            matrix[(vocab_to_id[word], docname_to_id[play])] += 1

    return matrix
    #
    # term_document_matrix = np.zeros((len(vocab), len(document_names)))
    #
    # for document_name, line_tokens in line_tuples:
    #     for token in line_tokens:
    #         term_document_matrix[vocab_to_id[token], docname_to_id[document_name]] += 1
    #
    # return term_document_matrix


def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    '''Returns a numpy array containing the term context matrix for the input lines.

    Inputs:
        line_tuples: A list of tuples, containing the name of the document and
        a tokenized line from that document.
        vocab: A list of the tokens in the vocabulary

    Let n = len(vocab).

    Returns:
        tc_matrix: A nxn numpy array where A_ij contains the frequency with which
            word j was found within context_window_size to the left or right of
            word i in any sentence in the tuples.
    '''

    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))

    term_context_matrix = np.zeros((len(vocab), len(vocab)))

    for tuples in line_tuples:
        line_tokens = tuples[1]
        len_line = len(line_tokens)
        for i in range(0, len_line):
            min_context = i - context_window_size
            if min_context < 0:
                min_context = 0
            max_context = i + context_window_size
            if max_context >= len_line:
                max_context = len_line - 1

            for j in range(min_context, max_context + 1):
                if j != i:
                    term_context_matrix[vocab_to_id[line_tokens[i]], vocab_to_id[line_tokens[j]]] += 1

    return term_context_matrix


def create_PPMI_matrix(term_context_matrix):
    '''Given a term context matrix, output a PPMI matrix.
    
    Hint: Use numpy matrix and vector operations to speed up implementation.
  
    Input:
        term_context_matrix: A nxn numpy array, where n is
            the numer of tokens in the vocab.
  
    Returns: A nxn numpy matrix, where A_ij is equal to the
        point-wise mutual information between the ith word
        and the jth word in the term_context_matrix.
    '''
    
    para = term_context_matrix.shape
    totals = np.sum(term_context_matrix)
    ppmi = np.zeros(para)
    rows = np.sum(term_context_matrix,axis=1)
    cols = np.sum(term_context_matrix,axis=0)
    for i in range(0,para[0]):
        for j in range (0,para[1]):
          ppmi[i][j] = (term_context_matrix[i][j] * totals + 1 ) / ((rows[i] * cols[j]) + totals)

    ppmi_final = np.log2(ppmi)

    return ppmi_final

#    term_context_matrix
#    matrix_sum = np.sum(term_context_matrix)
#    word_sum = np.sum(term_context_matrix, axis=1)
#    context_sum = np.sum(term_context_matrix, axis=0)
#
#    denominator = np.ones(term_context_matrix.shape)
#    denominator /= word_sum
#    denominator /= context_sum
#
#    PPMI_matrix = np.log2(matrix_sum * term_context_matrix * denominator)
#    PPMI_matrix = np.maximum(0, PPMI_matrix)
#
#    # YOUR CODE HERE
#    return PPMI_matrix


def create_tf_idf_matrix(term_document_matrix):
    '''Given the term document matrix, output a tf-idf weighted version.
  
    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
        term_document_matrix: Numpy array where each column represents a document
        and each row, the frequency of a word in that document.

    Returns:
        A numpy array with the same dimension as term_document_matrix, where
        A_ij is weighted by the inverse document frequency of document h.
    '''

    # (22602, 36)
    
    tf_idf_matrix = np.zeros(term_document_matrix.shape)
    df_raw = term_document_matrix.copy()
    tf_raw = term_document_matrix.copy()
    df_raw[df_raw > 0] = 1
    df = np.sum(df_raw, axis=1)
    docs = term_document_matrix.shape[1]
    idf = np.log(docs / df[df > 0])

    tf_raw[tf_raw > 0] = np.log10(tf_raw[tf_raw > 0]) + 1
    tf_raw[tf_raw <= 0] = 0

    tf = tf_raw

    for row in range(idf.shape[0]):
        tf_idf_matrix[row] = tf[row] * idf[row]

    return tf_idf_matrix
    
#    tf_idf_matrix = np.zeros(term_document_matrix.shape)
#    N = term_document_matrix.shape[1]
#    log_func = lambda t: np.log10(t) + 1 if t > 0 else 0
#    
#    for i in range(len(term_document_matrix)):
#        word_counts = get_row_vector(term_document_matrix, i)
#        tf_idf_matrix[i, :] = (
#            np.array(list(map(log_func, word_counts))) * np.log(N / np.count_nonzero(word_counts)))
#    
#    return tf_idf_matrix


def compute_cosine_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

    Inputs:
     vector1: A nx1 numpy array
     vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    '''

    num = np.dot(vector1, vector2)
    den1 = (np.sum(vector1 ** 2))**0.5
    den2 = (np.sum(vector2 ** 2))**0.5
    mul = den1 * den2
    if mul == 0:
        return 0
    return num / mul

def compute_jaccard_similarity(vector1, vector2):
    '''Computes the jaccard similarity of the two input vectors.

    Inputs:
        vector1: A nx1 numpy array
        vector2: A nx1 numpy array

    Returns:
        A scalar similarity value.
    '''

    minimums = np.sum(np.minimum(vector1, vector2))
    maximums = np.sum(np.maximum(vector1, vector2))
    return minimums / maximums


def compute_dice_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    '''

    numerator = 2 * np.sum(np.minimum(vector1, vector2))
    denominator = np.sum(vector1 + vector2)
    return numerator / denominator


def rank_plays(target_play_index, term_document_matrix, similarity_fn):
    ''' Ranks the similarity of all of the plays to the target play.

    Inputs:
        target_play_index: The integer index of the play we want to compare all others against.
        term_document_matrix: The term-document matrix as a mxn numpy array.
        similarity_fn: Function that should be used to compared vectors for two
        documents. Either compute_dice_similarity, compute_jaccard_similarity, or
        compute_cosine_similarity.

    Returns:
        A length-n list of integer indices corresponding to play names,
        ordered by decreasing similarity to the play indexed by target_play_index
    '''

    num_words, num_plays = term_document_matrix.shape
    similarities = np.zeros(num_plays)
    document_vector_target = get_column_vector(term_document_matrix, target_play_index)
    for i in range(num_plays):
        document_vector_current = get_column_vector(term_document_matrix, i)
        similarities[i] = similarity_fn(document_vector_target, document_vector_current)

    return np.argsort(-similarities)


def rank_words(target_word_index, matrix, similarity_fn):
    ''' Ranks the similarity of all of the words to the target word.

    Inputs:
        target_word_index: The index of the word we want to compare all others against.
        matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
        similarity_fn: Function that should be used to compared vectors for two word
        ebeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
        compute_cosine_similarity.

    Returns:
        A length-n list of integer word indices, ordered by decreasing similarity to the
        target word indexed by word_index
    '''

    # print(matrix.shape)

#    num_words, _ = matrix.shape
#    similarities = np.zeros(num_words)
#    word_vector_target = get_row_vector(matrix, target_word_index)
#    for i in range(num_words):
#        word_vector_comparing = get_row_vector(matrix, i)
#        similarities[i] = similarity_fn(word_vector_target, word_vector_comparing)
#    
#    return np.argsort(-similarities)

    num_words, _ = matrix.shape
    similarities = []
    vector1 = get_row_vector(matrix, target_word_index)
    for i in range(num_words):
        vector2 = get_row_vector(matrix, i)
        simi = similarity_fn(vector1, vector2)
        similarities.append((i, simi))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    # print(similarities)
    return [x[0] for x in similarities]


if __name__ == '__main__':
    tuples, document_names, vocab = read_in_shakespeare()

#    print('Computing term document matrix...')
#    td_matrix = create_term_document_matrix(tuples, document_names, vocab)
#
#    print('Computing tf-idf matrix...')
#    tf_idf_matrix = create_tf_idf_matrix(td_matrix)

    print('Computing term context matrix...')
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

    print('Computing PPMI matrix...')
    PPMI_matrix = create_PPMI_matrix(tc_matrix)

    random_idx = random.randint(0, len(document_names)-1)
    similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
#    for sim_fn in similarity_fns:
#        print('\nThe top 10 most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
#        ranks = rank_plays(random_idx, td_matrix, sim_fn)
#        for idx in range(0, 10):
#            doc_id = ranks[idx]
#            print('%d: %s' % (idx+1, document_names[doc_id]))
#
#    word = 'juliet'
#    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
#    for sim_fn in similarity_fns:
#        print('\nThe 10 most similar words to "%s" using %s on term-document frequency matrix are:' % (word, sim_fn.__qualname__))
#        ranks = rank_words(vocab_to_index[word], td_matrix, sim_fn)
#        for idx in range(0, 10):
#            word_id = ranks[idx]
#            print('%d: %s' % (idx+1, vocab[word_id]))

#    word = 'juliet'
#    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
#    for sim_fn in similarity_fns:
#        print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (word, sim_fn.__qualname__))
#        ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
#        for idx in range(0, 10):
#            word_id = ranks[idx]
#            print('%d: %s' % (idx+1, vocab[word_id]))

#    word = 'juliet'
#    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
#    for sim_fn in similarity_fns:
#        print('\nThe 10 most similar words to "%s" using %s on TF-IDF matrix are:' % (word, sim_fn.__qualname__))
#        ranks = rank_words(vocab_to_index[word], tf_idf_matrix, sim_fn)
#        for idx in range(0, 30):
#            word_id = ranks[idx]
#            print('%d: %s' % (idx+1, vocab[word_id]))
    
#    word = 'juliet'
#    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
#    print('\nThe 10 most similar words to "%s" using compute_cosine_similarity on TF-IDF matrix are:' % (word))
#    ranks = rank_words(vocab_to_index[word], tf_idf_matrix, compute_cosine_similarity)
#    for idx in range(0, 50):
#        word_id = ranks[idx]
#        print('%d: %s' % (idx+1, vocab[word_id]))
    
    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))
