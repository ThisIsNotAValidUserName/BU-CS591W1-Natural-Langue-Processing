import os
import csv
import subprocess
import re
import random
import numpy as np
import time
import math
import gc
import Classification
import nltk
nltk.download("stopwords")


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

    term_document_matrix = np.zeros((len(vocab), len(document_names)))

    for document_name, line_tokens in line_tuples:
        for token in line_tokens:
            term_document_matrix[vocab_to_id[token], docname_to_id[document_name]] += 1

    return term_document_matrix


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
    term_context_matrix += 1
    shape = term_context_matrix.shape
    sum_cw_fij = np.sum(term_context_matrix)

    p_ij = (term_context_matrix) / sum_cw_fij
    p_i = np.sum(p_ij, axis=1)[:, None]
    p_j = np.sum(p_ij, axis=0)[None, :]

    denom_mat = np.ones(shape)
    denom_mat /= p_i
    denom_mat /= p_j

    PMI_matrix = np.log2(p_ij * denom_mat)
    PPMI_matrix = np.maximum(PMI_matrix, 0.)

    return PPMI_matrix


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

    tf_idf_matrix = np.zeros(term_document_matrix.shape, dtype=np.float64)
    N = term_document_matrix.shape[1]

    for i in range(len(term_document_matrix)):
        word_counts = get_row_vector(term_document_matrix, i)
        tf_idf_matrix[i, :] = (
            np.array(list(word_counts)) * np.log(N / np.count_nonzero(word_counts)))

    return tf_idf_matrix


def compute_cosine_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

    Inputs:
     vector1: A nx1 numpy array
     vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    '''

    num = np.dot(vector1, vector2)
    norm1 = (np.sum(np.square(vector1))) ** 0.5
    norm2 = (np.sum(np.square(vector2))) ** 0.5
    mul = norm1 * norm2
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

    target_vector = term_document_matrix[:, target_play_index]
    result_tuples = []

    for index, curr_vector in enumerate(term_document_matrix.T):
        result_tuples.append(tuple((index, similarity_fn(target_vector, curr_vector))))

    sorted_results = sorted(result_tuples, key=lambda x: x[1], reverse=True)
    output = [i[0] for i in sorted_results]

    return output


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

    num_words, _ = matrix.shape
    similarities = []
    vector_target = get_row_vector(matrix, target_word_index)

    for i in range(num_words):
        vector_temp = get_row_vector(matrix, i)
        simi = similarity_fn(vector_target, vector_temp)
        similarities.append((i, simi))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    #    for i in range(50):
    #        print("%d: %s"%(i+1, similarities[i]))

    return [x[0] for x in similarities]


################ Additional Parts ###############


def read_character_in_shakspeare():
    '''Each tuple consists of
    tuple[0]: The name of the play
    tuple[1] A line from the play as a list of tokenized words.
    Returns:
        tuples: A list of tuples in the above format.
        ch_names: A list of the plays present in the corpus.
        vocab: A list of all tokens in the vocabulary.
    '''

    tuples = []
    with open('will_play_text.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        character_names = set()
        for row in csv_reader:
            character_name = row[4].strip().lower()
            # Remove asides & narrators
            if character_name == '':
                continue
            character_names.add(character_name)
            line = row[5]
            line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((character_name, line_tokens))

        f.close()
    character_names = list(character_names)

    # Sort the list to make sure the result is the same
    sorted_character = sorted(character_names)
    return tuples, sorted_character


def create_term_character_matrix(line_tuples, character_names, vocab):
    '''
    Similar to create_term_document_matrix
    '''

    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    chrctrname_to_id = dict(zip(character_names, range(0, len(character_names))))

    # YOUR CODE HERE
    term_character_matrix = np.zeros((len(vocab), len(character_names)))

    for character_name, line in line_tuples:
        for word in line:
            term_character_matrix[(vocab_to_id[word], chrctrname_to_id[character_name])] += 1

    return term_character_matrix


def rank_characters(target_character_index, term_character_matrix, similarity_fn, reverse=True):
    '''
    Ranks the similarity of all of the characters to the target character.
    '''

    # print(matrix.shape)

    target_vector = term_character_matrix[:, target_character_index]
    result_tuples = []

    for index, curr_vector in enumerate(term_character_matrix.T):
        result_tuples.append(tuple((index, similarity_fn(target_vector, curr_vector))))

    sorted_results = sorted(result_tuples, key=lambda x: x[1], reverse=reverse)
    output, sims = [i[0] for i in sorted_results], [i[1] for i in sorted_results]

    return output, sims


def character_most_similar(character_names, term_character_matrix, similarity_fns):
    characetrname_to_id = dict(zip(character_names, range(0, len(character_names))))
    most = []
    similarity, character1, character2 = 0.00000000000000000000000000, "", ""

    for sim_fn in similarity_fns:
        for character in character_names:
            # print("%s : %s" % (sim_fn.__qualname__, character))
            ranks, sims = rank_characters(characetrname_to_id[character], term_character_matrix, sim_fn, True)
            if sims[1] > similarity:
                similarity = sims[1]
                character1 = character
                character2 = character_names[ranks[1]]
        most.append((similarity, character1, character2))
        similarity, character1, character2 = 0.00000000000000000000000000, "", ""
    return most


def character_least_similar(character_names, term_character_matrix, similarity_fns):
    characetrname_to_id = dict(zip(character_names, range(0, len(character_names))))
    least = []
    similarity, character1, character2 = 1.00000000000000000000000000, "", ""

    for sim_fn in similarity_fns:
        for character in character_names:
            # print("%s : %s" % (sim_fn.__qualname__, character))
            ranks, sims = rank_characters(characetrname_to_id[character], term_character_matrix, sim_fn, False)
            if sims[1] < similarity:
                similarity = sims[1]
                character1 = character
                character2 = character_names[ranks[1]]
        least.append((similarity, character1, character2))
        similarity, character1, character2 = 1.00000000000000000000000000, "", ""
    return least


def rank_notable(target_notable_index, term_character_matrix, similarity_fn, notable_names_index, reverse=True):
    '''
    Ranks the similarity of all of the notable characters to the target character.
    '''
    # print(matrix.shape)
    num_words, num_characters = term_character_matrix.shape
    similarities = []
    vector1 = get_column_vector(term_character_matrix, target_notable_index)
    # print(vector1)
    for i in range(num_characters):
        if i not in notable_names_index:
            continue
        elif i == target_notable_index:
            continue
        vector2 = get_column_vector(term_character_matrix, i)
        simi = similarity_fn(vector1, vector2)
        similarities.append((i, simi))
        # print((names[i], simi))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=reverse)
    return [x[0] for x in similarities], [x[1] for x in similarities]


def notables_most_similar(character_names, notable_names_index, term_character_matrix, similarity_fns):
    characetrname_to_id = dict(zip(character_names, range(0, len(character_names))))
    most = []
    similarity, character1, character2 = 0.00000000000000000000000000, "", ""

    for sim_fn in similarity_fns:
        for character in character_names:
            if characetrname_to_id[character] not in notable_names_index:
                continue
            # print("%s : %s" % (sim_fn.__qualname__, character))
            ranks, sims = rank_notable(characetrname_to_id[character], term_character_matrix, sim_fn,
                                       notable_names_index, True)
            if sims[0] > similarity:
                similarity = sims[0]
                character1 = character
                character2 = character_names[ranks[0]]
        most.append((similarity, character1, character2))
        similarity, character1, character2 = 0.00000000000000000000000000, "", ""
    return most


def notables_least_similar(character_names, notable_names_index, term_character_matrix, similarity_fns):
    characetrname_to_id = dict(zip(character_names, range(0, len(character_names))))
    most = []
    similarity, character1, character2 = 1.00000000000000000000000000, "", ""

    for sim_fn in similarity_fns:
        for character in character_names:
            if characetrname_to_id[character] not in notable_names_index:
                continue
            # print("%s : %s" % (sim_fn.__qualname__, character))
            ranks, sims = rank_notable(characetrname_to_id[character], term_character_matrix, sim_fn,
                                       notable_names_index, False)
            if sims[0] < similarity:
                similarity = sims[0]
                character1 = character
                character2 = character_names[ranks[0]]
        most.append((similarity, character1, character2))
        similarity, character1, character2 = 1.00000000000000000000000000, "", ""
    return most


def read_names_in_shakspeare(file):
    name = []
    with open(file) as f:
        for line in f:
            name.append(line.strip().split('\t')[0].lower())
    f.close()
    return name


def male_and_female(term_character_matrix, similarity_fn, male, female, character_names):
    '''
    Ranks the similarity of all of the characters to the target character.
    '''

    # print(matrix.shape)
    chrctrname_to_id = dict(zip(character_names, range(0, len(character_names))))
    # num_words, num_characters = term_character_matrix.shape
    output = []

    male_indexes = [chrctrname_to_id[m] for m in male]
    female_indexes = [chrctrname_to_id[fm] for fm in female]

    for index_m in male_indexes:
        similarities_m_m, similarities_m_fm = [], []
        vector_target = get_column_vector(term_character_matrix, index_m)
        for index_fm in female_indexes:
            vector_temp = get_column_vector(term_character_matrix, index_fm)
            simi_fm = similarity_fn(vector_target, vector_temp)
            similarities_m_fm.append(simi_fm)
        for index_m_temp in male_indexes:
            vector_temp = get_column_vector(term_character_matrix, index_m_temp)
            simi_m = similarity_fn(vector_target, vector_temp)
            similarities_m_m.append(simi_m)

    for index_fm in female_indexes:
        similarities_fm_fm = []
        vector_target = get_column_vector(term_character_matrix, index_fm)
        for index_fm_temp in female_indexes:
            vector_temp = get_column_vector(term_character_matrix, index_fm_temp)
            simi_fm = similarity_fn(vector_target, vector_temp)
            similarities_fm_fm.append(simi_fm)

    output.append((np.sum(similarities_m_m) / len(similarities_m_m), np.std(similarities_m_m)))
    output.append((np.sum(similarities_m_fm) / len(similarities_m_fm), np.std(similarities_m_fm)))
    output.append((np.sum(similarities_fm_fm) / len(similarities_fm_fm), np.std(similarities_fm_fm)))
    return output


# Just for testing
def sim_between_particular(character_names, term_character_matrix, character1, character2, sim_fn):
    chrctrname_to_id = dict(zip(character_names, range(0, len(character_names))))
    vector1 = get_column_vector(term_character_matrix, chrctrname_to_id[character1])
    vector2 = get_column_vector(term_character_matrix, chrctrname_to_id[character2])
    similarity = sim_fn(vector1, vector2)
    return similarity


if __name__ == '__main__':
    tuples, document_names, vocab = read_in_shakespeare()

    print('Computing term document matrix...')
    td_matrix = create_term_document_matrix(tuples, document_names, vocab)

    print('Computing tf-idf matrix...')
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)

    print('Computing term context matrix...')
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

    print('Computing PPMI matrix...')
    PPMI_matrix = create_PPMI_matrix(tc_matrix)

    '''
    Codes below are just for testing
    My laptop has limited memory so I have to code like this to 
        test my PPMI matrix
    The create_PPMI_matrix function is based on the code below , but I can't test it
        so I am not sure if there is a bug or something
    The following code is written with add-one smoother
    '''
    #    tc_matrix_e = tc_matrix + 1
    #    shape = tc_matrix.shape
    #
    #    del tc_matrix
    #    gc.collect()
    #
    #    denominator = np.sum(tc_matrix_e)
    #
    #    p_ab = (tc_matrix_e) / denominator
    #
    #    del tc_matrix_e, denominator
    #    gc.collect()
    #
    #    p_a = np.sum(p_ab, axis=1)[:,None]  # sum each row
    #    p_b = np.sum(p_ab, axis=0)[None,:]  # sum each col
    #
    #    denom_mat = np.ones(shape)
    #    denom_mat /= p_a
    #    denom_mat /= p_b
    #
    #    del p_a, p_b, shape
    #    gc.collect()
    #
    #    pmi_mat = np.log2(p_ab * denom_mat)
    #    PPMI_matrix = np.maximum(pmi_mat, 0.)
    #

    random_idx = random.randint(0, len(document_names) - 1)
    similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
    for sim_fn in similarity_fns:
        print('\nThe top 10 most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
        ranks = rank_plays(random_idx, td_matrix, sim_fn)
        for idx in range(0, 10):
            doc_id = ranks[idx]
            print('%d: %s' % (idx+1, document_names[doc_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on term-document frequency matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], td_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on TF-IDF matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], tf_idf_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))


    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on PPMI matrix (with add-one smoother)'
              ' are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))

    ################ Additional Part 1 Similarity between characters ###############
    tuples_char, character_names = read_character_in_shakspeare()
    chrctrname_to_id = dict(zip(character_names, range(0, len(character_names))))

    print('Computing term character matrix...')
    term_character_matrix = create_term_character_matrix(tuples_char, character_names, vocab)

    print('Computing tf-idf matrix for characters...')
    tf_idf_matrix = create_tf_idf_matrix(term_character_matrix)

    num_terms, num_characters = term_character_matrix.shape

    print('Computing the most similar characters on term_character_matrix ...')
    character_most = character_most_similar(character_names, term_character_matrix, similarity_fns)
    print('Computing the least similar characters on term_character_matrix...')
    character_least = character_least_similar(character_names, term_character_matrix, similarity_fns)
    print(character_most)
    print(character_least)

    print('Computing the most similar characters on tf-idf matrix...')
    character_most = character_most_similar(character_names, tf_idf_matrix, similarity_fns)
    print('Computing the least similar characters on tf-idf matrix...')
    character_least = character_least_similar(character_names, tf_idf_matrix, similarity_fns)
    print(character_most)
    print(character_least)

    notable = read_names_in_shakspeare('./Notable.txt')
    notable_names = [name for name in notable if name in character_names]
    notable_names_index = [chrctrname_to_id[name] for name in character_names if name in notable_names]

    print('Computing the most similar notable characters on term_character_matrix...')
    most = notables_most_similar(character_names, notable_names_index, term_character_matrix, similarity_fns)
    print('Computing the least similar notable characters on term_character_matrix...')
    least = notables_least_similar(character_names, notable_names_index, term_character_matrix, similarity_fns)
    print(most)
    print(least)

    print('Computing the most similar notable characters on tf_idf_matrix...')
    most = notables_most_similar(character_names, notable_names_index, tf_idf_matrix, similarity_fns)
    print('Computing the least similar notable characters on tf_idf_matrix...')
    least = notables_least_similar(character_names, notable_names_index, tf_idf_matrix, similarity_fns)
    print(most)
    print(least)

    # for sim_fn in similarity_fns:
    # character = 'king henry iv'
    # print('\nThe 10 most similar characters to "%s" using %s on TF-IDF matrix are:' % (character, sim_fn))
    # ranks, sims = rank_characters(chrctrname_to_id[character], tf_idf_matrix, sim_fn)
    #     for idx in range(0, 10):
    #         word_id = ranks[idx]
    #         print('%d: %s' % (idx + 1, character_names[word_id]))


    ################ Additional Part2 Male and Female Characters ###############
    male = read_names_in_shakspeare('Male.txt')
    female = read_names_in_shakspeare('Female.txt')

    # Males, Females = [], []

    Males = [m for m in male if m in character_names]
    Females = [fm for fm in female if fm in character_names]

    print("There are", len(Males), "males and", len(Females), "females appeared in will_play_text.csv")

    for sim_fn in similarity_fns:
        print('\nAverage and standard deviation of similarities between female and male characters with %s are: '
              %(sim_fn.__qualname__))
        output = male_and_female(term_character_matrix, sim_fn, Males, Females, character_names)
        print("Males and males Avg: %s, Std: %s"%(output[0][0], output[0][1]))
        print("Males and females Avg: %s, Std: %s" % (output[1][0], output[1][1]))
        print("Females and females Avg: %s, Std: %s" % (output[2][0], output[2][1]))


    ################ Additional Part3 Classification of Plays ###############
    # Classification.main()

