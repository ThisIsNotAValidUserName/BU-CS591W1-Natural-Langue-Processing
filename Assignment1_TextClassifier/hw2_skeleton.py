#############################################################
## ASSIGNMENT 1 CODE SKELETON
## RELEASED: 2/6/2019
## DUE: 2/15/2019
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import gzip
import numpy as np
import features
import utils
import models

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels


## Get data of the confusion matrix
def get_confusion_matrix_data(y_pred, y_true):
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i] and y_true[i] == 1:
            tp += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
        else:
            tn += 1
    data = [tp, fp, fn, tn]
    return data


## Calculates the accuracy of the predicted labels
def get_accuracy(y_pred, y_true):
    data = get_confusion_matrix_data(y_pred, y_true)
    accuracy = float((data[0] + data[3]) / (data[0] + data[1] + data[2] + data[3]))
    return accuracy


## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    ## YOUR CODE HERE...
    data = get_confusion_matrix_data(y_pred, y_true)
    precision = float(data[0] / (data[0] + data[1]))
    return precision


## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    ## YOUR CODE HERE...
    data = get_confusion_matrix_data(y_pred, y_true)
    recall = float(data[0] / (data[0] + data[2]))
    return recall


## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    ## YOUR CODE HERE...
    ## Beta = 1
    recall = get_recall(y_pred, y_true)
    precision = get_precision(y_pred, y_true)
    fscore = float(2.0 * precision * recall / (precision + recall))
    return fscore

#### 2. Complex Word Identification ####


## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="UTF-8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline


## Makes feature matrix for all complex
def all_complex_feature(words):
    labels_all_complex = []
    for i in range(len(words)):
        labels_all_complex.append(1)

    return labels_all_complex


## Labels every word complex
def all_complex(data_file):
    ## YOUR CODE HERE...
    words, labels_true = load_file(data_file)
    labels_all_complex = all_complex_feature(words)

    accuracy = get_accuracy(labels_all_complex, labels_true)
    precision = get_precision(labels_all_complex, labels_true)
    recall = get_recall(labels_all_complex, labels_true)
    fscore = get_fscore(labels_all_complex, labels_true)
    performance = [accuracy, precision, recall, fscore]
    return performance


### 2.2: Word length thresholding

## Finds the length of the longest word in a data set
def length_longest_word(words):
    longest = 0
    for i in range(len(words)):
        if len(words[i]) > longest:
            longest = len(words[i])
    return longest


## Makes feature matrix for word_length_threshold
## Returns a list of labels [0, 1, 0......]
def length_threshold_feature(words, threshold):
    labels_length_threshold = []
    for i in range(len(words)):
        if len(words[i]) > threshold:
            labels_length_threshold.append(1)
        else:
            labels_length_threshold.append(0)

    return labels_length_threshold


## Returns a numpy list of word length [2, 14, 3, .....] rather than just labels
def length_feature(words):
    word_length_feature = []
    for i in range(len(words)):
        word_length_feature.append(len(words[i]))

    return np.array(word_length_feature).T


## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    ## YOUR CODE HERE
    twords, tlabels_true = load_file(training_file)
    dwords, dlabels_true = load_file(development_file)

    fscore, threshold = 0.0, 0
    for temp_threshold in range(length_longest_word(twords)):
        tlabel_pred_temp = length_threshold_feature(twords, temp_threshold)
        temp_fscore = get_fscore(tlabel_pred_temp, tlabels_true)
        if temp_fscore > fscore:
            fscore = temp_fscore
            threshold = temp_threshold

    tlabels_pred = length_threshold_feature(twords, threshold)
    dlabels_pred = length_threshold_feature(dwords, threshold)

    taccuracy = get_accuracy(tlabels_pred, tlabels_true)
    tprecision = get_precision(tlabels_pred, tlabels_true)
    trecall = get_recall(tlabels_pred, tlabels_true)
    tfscore = get_fscore(tlabels_pred, tlabels_true)
    daccuracy = get_accuracy(dlabels_pred, dlabels_true)
    dprecision = get_precision(dlabels_pred, dlabels_true)
    drecall = get_recall(dlabels_pred, dlabels_true)
    dfscore = get_fscore(dlabels_pred, dlabels_true)

    training_performance = [taccuracy, tprecision, trecall, tfscore]
    development_performance = [daccuracy, dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.3: Word frequency thresholding


## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt', encoding='UTF-8') as f:
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set


## Make feature matrix for word_frequency_threshold
## Returns a numpy array of labels [0, 1, 0......]
def frequency_threshold_feature(words, threshold, counts):
    labels_frequency_pred = []
    for i in range(len(words)):
        if counts[words[i]] > threshold:
            labels_frequency_pred.append(1)
        else:
            labels_frequency_pred.append(0)

    return np.array(labels_frequency_pred)


# Returns a numpy array of word counts [count1, count2....] rather than labels
def frequency_feature(words, counts):
    word_frequency_feature = []
    for i in range(len(words)):
        word_frequency_feature.append(counts[words[i]])
        # print("Word: " + str(words[i]), end=" ")
        # print("Count: " + str(counts[words[i]]))

    return np.array(word_frequency_feature).T


def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE
    twords, tlabels_true = load_file(training_file)
    dwords, dlabels_true = load_file(development_file)
    fscore, threshold = 0, 0
    for temp_threshold in range(5000):
        # print(str(temp_threshold))
        tlabel_pred_temp = frequency_threshold_feature(twords, temp_threshold, counts)
        temp_fscore = get_fscore(tlabel_pred_temp, tlabels_true)
        if temp_fscore > fscore:
            fscore = temp_fscore
            threshold = temp_threshold

    # print("The best word frequency threshold is: " + str(threshold))
    # print("Fscore is: " + str(fscore))

    tlabels_pred = frequency_threshold_feature(twords, threshold, counts)
    dlabels_pred = frequency_threshold_feature(dwords, threshold, counts)

    taccuracy = get_accuracy(tlabels_pred, tlabels_true)
    tprecision = get_precision(tlabels_pred, tlabels_true)
    trecall = get_recall(tlabels_pred, tlabels_true)
    tfscore = get_fscore(tlabels_pred, tlabels_true)
    daccuracy = get_accuracy(dlabels_pred, dlabels_true)
    dprecision = get_precision(dlabels_pred, dlabels_true)
    drecall = get_recall(dlabels_pred, dlabels_true)
    dfscore = get_fscore(dlabels_pred, dlabels_true)

    training_performance = [taccuracy, tprecision, trecall, tfscore]
    development_performance = [daccuracy, dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.4: Naive Bayes


## Normalized length and frequency
def normalized_length_frequency(twords, dwords, counts):
    tfrequency_feature = frequency_feature(twords, counts)
    tfrequency_normalized, tf_mean, tf_std = utils.normalize(tfrequency_feature)

    tlength_feature = length_feature(twords)
    tlength_normalized, tl_mean, tl_std = utils.normalize(tlength_feature)

    dfrequency_feature = frequency_feature(dwords, counts)
    df_size = len(dfrequency_feature)
    dfrequency_normalized = \
        np.array([float((dfrequency_feature[i] - tf_mean) / tf_std) for i in range(df_size)]).reshape(df_size, 1)

    dlength_feature = length_feature(dwords)
    dl_size = len(dlength_feature)
    dlength_normalized = \
        np.array([float((dlength_feature[i] - tl_mean) / tl_std) for i in range(dl_size)]).reshape(dl_size, 1)

    return tlength_normalized, tfrequency_normalized, dlength_normalized, dfrequency_normalized


## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    twords, tlabels_true = load_file(training_file)
    dwords, dlabels_true = load_file(development_file)

    tlength_normalized, tfrequency_normalized, dlength_normalized, dfrequency_normalized \
        = normalized_length_frequency(twords, dwords, counts)

    x_train = np.column_stack((tlength_normalized, tfrequency_normalized))
    y = tlabels_true

    clf = GaussianNB()
    clf.fit(x_train, y)

    x_development = np.column_stack((dlength_normalized, dfrequency_normalized))
    y_pred = clf.predict(x_development)

    daccuracy = get_accuracy(y_pred, dlabels_true)
    dprecision = get_precision(y_pred, dlabels_true)
    drecall = get_recall(y_pred, dlabels_true)
    dfscore = get_fscore(y_pred, dlabels_true)

    # training_performance = (tprecision, trecall, tfscore)
    development_performance = (daccuracy, dprecision, drecall, dfscore)
    return development_performance

### 2.5: Logistic Regression


## Trains a logistic regression classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE
    twords, tlabels_true = load_file(training_file)
    dwords, dlabels_true = load_file(development_file)

    tlength_normalized, tfrequency_normalized, dlength_normalized, dfrequency_normalized \
        = normalized_length_frequency(twords, dwords, counts)

    x_train = np.column_stack((tlength_normalized, tfrequency_normalized))
    y = tlabels_true

    clf = LogisticRegression(solver='liblinear')
    clf.fit(x_train, y)

    x_development = np.column_stack((dlength_normalized, dfrequency_normalized))
    y_pred = clf.predict(x_development)

    daccuracy = get_accuracy(y_pred, dlabels_true)
    dprecision = get_precision(y_pred, dlabels_true)
    drecall = get_recall(y_pred, dlabels_true)
    dfscore = get_fscore(y_pred, dlabels_true)

    # training_performance = (tprecision, trecall, tfscore)
    development_performance = (daccuracy, dprecision, drecall, dfscore)
    return development_performance

### 2.7: Build your own classifier


## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE

##
'''Please go to models.py, my models are in there'''
##


def tune_parameter(training_file, development_file, counts):
    twords, tlabels_true = load_file(training_file)

    ## Length
    tlength_feature = length_feature(twords)
    tlength_normalized, tl_mean, tl_std = utils.normalize(tlength_feature)

    ## Frequency
    tfrequency_feature = frequency_feature(twords, counts)
    tfrequency_normalized, tf_mean, tf_std = utils.normalize(tfrequency_feature)

    ## Vowels
    tvowels_feature = features.vowels_feature(twords)
    tvowels_normalized, tv_mean, tv_std = utils.normalize(tvowels_feature)

    ## Consonants
    tconsonant_feature = features.vowels_feature(twords)
    tconsonant_normalized, tc_mean, tc_std = utils.normalize(tconsonant_feature)

    ## Senses
    tsenses_feature = features.senses_feature(twords)
    tsenses_normalized, tse_mean, tse_std = utils.normalize(tsenses_feature)

    ## Syllables
    tsyllables_feature = features.syllables_feature(twords)
    tsyllables_normalized, ts_mean, ts_std = utils.normalize(tsyllables_feature)

    ## Hypernyms
    thypernyms_feature = features.hypernyms_feature(twords)
    thypernyms_normalized, th_mean, th_std = utils.normalize(thypernyms_feature)

    x_train = np.column_stack((
        tlength_normalized, tfrequency_normalized,
        tsyllables_normalized,
        tsenses_normalized, thypernyms_normalized))
    y = tlabels_true

    ## Grid search for the best parameter
    # param_test1 = {'n_estimators': range(40, 91, 10),
    #                'max_depth': range(3, 14, 2),
    #                'min_samples_split': range(10, 101, 20),
    #                'min_samples_leaf': range(1, 15, 1),
    #                'max_features': range(1, 6, 1)}
    # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
    #                         param_grid=param_test1, scoring='roc_auc', cv=5)

    param_test1 = {'C': range(1, 51, 1)}
    gsearch1 = GridSearchCV(estimator=SVC(kernel='linear'),
                            param_grid=param_test1, scoring='roc_auc', cv=5)
    gsearch1.fit(x_train, y)
    print(str(gsearch1.best_estimator_))
    print(str(gsearch1.best_params_))
    print(str(gsearch1.best_score_))


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    # train_data = load_file(training_file)

    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    print("             Accuracy     Precision       Recall      Fscore")
    print()

    all_complex = all_complex(training_file)
    print("All complex: " + str(all_complex))

    training_performance, development_performance = word_length_threshold(training_file, development_file)
    print("Word length training: " + str(training_performance))
    print("Word length development: " + str(development_performance))

    frequency_training_performance, frequency_development_performance = \
        word_frequency_threshold(training_file, development_file, counts)
    print("Word frequency training: " + str(frequency_training_performance))
    print("Word frequency development: " + str(frequency_development_performance))

    nb_development_performance = naive_bayes(training_file, development_file, counts)
    print("Naive bayes development: " + str(nb_development_performance))

    lr_development_performance = logistic_regression(training_file, development_file, counts)
    print("Logistic Regression development: " + str(lr_development_performance))

    svm_development_performance = models.svm(training_file, development_file, test_file, counts)
    print("Support vector machine development: " + str(svm_development_performance))

    random_forest_development_performance = models.random_forest(training_file, development_file, test_file, counts)
    print("Random forest development: " + str(random_forest_development_performance))

    # tune_parameter(training_file, development_file, counts)

