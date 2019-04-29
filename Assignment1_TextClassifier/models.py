import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import hw2_skeleton as hs
import features
import utils


## Support vector machine
## Features: length, frequency, syllables
def svm(training_file, development_file, test_file, counts):
    twords, tlabels_true = hs.load_file(training_file)
    dwords, dlabels_true = hs.load_file(development_file)
    test_words = utils.load_test(test_file)

    ## Length
    tlength_feature = hs.length_feature(twords)
    tlength_normalized, tl_mean, tl_std = utils.normalize(tlength_feature)
    dlength_feature = hs.length_feature(dwords)
    dlength_normalized = utils.normalize_with_params(dlength_feature, tl_mean, tl_std)

    ## Frequency
    tfrequency_feature = hs.frequency_feature(twords, counts)
    tfrequency_normalized, tf_mean, tf_std = utils.normalize(tfrequency_feature)
    dfrequency_feature = hs.frequency_feature(dwords, counts)
    dfrequency_normalized = utils.normalize_with_params(dfrequency_feature, tf_mean, tf_std)

    ## Syllables
    tsyllables_feature = features.syllables_feature(twords)
    tsyllables_normalized, tsy_mean, tsy_std = utils.normalize(tsyllables_feature)
    dsyllables_feature = features.syllables_feature(dwords)
    dsyllables_normalized = utils.normalize_with_params(dsyllables_feature, tsy_mean, tsy_std)

    ## Vowels
    tvowels_feature = features.vowels_feature(twords)
    tvowels_normalized, tv_mean, tv_std = utils.normalize(tvowels_feature)
    dvowels_feature = features.vowels_feature(dwords)
    dvowels_normalized = utils.normalize_with_params(dvowels_feature, tv_mean, tv_std)

    ## Consonants
    tconsonant_feature = features.vowels_feature(twords)
    tconsonant_normalized, tc_mean, tc_std = utils.normalize(tconsonant_feature)
    dconsonant_feature = features.vowels_feature(dwords)
    dconsonant_normalized = utils.normalize_with_params(dconsonant_feature, tc_mean, tc_std)

    ## Senses
    tsenses_feature = features.senses_feature(twords)
    tsenses_normalized, tse_mean, tse_std = utils.normalize(tsenses_feature)
    dsenses_feature = features.senses_feature(dwords)
    dsenses_normalized = utils.normalize_with_params(dsenses_feature, tse_mean, tse_std)

    ## Hypernyms
    thypernyms_feature = features.hypernyms_feature(twords)
    thypernyms_normalized, th_mean, th_std = utils.normalize(thypernyms_feature)
    dhypernyms_feature = features.hypernyms_feature(dwords)
    dhypernyms_normalized = utils.normalize_with_params(dhypernyms_feature, th_mean, th_std)

    x_train = np.column_stack((
        tlength_normalized, tfrequency_normalized,
        tsyllables_normalized,
        tsenses_normalized))
    y = tlabels_true

    x_dev = np.column_stack((
        dlength_normalized, dfrequency_normalized,
        dsyllables_normalized,
        dsenses_normalized))

    clf = SVC(C=48, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    clf.fit(x_train, y)
    y_pred = clf.predict(x_dev)

    daccuracy = hs.get_accuracy(y_pred, dlabels_true)
    dprecision = hs.get_precision(y_pred, dlabels_true)
    drecall = hs.get_recall(y_pred, dlabels_true)
    dfscore = hs.get_fscore(y_pred, dlabels_true)

    # Test Set
    # test_length_feature = hs.length_feature(test_words)
    # test_frequency_feature = hs.frequency_feature(test_words, counts)
    # test_syllables_feature = features.syllables_feature(test_words)
    # test_senses_feature = features.senses_feature(test_words)
    #
    # test_length_normalized = utils.normalize_with_params(test_length_feature, tl_mean, tl_std)
    # test_frequency_normalized = utils.normalize_with_params(test_frequency_feature, tf_mean, tf_std)
    # test_syllables_normalized = utils.normalize_with_params(test_syllables_feature, tsy_mean, tsy_std)
    # test_senses_normalized = utils.normalize_with_params(test_senses_feature, tse_mean, tse_std)
    #
    # x_test = np.column_stack((test_length_normalized, test_frequency_normalized,
    #                          test_syllables_normalized, test_senses_normalized))
    # y_pred_test = clf.predict(x_test)
    #
    # f = open('test_labels.txt', 'w')
    # for item in y_pred_test:
    #     print(item, file=f)
    # f.close()

    # training_performance = (tprecision, trecall, tfscore)
    development_performance = (daccuracy, dprecision, drecall, dfscore)
    return development_performance


## Random Forest
def random_forest(training_file, development_file, test_file, counts):
    twords, tlabels_true = hs.load_file(training_file)
    dwords, dlabels_true = hs.load_file(development_file)
    test_words = utils.load_test(test_file)

    ## Length
    tlength_feature = hs.length_feature(twords)
    tlength_normalized, tl_mean, tl_std = utils.normalize(tlength_feature)
    dlength_feature = hs.length_feature(dwords)
    dlength_normalized = utils.normalize_with_params(dlength_feature, tl_mean, tl_std)

    ## Frequency
    tfrequency_feature = hs.frequency_feature(twords, counts)
    tfrequency_normalized, tf_mean, tf_std = utils.normalize(tfrequency_feature)
    dfrequency_feature = hs.frequency_feature(dwords, counts)
    dfrequency_normalized = utils.normalize_with_params(dfrequency_feature, tf_mean, tf_std)

    ## Syllables
    tsyllables_feature = features.syllables_feature(twords)
    tsyllables_normalized, tsy_mean, tsy_std = utils.normalize(tsyllables_feature)
    dsyllables_feature = features.syllables_feature(dwords)
    dsyllables_normalized = utils.normalize_with_params(dsyllables_feature, tsy_mean, tsy_std)

    ## Vowels
    tvowels_feature = features.vowels_feature(twords)
    tvowels_normalized, tv_mean, tv_std = utils.normalize(tvowels_feature)
    dvowels_feature = features.vowels_feature(dwords)
    dvowels_normalized = utils.normalize_with_params(dvowels_feature, tv_mean, tv_std)

    ## Consonants
    tconsonant_feature = features.vowels_feature(twords)
    tconsonant_normalized, tc_mean, tc_std = utils.normalize(tconsonant_feature)
    dconsonant_feature = features.vowels_feature(dwords)
    dconsonant_normalized = utils.normalize_with_params(dconsonant_feature, tc_mean, tc_std)

    ## Senses
    tsenses_feature = features.senses_feature(twords)
    tsenses_normalized, tse_mean, tse_std = utils.normalize(tsenses_feature)
    dsenses_feature = features.senses_feature(dwords)
    dsenses_normalized = utils.normalize_with_params(dsenses_feature, tse_mean, tse_std)

    ## Hypernyms
    thypernyms_feature = features.hypernyms_feature(twords)
    thypernyms_normalized, th_mean, th_std = utils.normalize(thypernyms_feature)
    dhypernyms_feature = features.hypernyms_feature(dwords)
    dhypernyms_normalized = utils.normalize_with_params(dhypernyms_feature, th_mean, th_std)

    x_train = np.column_stack((
        tlength_normalized, tfrequency_normalized,
        tsyllables_normalized,
        tsenses_normalized, thypernyms_normalized))
    y = tlabels_true

    x_dev = np.column_stack((
        dlength_normalized, dfrequency_normalized,
        dsyllables_normalized,
        dsenses_normalized, dhypernyms_normalized))

    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=7, max_features=3, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=8, min_samples_split=50,
            min_weight_fraction_leaf=0.0, n_estimators=70, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

    clf.fit(x_train, y)
    y_pred = clf.predict(x_dev)

    daccuracy = hs.get_accuracy(y_pred, dlabels_true)
    dprecision = hs.get_precision(y_pred, dlabels_true)
    drecall = hs.get_recall(y_pred, dlabels_true)
    dfscore = hs.get_fscore(y_pred, dlabels_true)

    # Test Set
    test_length_feature = hs.length_feature(test_words)
    test_frequency_feature = hs.frequency_feature(test_words, counts)
    test_syllables_feature = features.syllables_feature(test_words)
    test_vowels_feature = features.vowels_feature(test_words)
    test_consonants_feature = features.consonants_feature(test_words)
    test_senses_feature = features.senses_feature(test_words)
    test_hypernyms_feature = features.hypernyms_feature(test_words)

    test_length_normalized = utils.normalize_with_params(test_length_feature, tl_mean, tl_std)
    test_frequency_normalized = utils.normalize_with_params(test_frequency_feature, tf_mean, tf_std)
    test_syllables_normalized = utils.normalize_with_params(test_syllables_feature, tsy_mean, tsy_std)
    test_vowels_normalized = utils.normalize_with_params(test_vowels_feature, tv_mean, tv_std)
    test_consonants_normalized = utils.normalize_with_params(test_consonants_feature, tc_mean, tc_std)
    test_senses_normalized = utils.normalize_with_params(test_senses_feature, tse_mean, tse_std)
    test_hypernyms_normalized = utils.normalize_with_params(test_hypernyms_feature, th_mean, th_std)

    x_test = np.column_stack((test_length_normalized, test_frequency_normalized,
                              test_syllables_normalized,
                              test_senses_normalized, test_hypernyms_normalized))
    y_pred_test = clf.predict(x_test)

    f = open('test_labels.txt', 'w')
    for item in y_pred_test:
        print(item, file=f)
    f.close()

    # training_performance = (tprecision, trecall, tfscore)
    development_performance = (daccuracy, dprecision, drecall, dfscore)
    return development_performance
