import numpy as np
import syllables
from nltk.corpus import wordnet as wn


## Feature: number of syllables
def syllables_feature(words):
    word_syllables_feature = []
    for word in words:
        word_syllables_feature.append(syllables.count_syllables(word))
    return np.array(word_syllables_feature).T


## Feature: number of senses
def senses_feature(words):
    word_senses_feature = []
    for word in words:
        word_senses_feature.append(len(wn.synsets(word)))
    return np.array(word_senses_feature).T


## Feature: number of hypernyms
def hypernyms_feature(words):
    word_hypernyms_feature = []
    for word in words:
        word_hypernyms_feature.append(hypernyms_each_word(word))
    return np.array(word_hypernyms_feature).T


def hypernyms_each_word(word):
    num = 0
    synsets = wn.synsets(word)
    for synset_temp in synsets:
        num += len(synset_temp.hypernyms())
    return num


## Feature: number of vowels
def vowels_feature(words):
    word_vowels_feature = []
    for word in words:
        word_vowels_feature.append(count_vowels(word))
    return np.array(word_vowels_feature).T


def count_vowels(word):
    vowels = "aeiou"
    num = 0
    for char in word:
        if char in vowels:
            num += 1
    return num


## Feature: number of consonants
def consonants_feature(words):
    word_consonants_feature = []
    for word in words:
        word_consonants_feature.append(count_consonants(word))
    return np.array(word_consonants_feature).T


def count_consonants(word):
    vowels = "aeiou"
    num = 0
    for char in word:
        if char not in vowels:
            num += 1
    return num
