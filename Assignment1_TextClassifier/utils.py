import numpy as np


## Normalization
## input a numpy vector, output the normalized vector
def normalize(data):
    size = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return np.array([float((data[i] - mean) / std) for i in range(size)]).reshape(size, 1), mean, std


def normalize_with_params(data, mean, std):
    size = len(data)
    return np.array([float((data[i] - mean) / std) for i in range(size)]).reshape(size, 1)


## Load unlabeled test file
def load_test(test_file):
    words = []
    with open(test_file, 'rt', encoding="UTF-8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
            i += 1
    return words

