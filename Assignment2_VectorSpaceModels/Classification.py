import csv
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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


def read_in_shakespeare_play_lines(document_names):
    '''
    Read in the lines of plays
    Returns: An list of lists containig lines of all plays, one list per play.
    '''
    document_to_index = dict(zip(document_names, range(0, len(document_names))))

    document_lines = [[] for i in range(len(document_names))]

    with open('will_play_text.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            # play_name = row[1]
            index = document_to_index[row[1]]
            # line = row[5]
            document_lines[index].append(row[5])

    return document_lines


def main():
    tuples, document_names, vocab = read_in_shakespeare()

    Comedies = ['The Tempest', 'Two Gentlemen of Verona', 'Merry Wives of Windsor', 'Measure for measure',
                'A Comedy of Errors', 'Much Ado about nothing', 'Loves Labours Lost', 'A Midsummer nights dream',
                'Merchant of Venice', 'As you like it', 'Taming of the Shrew', 'Alls well that ends well',
                'Twelfth Night', 'A Winters Tale', 'Pericles']
    Histories = ['King John', 'Henry IV', 'Henry V', 'Henry VI Part 1', 'Henry VI Part 2',
                 'Henry VI Part 3', 'Richard II', 'Richard III', 'Henry VIII']
    Tragedies = ['Troilus and Cressida', 'Coriolanus', 'Titus Andronicus', 'Romeo and Juliet', 'Timon of Athens',
                 'Julius Caesar', 'macbeth', 'Hamlet', 'King Lear', 'Othello', 'Antony and Cleopatra', 'Cymbeline']

    print(
    "There are", len(Comedies), "Comedies", len(Histories), "Histories", len(Tragedies), "Tragedies")
    labels_true = [(0*(doc in Comedies) + 1*(doc in Histories) + 2*(doc in Tragedies)) for doc in document_names]

    document_lines_by_play = read_in_shakespeare_play_lines(document_names)
    document_lines_by_play_toStr = [' '.join(list) for list in document_lines_by_play]

    # Computing TF-IDF matrix
    print("Computing TF-IDF matrix.....")
    vectorizer = CountVectorizer(min_df=5, max_df=36, stop_words=stopwords.words('english'))
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(vectorizer.fit_transform(document_lines_by_play_toStr))

    #Select best K features
    print("Selecting best K features.....")
    skb = SelectKBest(chi2, k=100)  # 选择k个最佳特征
    tf_idf_K = skb.fit_transform(tf_idf, labels_true)  # iris.data是特征数据，iris.target是标签数据，该函数可以选择出k个特征

    X = tf_idf_K.toarray()
    Y = labels_true

    # Clustering
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    print("Centers of 3 clusters are: ")
    print(kmeans.cluster_centers_)

    pca = PCA(n_components=2)
    pca.fit(X)
    X_new = pca.transform(X)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=Y)
    plt.savefig('PCA.png')
    plt.show()

    lda = LatentDirichletAllocation(n_components=2)
    lda.fit(X)
    X_new = lda.transform(X)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=Y)
    plt.savefig('LDA.png')
    plt.show()
