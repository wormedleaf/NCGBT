import numpy as np
from gensim.models import Word2Vec
# from tensorflow.python.keras.utils import np_utils

np.random.seed(1337)


def filter_with_vocabulary(sentences, labels, vocabulary, min_sentence_length=15):
    updated_sentences = []
    updated_labels = []
    for label_index, sentence in enumerate(sentences):
        current_train_filter = [word for word in sentence if word in vocabulary]
        if len(current_train_filter) >= min_sentence_length:
            updated_sentences.append(current_train_filter)
            updated_labels.append(labels[label_index])

    return updated_sentences, updated_labels


def filter_with_labels(sentences, labels, known_labels):
    known_labels_unique = set(known_labels)
    labels_unique = set(labels)
    unwanted_labels = list(labels_unique - known_labels_unique)
    updated_sentences = []
    updated_labels = []
    for j in range(len(labels)):
        if labels[j] not in unwanted_labels:
            updated_sentences.append(sentences[j])
            updated_labels.append(labels[j])

    return updated_sentences, updated_labels


def load_data(dataset_name, min_train_samples_per_class, merged_wordvec_model):
    wordvec_model = None
    if merged_wordvec_model:
        wordvec_model = Word2Vec.load("./data/merged/word2vec.model")
    else:
        wordvec_model = Word2Vec.load("./data/{0}/word2vec.model".format(dataset_name))
    all_data = np.load(
        "./data/{0}/all_data_{1}.npy".format(dataset_name, min_train_samples_per_class),
        allow_pickle=True,
    )
    all_owner = np.load(
        "./data/{0}/all_owner_{1}.npy".format(
            dataset_name, min_train_samples_per_class
        ),
        allow_pickle=True,
    )

    return wordvec_model, all_data, all_owner


def embedding(
        sentences,
        labels,
        unique_labels,
        wordvec_model,
        vocabulary,
        max_sentence_len=50,
        embed_size_word2vec=200,
):
    X = np.empty(
        shape=[len(sentences), max_sentence_len, embed_size_word2vec], dtype="float32"
    )
    Y = np.empty(shape=[len(labels), 1], dtype="int32")
    for j, curr_row in enumerate(sentences):
        sequence_cnt = 0
        for item in curr_row:
            if item in vocabulary:
                X[j, sequence_cnt, :] = wordvec_model.wv[item]
                sequence_cnt = sequence_cnt + 1
                if sequence_cnt == max_sentence_len - 1:
                    break
        for k in range(sequence_cnt, max_sentence_len):
            X[j, k, :] = np.zeros((1, embed_size_word2vec))
        Y[j, 0] = unique_labels.index(labels[j])

    return X, Y


def chronological_cv(
        dataset_name, min_train_samples_per_class, num_cv, merged_wordvec_model
):
    wordvec_model, sentences, labels = load_data(
        dataset_name, min_train_samples_per_class, merged_wordvec_model
    )
    vocabulary = wordvec_model.wv.index_to_key
    splitLength = len(sentences) // (num_cv + 1)
    for i in range(1, num_cv + 1):
        train_data = sentences[: i * splitLength - 1]
        test_data = sentences[i * splitLength: (i + 1) * splitLength - 1]
        train_owner = labels[: i * splitLength - 1]
        test_owner = labels[i * splitLength: (i + 1) * splitLength - 1]
        updated_train_data, updated_train_owner = filter_with_vocabulary(
            train_data, train_owner, vocabulary
        )
        final_test_data, final_test_owner = filter_with_vocabulary(
            test_data, test_owner, vocabulary
        )
        updated_test_data, updated_test_owner = filter_with_labels(
            final_test_data, final_test_owner, updated_train_owner
        )
        unique_train_label = list(set(updated_train_owner))
        classes = np.array(unique_train_label)
        X_train, Y_train = embedding(
            updated_train_data,
            updated_train_owner,
            unique_train_label,
            wordvec_model,
            vocabulary,
        )
        X_test, Y_test = embedding(
            updated_test_data,
            updated_test_owner,
            unique_train_label,
            wordvec_model,
            vocabulary,
        )
        # y_train = np_utils.to_categorical(Y_train, len(unique_train_label))
        # y_test = np_utils.to_categorical(Y_test, len(unique_train_label))

        yield X_train, Y_train, X_test, Y_test, classes

X_train, Y_train, X_test, Y_test, classes = chronological_cv ('chrome', 10, 10, merged_wordvec_model = True)
print(X_train, Y_train, X_test, Y_test, classes)
