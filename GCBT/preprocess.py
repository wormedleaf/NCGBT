import json
import nltk
import re
import string

import numpy as np
from gensim.models import Word2Vec

np.random.seed(1337)
# nltk.download('punkt')

def clean_word_list(item):
    current_title = item["issue_title"].replace("\r", " ")
    current_desc = item["description"].replace("\r", " ")

    current_desc = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        current_desc,
    )

    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]

    current_desc = re.sub(r"(\w+)0x\w+", "", current_desc)
    current_title = re.sub(r"(\w+)0x\w+", "", current_title)

    current_desc = current_desc.lower()
    current_title = current_title.lower()

    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)

    current_desc_filter = [
        word.strip(string.punctuation) for word in current_desc_tokens
    ]
    current_title_filter = [
        word.strip(string.punctuation) for word in current_title_tokens
    ]
    current_data = current_title_filter + current_desc_filter
    current_data = [x for x in current_data if x]  # list(filter(None, current_data))

    return current_data


def preprocess_dataset(dataset_name):
    print("Preprocessing {0} dataset: Start".format(dataset_name))
    open_bugs_json = "./data/{0}/deep_data.json".format(dataset_name)

    min_word_frequency_word2vec = 5
    embed_size_word2vec = 200
    context_window_word2vec = 5

    with open(open_bugs_json, encoding='UTF-8') as data_file:
        text = data_file.read()
        text = text.replace('" : NULL', '" : "NULL"')
        data = json.loads(text, strict=False)

    all_data = []
    for item in data:
        current_data = clean_word_list(item)
        all_data.append(current_data)

    print("Preprocessing {0} dataset: Word2Vec model".format(dataset_name))
    wordvec_model = Word2Vec(
        all_data,
        min_count=min_word_frequency_word2vec,
        vector_size=embed_size_word2vec,
        window=context_window_word2vec,
    )
    wordvec_model.save("./data/{0}/word2vec.model".format(dataset_name))
    for min_train_samples_per_class in [0, 5, 10, 20]:
        print(
            "Preprocessing {0} dataset: Classifier data {1}".format(
                dataset_name, min_train_samples_per_class
            )
        )

        closed_bugs_json = "./data/{0}/classifier_data_{1}.json".format(
            dataset_name, min_train_samples_per_class
        )

        with open(closed_bugs_json, encoding='UTF-8') as data_file:
            text = data_file.read()
            # Fix json files for mozilla core and mozilla firefox
            text = text.replace('" : NULL', '" : "NULL"')
            data = json.loads(text, strict=False)

        all_data = []
        all_owner = []
        for item in data:
            current_data = clean_word_list(item)
            all_data.append(current_data)
            all_owner.append(item["owner"])

        np.save(
            "./data/{0}/all_data_{1}.npy".format(
                dataset_name, min_train_samples_per_class
            ),
            all_data,
        )
        np.save(
            "./data/{0}/all_owner_{1}.npy".format(
                dataset_name, min_train_samples_per_class
            ),
            all_owner,
        )


def read_json_and_clean(filename):
    with open(filename, encoding='UTF-8') as data_file:
        text = data_file.read()
        text = text.replace('" : NULL', '" : "NULL"')
        data = json.loads(text, strict=False)

    all_data = []
    for item in data:
        current_data = clean_word_list(item)
        all_data.append(current_data)

    return all_data


def wordvec_all_datasets_merged():
    print("Preprocessing all datasets merged: Word2Vec model")
    open_bugs_json_gc = "./data/{0}/deep_data.json".format("google_chromium")
    open_bugs_json_mc = "./data/{0}/deep_data.json".format("mozilla_core")
    open_bugs_json_mf = "./data/{0}/deep_data.json".format("mozilla_firefox")

    all_data_gc = read_json_and_clean(open_bugs_json_gc)
    all_data_mc = read_json_and_clean(open_bugs_json_mc)
    all_data_mf = read_json_and_clean(open_bugs_json_mf)

    all_data_merged = all_data_gc + all_data_mc + all_data_mf

    min_word_frequency_word2vec = 5
    embed_size_word2vec = 200
    context_window_word2vec = 5

    wordvec_model = Word2Vec(
        all_data_merged,
        min_count=min_word_frequency_word2vec,
        size=embed_size_word2vec,
        window=context_window_word2vec,
    )

    wordvec_model.save("./data/merged/word2vec.model")


if __name__ == '__main__':
    preprocess_dataset("mozilla_firefox")
    preprocess_dataset("mozilla_core")
    preprocess_dataset("google_chromium")
