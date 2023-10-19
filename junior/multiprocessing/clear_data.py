import re
from string import punctuation
from time import sleep
from typing import List

import pandas as pd
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# def parallel(n_jobs=-1):
#     """Parallel computing"""
#     result = Parallel(
#         n_jobs=n_jobs, backend="multiprocessing", verbose=5 * n_jobs
#     )(delayed(sleep)(0.2) for _ in range(50))
#     return result


# print(parallel(n_jobs=1))
# print(parallel(n_jobs=2))


def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    text = str(text)
    text = re.sub(r"https?://[^,\s]+,?", "", text)
    text = re.sub(r"@[^,\s]+,?", "", text)

    stop_words = stopwords.words("english")
    transform_text = text.translate(str.maketrans("", "", punctuation))
    transform_text = re.sub(" +", " ", transform_text)

    text_tokens = word_tokenize(transform_text)

    lemma_text = [lemmatizer.lemmatize(word.lower()) for word in text_tokens]
    cleaned_text = " ".join(
        [str(word) for word in lemma_text if word not in stop_words]
    )
    return cleaned_text


def clean_texts_parallel(texts, n_jobs=-1):
    result = Parallel(
        n_jobs=n_jobs, backend="multiprocessing", verbose=5 * n_jobs
    )(delayed(clean_text)(text) for text in texts)
    return result


def clear_data(source_path: str, target_path: str, n_jobs: int):
    """Baseline process df

    Parameters
    ----------
    source_path : str
        Path to load dataframe from

    target_path : str
        Path to save dataframe to
    """
    data = pd.read_parquet(source_path)
    data = data.copy().dropna().reset_index(drop=True)

    cleaned_text_list = clean_texts_parallel(data["text"], n_jobs=n_jobs)

    data["cleaned_text"] = cleaned_text_list
    data.to_parquet(target_path)
