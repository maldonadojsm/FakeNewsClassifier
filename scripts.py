# !/usr/bin/env python
# title           :scripts.py
# description     :Houses several scripts used during ML training and model serving.
# author          :Sebastian Maldonado
# date            :7/22/2020
# version         :0.0
# usage           :SEE README.md
# notes           :Enter Notes Here
# python_version  :3.7,7
# conda_version   :4.8.2
# tf_version      :1.14
# =================================================================================================================
from newspaper import Article
import re
from nltk.corpus import stopwords
import tensorflow as tf
import pickle


def check_size(d1: int, d2: int) -> bool:
    """
    Check if dataset size is the same after dataset cleaning
    :param d1: Original Dataset Size
    :param d2: Cleaned Dataset Size
    :return: Return boolean. True
    """
    return True if d1 == d2 else False


def scrape_article(url: str) -> list:
    """
    Downloads and parses news article found in URL
    :param url: URL page of news article
    :return: List containing text that will be used for inference along with article elements.
    """
    article = Article(url)
    article.download()
    article.parse()
    ml_text = article.title + article.text
    contents = list([article.title, article.text, article.top_image, ml_text])
    return contents


def clean_text(article: str) -> list:
    """
    Prepare news article for ML inference
    :param article: new article in string format
    :return: Cleaned article text
    """
    cleaned_text = list()
    # Remove non-alphanumeric characters
    result = re.sub('[^a-zA-Z]', ' ', article)
    # Convert remaining characters to lowercase
    result = result.lower()
    result = result.split()

    # Remove stopwords from dataset (NLTK)
    result = [i for i in result if i not in set(stopwords.words('english'))]
    cleaned_text.append(" ".join(result))

    return cleaned_text


def perform_inference(text) -> float:
    """
    Performs inference, using pre-trained ML model and tokenizer from training, for submitted news article.
    :param text: Cleaned text return from clean_text()
    :return: Returns inference probability whether news article isn't fake news (float)
    """
    # Load Tokenizer
    with open('../input/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load LSTM Trained Model
    model = tf.keras.models.load_model('../saved_models/LSTM/LSTM_Model.hdf5')
    text_sequences = tokenizer.texts_to_sequences(text)
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(text_sequences, maxlen=400)

    prediction = model.predict(padded_sequence)

    return prediction
