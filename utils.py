#Every sub-function is defined here.
import numpy as np
from umap import UMAP
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from typing import List
from numpy import ndarray


EMBEDDING_MODEL = "multi-qa-MiniLM-L6-dot-v1"
RANDOM_SEED = 69420 #ha ha funni number
SUBTOPIC_THRESHOLD = 0.3


def fit_transform_new_model(docs: List[str], embeddings: ndarray = None):
    '''
    Creating a new model that fits the documents, returning the model and the topics for the docs

    Parameters
    ----------
        docs (List[str]): The list of string docs that need to be fitted to the model
        embeddings (ndarray): The pre-defined embeddings for the docs (optional)
 
    Returns
    -------
        topic_model (BERTopic): The topic model
        topics_for_each_document (List[List[int]]): The lists that contains possible topics of each documents 
    '''
    sentence_model = SentenceTransformer("multi-qa-MiniLM-L6-dot-v1")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)
    topic_model = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='euclidean', random_state=RANDOM_SEED))
    topics, probs = topic_model.fit_transform(docs, embeddings)
    topic_model.merge_topics(docs, [0, -1]) #Topic 0 is also a group of outlier, as stated in https://github.com/MaartenGr/BERTopic/discussions/1164 

    topic_distr, _ = topic_model.approximate_distribution(docs)
    #current problem: topic_distr works incorrectly for our dataset for some reason ?
    topics_for_each_document = _topic_from_topic_distr(topic_distr, threshold = SUBTOPIC_THRESHOLD, topics_from_model = topics)
    return topic_model, topics_for_each_document


def _topic_from_topic_distr(topic_distr: ndarray, threshold: float = 0.2, topics_from_model = None):
    '''
    For each document, returns a list of possible topics 

    Parameters
    ----------
        topic_distr: array of probability-that-this-document-belongs-to-this
        threshold (float): The minimum acceptable probability
 
    Returns
    -------
        topics_list (List[List[int]]): The lists that contains possible topics of each documents 
    '''
    topics_list = []
    for i in range(topic_distr.shape[0]):
        row = topic_distr[i]
        # Find indexes where data surpasses the threshold
        indexes_above_threshold = np.where(row > threshold)[0]
        # Sort indexes based on the corresponding data values in descending order
        sorted_indexes = sorted(indexes_above_threshold, key=lambda i: row[i], reverse=True)
        if len(sorted_indexes != 0):
            topics_list.append(sorted_indexes)
        elif topics_from_model is not None:
            topics_list.append([topics_from_model[i]])
        else:
            topics_list.append([-1])
    return topics_list


def save_topic_model(topic_model: BERTopic, path: str = "./models/model"):

    '''
    Save the topic model to the specified path.

    Parameters
    ----------
        topic_model (BERTopic): The model.
        path (str): The path to save the model.
 
    Returns
    -------
        None
    '''
    embedding_model = EMBEDDING_MODEL
    topic_model.save(path, serialization="pytorch", save_ctfidf=True, save_embedding_model=embedding_model)


def load_topic_model(path: str = "./models/model"):

    '''
    Load the topic model from the specified path.

    Parameters
    ----------
        path (str): The path to save the model.
    Returns
    -------
        topic_model: The loaded model.
    '''
    topic_model = BERTopic.load(path)
    #reloading the umap model to prevent randomness
    topic_model.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='euclidean', random_state=RANDOM_SEED)
    return topic_model


def predict_new_document(topic_model: BERTopic, docs: List[str], embeddings: ndarray = None):
    '''
    Create new results from old model

    Parameters
    ----------
        topic_model (BERTopic): The model.
        docs (List[str]): The list of string docs that need to be fitted to the model
        embeddings (ndarray): The pre-defined embeddings for the docs (optional)
    Returns
    -------
        topics_for_each_document (List[List[int]]): The lists that contains possible topics of each documents 
    '''
    # return topic_model.transform(docs, embeddings)
    topic_distr, _ = topic_model.approximate_distribution(docs)
    #current problem: topic_distr works incorrectly for our dataset for some reason ?
    topics_for_each_document = _topic_from_topic_distr(topic_distr, threshold = SUBTOPIC_THRESHOLD)
    return topics_for_each_document