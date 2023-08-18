from transformers import pipeline
import os
import configparser
import json


def get_path() -> str:
    """ Returns the path of the current working directory with correct unicode format."""

    path = os.getcwd()
    return path.replace("\\", "/")


def get_model_data_path(path: str) -> tuple:
    """ Reads the config file and forms absolute model and data paths.
        Args:
            path: current working directory
        Returns:
            a tuple of the lists.(model_path, data path)
    """
    config = configparser.ConfigParser()
    config.read(f"{path}/config/config.ini")

    relative_model_path = config['Paths']['model_path']
    relative_data_path = config['Paths']['data_path']

    return path + relative_model_path, path + relative_data_path


def get_labels(path:str) -> tuple:
    """ Reads the config file and forms a tuple of the labels for sentiment and intention.
        Args:
            path: current working directory
        Returns:
            a tuple of the lists.(sentiment_labels, intention_labels)
    """
    config = configparser.ConfigParser()
    config.read(f"{path}/config/config.ini")

    return json.loads(config['Labels']['sentiment']), json.loads(config['Labels']['intent'])


def create_output(message: str , candidates: list, classifier: pipeline) -> str:
    """ Makes the inference with classifier.

        Args:
            message: A string of the message to be classified.
            candidates: A list of the candidate labels.
            classifier: A pipeline object(Transformers library) for the classifier.
        Returns:
            A string of label with its probability.
    """
    full_output = classifier(message, candidates)
    label = full_output['labels'][0]
    probability = full_output['scores'][0]
    return f"{label} with probability = {probability}"
