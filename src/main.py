from transformers import pipeline
import os
import configparser
import json


def get_parent_path() -> str:
    """ Returns the parent path of the current working directory."""

    path = os.getcwd()
    parent = os.path.dirname(path)
    return parent.replace("\\", "/")


def get_model_data_path(parent: str) -> tuple:
    """ Reads the config file and forms absolute model and data paths.
        returns a tuple of the strings.(model_path, data_path)
    """
    config = configparser.ConfigParser()
    config.read(f"{parent}/config/config.ini")

    relative_model_path = config['Paths']['model_path']
    relative_data_path = config['Paths']['data_path']

    return parent + relative_model_path, parent + relative_data_path


def get_labels(parent:str) -> tuple:
    """ Reads the config file and forms a tuple of the labels for sentiment and intention.
        returns a tuple of the lists.(sentiment_labels, intention_labels)"""
    config = configparser.ConfigParser()
    config.read(f"{parent}/config/config.ini")

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


def main():
    parent = get_parent_path()
    model_path, data_path = get_model_data_path(parent)
    candidate_sentiments, candidate_intentions = get_labels(parent)

    # Load the data
    with open(data_path) as f:
        data = json.load(f)

    classifier = pipeline(task="zero-shot-classification",
                          model=model_path)

    # Write the results to the console
    for sequence in data:
        print(f"Sequence: {sequence['message']}")
        print(f"Sentiment is: {create_output(sequence['message'], candidate_sentiments, classifier)}")
        print(f"Intention is: {create_output(sequence['message'], candidate_intentions, classifier)}\n")


