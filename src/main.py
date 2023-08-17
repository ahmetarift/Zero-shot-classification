from transformers import pipeline
import os
import configparser
import json
import numpy as np


def get_parent_path():
    path = os.getcwd()
    parent = os.path.dirname(path)
    return parent.replace("\\", "/")


def get_model_data_path(parent):
    config = configparser.ConfigParser()
    config.read(f"{parent}/config/config.ini")

    relative_model_path = config['Paths']['model_path']
    relative_data_path = config['Paths']['data_path']

    return parent + relative_model_path, parent + relative_data_path


def get_labels(parent):
    config = configparser.ConfigParser()
    config.read(f"{parent}/config/config.ini")

    return json.loads(config['Labels']['sentiment']), json.loads(config['Labels']['intent'])


def create_output(message, candidates):
    full_output = classifier(message, candidates)
    label = full_output['labels'][0]
    probability = full_output['scores'][0]
    return f"{label} with probability = {probability}"


if __name__ == '__main__':
    parent = get_parent_path()
    model_path, data_path = get_model_data_path(parent)
    candidate_sentiments, candidate_intentions = get_labels(parent)

    with open(data_path) as f:
        data = json.load(f)

    classifier = pipeline(task="zero-shot-classification",
                          model=model_path)

    for sequence in data:
        print(f"Sequence: {sequence['message']}")
        print(f"Sentiment is: {create_output(sequence['message'], candidate_sentiments)}")
        print(f"Intention is: {create_output(sequence['message'], candidate_intentions)}\n")

