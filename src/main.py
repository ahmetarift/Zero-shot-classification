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


if __name__ == '__main__':
    parent = get_parent_path()
    model_path, data_path = get_model_data_path(parent)
    candidate_sentiments, candidate_intentions = get_labels(parent)

    with open(data_path) as f:
        data = json.load(f)

    classifier = pipeline(task="zero-shot-classification",
                          model=model_path)



    print(candidate_sentiments)
    print(candidate_intentions)


    sequence_to_classify = "Certainly, the iPhone 14 comes with a range of new features. Improved camera, faster processor, and longer battery life, to name a few."
    candidate_sentiments = ['positive', 'neutral', 'negative']
    print(classifier(sequence_to_classify, candidate_sentiments))
    print(classifier(sequence_to_classify, candidate_sentiments)['labels'][0])

    print('Intention classifier:')
    candidate_intentions = ['Greetings and farewells', 'Discuss features of product', 'Price and discounts']

    print(classifier(sequence_to_classify, candidate_intentions))
    print(classifier(sequence_to_classify, candidate_intentions)['labels'][0])
