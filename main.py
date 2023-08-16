from transformers import pipeline
import numpy as np



if __name__ == '__main__':
    classifier = pipeline("zero-shot-classification",
                          model="./resources")

    sequence_to_classify = "one day Bilge and I will see the world."
    candidate_labels = ['travel', 'dancing', 'playing soccer']
    print(classifier(sequence_to_classify, candidate_labels))

    print(classifier(sequence_to_classify, candidate_labels)['labels'][0])


    # add tests




