from transformers import pipeline
import numpy as np



if __name__ == '__main__':
    classifier = pipeline("zero-shot-classification",
                          model="./resources")

    sequence_to_classify = "Certainly, the iPhone 14 comes with a range of new features. Improved camera, faster processor, and longer battery life, to name a few."
    candidate_sentiments = ['positive', 'neutral', 'negative']
    print(classifier(sequence_to_classify, candidate_sentiments))
    print(classifier(sequence_to_classify, candidate_sentiments)['labels'][0])

    print('Intention classifier:')
    candidate_intentions = ['Greetings and small talks', 'Discuss features of product', 'Price and discounts']

    print(classifier(sequence_to_classify, candidate_intentions))
    print(classifier(sequence_to_classify, candidate_intentions)['labels'][0])




