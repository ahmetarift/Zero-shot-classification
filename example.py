from zero_shot_classifier.inference import get_path, get_model_data_path, get_labels, create_output
import json
from transformers import pipeline

path = get_path()
model_path, data_path = get_model_data_path(path)
candidate_sentiments, candidate_intentions = get_labels(path)

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