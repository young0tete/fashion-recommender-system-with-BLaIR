import pandas as pd
from os import path
import json

data_dir = path.join(path.dirname(__file__), 'AmazonFashion')
data_dir = path.join(data_dir, 'res')

def saveOutput(ngcf, useRating, useSentiment):
    if useRating and useSentiment:
        filename = 'top_k_predictions_rating_sentiment.json'
    elif useRating:
        filename = 'top_k_predictions_rating.json'
    elif useSentiment:
        filename = 'top_k_predictions_sentiment.json'
    else:
        raise ValueError("Invalid condition: Neither ratings nor sentiment are used")
    f = path.join(data_dir, filename)
    with open(f, 'w') as json_file:
        json.dump(ngcf, json_file)

def saveStatistics(statistics, useRating, useSentiment):
    if useRating and useSentiment:
        filename = 'statistics_rating_sentiment.json'
    elif useRating:
        filename = 'statistics_rating.json'
    elif useSentiment:
        filename = 'statistics_sentiment.json'
    else:
        raise ValueError("Invalid condition: Neither ratings nor sentiment are used")
    f = path.join(data_dir, filename)
    with open(f, 'w') as json_file:
        json.dump({key: value.tolist() for key, value in statistics.items()}, json_file)