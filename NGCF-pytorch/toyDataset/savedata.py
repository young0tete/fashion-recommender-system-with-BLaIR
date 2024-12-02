import pandas as pd
from os import path
import json

data_dir = path.join(path.dirname(__file__), 'AmazonFashion')
data_dir = path.join(data_dir, 'res')

def saveOutput(ngcf):
    f = path.join(data_dir, 'top_k_predictions_rating.json')
    with open(f, 'w') as json_file:
        json.dump(ngcf, json_file)

def saveStatistics(statistics):
    f = path.join(data_dir, 'statistics_rating.json')
    with open(f, 'w') as json_file:
        json.dump({key: value.tolist() for key, value in statistics.items()}, json_file)