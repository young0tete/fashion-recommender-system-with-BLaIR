import numpy as np
import torch

def recallAtK(topKItems, truthItems):
    hits = torch.isin(topKItems, truthItems)
    hit_count = hits.sum().item()

    # Compute recall as hits / len(truthItems)
    recall = hit_count / len(truthItems) if len(truthItems) > 0 else 0.0
    return recall


def apAtK(topKItems, truthItems):
    hits = torch.isin(topKItems, truthItems).float()
    ranks = torch.arange(1, len(topKItems) + 1).cuda().float()

    precision = (torch.cumsum(hits, dim=0) / ranks)
    averagePrecision = precision.sum() / len(topKItems) if len(topKItems) > 0 else 0.0

    return averagePrecision.item()


def testOneUser(predictionPairs, truthPairs, KS):
    ret = {'recall': np.zeros(len(KS)), 'ap': np.zeros(len(KS))}

    predictedRatings = predictionPairs[:, 1]
    truthItems = truthPairs[:, 0]

    for i, topK in enumerate(KS):
        topKPairs = predictionPairs[torch.topk(predictedRatings, topK).indices]
        topKItems = topKPairs[:, 0]

        ret['recall'][i] = recallAtK(topKItems, truthItems)
        ret['ap'][i] = apAtK(topKItems, truthItems)

    return ret