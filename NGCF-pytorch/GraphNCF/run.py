import torch
from torch import nn as nn
from toyDataset.loaddata import loadTrainData, loadTestData
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from numpy import diag
from GraphNCF.GCFmodel import GCF
from torch.utils.data import DataLoader, TensorDataset
from GraphNCF.dataPreprosessing import PreprocessedData, Preprocessor
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
from GraphNCF.GCFmodel import SVD
from GraphNCF.GCFmodel import NCF
from GraphNCF.metrics import testOneUser

raw_train = loadTrainData()
raw_test = loadTestData()

para = {
    'epoch':60,
    'lr':0.01,
    'batch_size':2048
    # solesie: Train/Test data is statically fixed, so no use more.
    # 'train':0.8
}

pp = Preprocessor(raw_train,raw_test)

model = GCF(pp.userNum, pp.itemNum, pp.preprocessedTrain, 64, layers=[64, 64, 64]).cuda()
# model = SVD(userNum,itemNum,50).cuda()
# model = NCF(userNum,itemNum,64,layers=[128,64,32,16,8]).cuda()
optim = Adam(model.parameters(), lr=para['lr'],weight_decay=0.001)
lossfn = MSELoss()

# solesie: train
traindl = DataLoader(pp.preprocessedTrain, batch_size=para['batch_size'], shuffle=True, pin_memory=True)
for i in range(para['epoch']):
    for id,batch in enumerate(traindl):
        print('epoch:',i,' batch:',id)
        optim.zero_grad()
        prediction = model(batch[0].cuda(), batch[1].cuda())
        loss = lossfn(batch[2].float().cuda(),prediction)
        loss.backward()
        optim.step()
        print(loss)

# solesie: test
KS = [5, 10, 20, 100]
result = {'recall': np.zeros(len(KS)), 'ap': np.zeros(len(KS))}
for u in range(pp.userNum):
    truthPairs = torch.tensor(pp.preprocessedTest.adjList[u]).float().cuda()

    # solesie: It is possible that there is no way to evaluate the user
    # because of the very sparse data.
    if truthPairs.numel() == 0:
        continue

    allItems = torch.arange(pp.itemNum).cuda()
    exclude = torch.tensor([item for item, _ in pp.preprocessedTrain.adjList[u]]).cuda()

    remaining = allItems[~torch.isin(allItems, exclude)]
    curUsers = torch.full((remaining.size(0), ), u, dtype=torch.int).cuda()
    predictions = model(curUsers, remaining)

    predictionPairs = torch.stack([remaining, predictions], dim=1)

    metrics = testOneUser(predictionPairs, truthPairs, KS)

    for k in result.keys():
        result[k] += metrics[k] / pp.userNum

    print("user:", u, ": ", metrics)

print(result)

testdl = DataLoader(pp.preprocessedTest,batch_size=len(pp.preprocessedTest),)
for data in testdl:
    prediction = model(data[0].cuda(),data[1].cuda())
    loss = lossfn(data[2].float().cuda(),prediction)
    print(loss) # MSEloss