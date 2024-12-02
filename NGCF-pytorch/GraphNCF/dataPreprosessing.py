from collections import defaultdict

from torch.utils.data import Dataset
import numpy as np
import pandas as pd

# class ML1K(Dataset):
#
#     def __init__(self,rt):
#         super(Dataset,self).__init__()
#         self.uId = list(rt['userId'])
#         self.iId = list(rt['itemId'])
#         self.rt = list(rt['rating'])
#
#     def __len__(self):
#         return len(self.uId)
#
#     def __getitem__(self, item):
#         return (self.uId[item],self.iId[item],self.rt[item])

class PreprocessedData(Dataset):

    def __init__(self, uIndices, iIndices, ratings):
        super(Dataset, self).__init__()

        self.uIndices = uIndices
        self.iIndices = iIndices
        self.rt = ratings
        self.adjList = defaultdict(list)
        for u, i, r in zip(self.uIndices, self.iIndices, self.rt):
            self.adjList[u].append((i, r))

    def __len__(self):
        return len(self.uIndices)

    def __getitem__(self, i):
        return self.uIndices[i], self.iIndices[i], self.rt[i]

class Preprocessor:
    def __init__(self, raw_train, raw_test):
        # solesie: user_id and item_id are String format, so convert them to 0-based index.
        uIdStr = pd.concat([raw_train['user_id'], raw_test['user_id']], axis=0, ignore_index=True)
        iIdStr = pd.concat([raw_train['item_id'], raw_test['item_id']], axis=0, ignore_index=True)

        uset = set(uIdStr)
        iset = set(iIdStr)

        self.__uStr2Int = {user: idx for idx, user in enumerate(uset)}
        self.__uInt2Str = {idx: user for idx, user in enumerate(uset)}
        self.__iStr2Int = {item: idx for idx, item in enumerate(iset)}
        self.__iInt2Str = {idx: item for idx, item in enumerate(iset)}

        self.preprocessedTrain = PreprocessedData(np.array([self.__uStr2Int[user] for user in raw_train['user_id']])
                                                  , np.array([self.__iStr2Int[item] for item in raw_train['item_id']])
                                                  , np.array(np.array(raw_train['rating'])))
        self.preprocessedTest = PreprocessedData(np.array([self.__uStr2Int[user] for user in raw_test['user_id']])
                                                 , np.array([self.__iStr2Int[item] for item in raw_test['item_id']])
                                                 , np.array(np.array(raw_test['rating'])))
        self.userNum = len(uset)
        self.itemNum = len(iset)

    def restore(self, uIdx, iIdx):
        return self.__uInt2Str[uIdx], self.__iInt2Str[iIdx]

    def restoreUser(self, uIdx):
        return self.__uInt2Str[uIdx]

    def restoreItem(self, iIdx):
        return self.__iInt2Str[iIdx]