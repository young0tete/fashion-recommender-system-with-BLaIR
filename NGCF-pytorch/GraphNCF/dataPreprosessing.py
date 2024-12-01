from torch.utils.data import Dataset
import numpy as np

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

    def __init__(self, rt):
        super(Dataset, self).__init__()

        # solesie: user_id and item_id are String format, so convert them to 0-based index.
        self.__uIdStr = np.array(rt['user_id'])
        self.__iIdStr = np.array(rt['item_id'])

        uset = set(self.__uIdStr)
        iset = set(self.__iIdStr)

        self.__uStr2Int = {user: idx for idx, user in enumerate(uset)}
        self.__uInt2Str = {idx: user for idx, user in enumerate(uset)}
        self.__iStr2Int = {item: idx for idx, item in enumerate(iset)}
        self.__iInt2Str = {idx: item for idx, item in enumerate(iset)}

        self.uIdInt = np.array([self.__uStr2Int[user] for user in self.__uIdStr])
        self.iIdInt = np.array([self.__iStr2Int[item] for item in self.__iIdStr])
        self.rt = np.array(rt['rating'])

        self.userNum = len(uset)
        self.itemNum = len(iset)

    def __len__(self):
        return len(self.uIdInt)

    def __getitem__(self, i):
        return self.uIdInt[i], self.iIdInt[i], self.rt[i]

    def restore(self, item):
        return self.__uInt2Str[item.first], self.__iInt2Str[item.second], self.rt[item.third]
