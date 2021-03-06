# -*- coding:utf-8 -*-
__author__ = 'c11tch'
from GraphMng import GraphMng
from loadDataFromLocal import loadDataFromLocal
import numpy as np


class localTest:

    dictUser2UID = {}
    dictUID2User = {}
    dictTag2TID = {}
    dictTID2Tag = {}
    dictItem2IID = {}
    dictIID2Item = {}
    valUsersNum = 0
    valTagsNum = 0
    valItemsNum = 0

    # 製作GM物件以管理Graph
    GraphMng = GraphMng()
    #  讀檔並存於localTest.list_locData中
    # list_locData = loadDataFromLocal(r'train.dat')
    list_locData = loadDataFromLocal(r'new_train.dat')

    """---------------------------------- Vertex Augment ---------------------------------- """

    """ 若無此ID，加入dict中，以供後續查詢。有就當沒事 """
    def AddNewUser(self, userID):
        if userID not in self.dictUser2UID:
            """ 讓儲存的數值改為0開始 以免發生'ValueError: row index exceeds matrix dimensions' """
            self.dictUser2UID[userID] = self.valUsersNum
            self.dictUID2User[self.valUsersNum] = userID
            self.valUsersNum += 1
            self.GraphMng.AddNewUser(userID)    # 將使用者加入至GraphMng物件中

    def AddNewTag(self, tagID):
        if tagID not in self.dictTag2TID:
            self.dictTag2TID[tagID] = self.valTagsNum
            self.dictTID2Tag[self.valTagsNum] = tagID
            self.valTagsNum += 1
            self.GraphMng.AddNewTag(tagID)    # 將物件加入至GraphMng物件中

    def AddNewItem(self, itemID):
        if itemID not in self.dictItem2IID:
            self.dictItem2IID[itemID] = self.valItemsNum
            self.dictIID2Item[self.valItemsNum] = itemID
            self.valItemsNum += 1
            self.GraphMng.AddNewItem(itemID)    # 將標籤加入至GraphMng物件中

    """ ---------------------------------- Edge Augment ---------------------------------- """
    """ 先確認點是否都已存入dict中，再做加邊動作 """
    def AddEdgeUser2Item(self,userID, itemID, Rating):
        self.AddNewUser(userID)
        self.AddNewItem(itemID)

        uid = self.dictUser2UID.get(userID)
        iid = self.dictItem2IID.get(itemID)
        self.GraphMng.AddEdgeUID2IID(uid, iid, Rating)

    def AddEdgeUser2Tag(self, userID,tagID):
        self.AddNewUser(userID)
        self.AddNewTag(tagID)

        uid = self.dictUser2UID.get(userID)
        tid = self.dictTag2TID.get(tagID)
        self.GraphMng.AddEdgeUID2TID(uid, tid)

    def AddEdgeItem2Tag(self, itemID, tagID):
        self.AddNewItem(itemID)
        self.AddNewTag(tagID)

        iid = self.dictItem2IID.get(itemID)
        tid = self.dictTag2TID.get(tagID)
        self.GraphMng.AddEdgeIID2TID(iid, tid)

    """ ---------------------------------- To Generate Graphs ---------------------------------- """
    """ 將讀檔後的資料存入Graphs """
    def makeGraphFromData(self, datas):
        valUserID = 0
        valItemID = 0

        count = 0
        for data in datas:
            count += 1
            valAID = int(data[1])

            """ 因資料已排序過，若itemID有改變了才新增使用者以及物件 """
            if valAID != valItemID:
                valUserID = int(data[0])
                valItemID = int(data[1])
                valRating = int(data[2])

                """新增使用者與物件以及他們之間的關係"""
                self.AddNewUser(valUserID)
                self.AddNewItem(valItemID)
                self.AddEdgeUser2Item(valUserID, valItemID, valRating)

            valTagID = int(data[3])
            """新增標籤以及與使用者/物件之間的關係"""
            self.AddNewTag(valTagID)
            self.AddEdgeUser2Tag(valUserID, valTagID)
            self.AddEdgeItem2Tag(valItemID, valTagID)

        """ 完成矩陣 """
        self.GraphMng.makeAllMat()
        self.GraphMng.UpdateTransMatrix()
        # m = self.GraphMng.ColumnNomalization(self.GraphMng.matrixTrans)    # Column Normalization
        # m = self.GraphMng.RowNomalization(self.GraphMng.matrixTrans)    # Row Normalizatin
        # m.tolil()[0,0] = 1    # 改變數值時 lil 格式比 csr格式有效率
        # print m.todense()
        """ ---------------------------到此為止無問題-------------------------------------------- """
        self.run()

    def run(self):
        vecQueryUID = np.array([2])
        vecQueryUID = [self.dictUser2UID[index] for index in vecQueryUID]
        vecQueryIID = np.array([])
        vecQueryIID = [self.dictItem2IID[index] for index in vecQueryIID]
        vecQueryTID = np.array([3])
        vecQueryTID = [self.dictTag2TID[index] for index in vecQueryTID]
        resultType = 3
        normType = 1
        k = 10
        resultsIndex = self.GraphMng.ExecRankingProcess(vecQueryUID, vecQueryIID, vecQueryTID, resultType, normType, k)

        # resultsIndex = [i+1 for i in resultsIndex]
        print('[LT][run] resultType = ', resultType)
        if resultType == 1:
            print('[LT][run] recommand users : ')
            for index in resultsIndex:
                print(self.dictUID2User[index])
        elif resultsIndex == 2:
            print('[LT][run] recommand items : ')
            for index in resultsIndex:
                print(self.dictIID2Item[index])
        else:
            print('[LT][run] recommand tags : ')
            for index in resultsIndex:
                print(self.dictTID2Tag[index])


    def __init__(self):
        self.list_locData = self.makeGraphFromData(self.list_locData)

if __name__ == '__main__':
    localTest = localTest()