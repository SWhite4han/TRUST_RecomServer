# -*- coding:utf-8 -*-
__author__ = 'c11tch'
from InfRank import InfRank
from scipy.sparse import lil_matrix, coo_matrix, eye, vstack, hstack
import numpy as np
import time
from queue import Queue
import logging
import logging.config


class GraphMng:

    def __init__(self):
        # 0*0的稀疏矩陣
        self.matrix_UU = lil_matrix((0, 0))
        self.matrix_UI = lil_matrix((0, 0))
        self.matrix_UT = lil_matrix((0, 0))
        self.matrix_IU = lil_matrix((0, 0))
        self.matrix_II = lil_matrix((0, 0))
        self.matrix_IT = lil_matrix((0, 0))
        self.matrix_TU = lil_matrix((0, 0))
        self.matrix_TI = lil_matrix((0, 0))
        self.matrix_TT = lil_matrix((0, 0))

        # 建造上述矩陣用的個別的list
        self.list_UI_uid = []
        self.list_UI_iid = []
        self.list_UI_rat = []
        self.list_UT_uid = []
        self.list_UT_tid = []
        self.list_UT_rat = []
        self.list_IT_iid = []
        self.list_IT_tid = []
        self.list_IT_rat = []

        self.matrixTrans = lil_matrix((0, 0))

        self.valUsersNum = 0
        self.valItemsNum = 0
        self.valTagsNum = 0

        self.valUU = 1./3
        self.valIU = 1./3
        self.valUI = 1./3
        self.valII = 1./3
        self.valUT = 1./3
        self.valIT = 1./3

        self.valQueryUser = 0.05
        self.valQueryItem = 0.45
        self.valQueryTag = 0.45

        self.valSigmoidScale = 30
        self.valIterationNum = 30
        self.valDiffusionRate = 0.8
        self.valInfDiffRate = 0.8
        self.valDampFactor = 0.5
        self.valInfDampFactor = 0.5

        self.objInfRank = InfRank(self.valInfDampFactor, self.valDampFactor, self.valInfDiffRate, self.valIterationNum)
        """ For check no one using this graph """
        self.userQueue = Queue()

        self.log = logging.getLogger('[RS].[GM]')

    def makeTestMatrix(self):

        a = [[1, 2	,3,	4,	5,	6,	7,	8,	9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24, 25, 26, 27],
            [28, 29, 30, 31, 32, 33, 34, 35, 36],
            [37, 38, 39, 40, 41, 42, 43, 44, 45],
            [46, 47, 48, 49, 50, 51, 52, 53, 54],
            [55, 56, 57, 58, 59, 60, 61, 62, 63],
            [64, 65, 66, 67, 68, 69, 70, 71, 72],
            [73, 74, 75, 76, 77, 78, 79, 80, 81]]

        a = lil_matrix(a)
        return a


    """ ---------------------------------- Parameters modification ---------------------------------- """
    def UpdateTransParam(self, vUU, vIU, vUI, vII, vUT, vIT):
        self.valUU = vUU
        self.valIU = vIU
        self.valUI = vUI
        self.valII = vII
        self.valUT = vUT
        self.valIT = vIT

    """---------------------------------- Vertex management ---------------------------------- """

    """ Add a user to the graph """
    def AddNewUser(self, userID):
        self.valUsersNum += 1

    def makeUserMat(self):
        self.matrix_UU = lil_matrix((self.valUsersNum, self.valUsersNum)).tocsr()

    """ Add a tag to the graph """
    def AddNewTag(self, tagID):
        self.valTagsNum += 1
        # self.matrix_TT = lil_matrix((self.valTagsNum,self.valTagsNum))

    def makeTagMat(self):
        self.matrix_TT = lil_matrix((self.valTagsNum, self.valTagsNum)).tocsr()

    """ Add an item to the graph """
    def AddNewItem(self, tagID):
        self.valItemsNum += 1
        # self.matrix_II = lil_matrix((self.valItemsNum,self.valItemsNum))

    def makeItemMat(self):
        self.matrix_II = lil_matrix((self.valItemsNum, self.valItemsNum)).tocsr()

    """ ---------------------------------- Edge management ---------------------------------- """

    """ Add an edge at UI/IU matrices """
    def AddEdgeUID2IID(self, uid, iid, rating):
        self.list_UI_uid.append(uid)
        self.list_UI_iid.append(iid)
        self.list_UI_rat.append(rating)

    """ Add an edge at UT/TU matrices """
    def AddEdgeUID2TID(self, uid, tid):
        self.list_UT_uid.append(uid)
        self.list_UT_tid.append(tid)
        self.list_UT_rat.append(1)

    """ Add an edge at IT/TI matrices """
    def AddEdgeIID2TID(self, iid, tid):
        self.list_IT_iid.append(iid)
        self.list_IT_tid.append(tid)
        self.list_IT_rat.append(1)

    """ ---------------------------------- Make All Matrices by Lists ---------------------------------- """

    """依據輸入結果完成所有矩陣。
            例如：根據AddEdgeUID2IID的陣列結果完成matrix_UI/IU
            shape是matUU大小*matII大小，UI / IU 則大小相反 """
    def makeAllMat(self):
        """ 做出基本UU / II / TT 矩陣 """
        self.makeUserMat()
        self.makeItemMat()
        self.makeTagMat()

        """ 合併矩陣並轉為csr格式儲存 """
        """ UI / IU mat """
        self.matrix_UI = coo_matrix((self.list_UI_rat, (self.list_UI_uid, self.list_UI_iid)), shape=(self.matrix_UU.shape[0], self.matrix_II.shape[0])).tocsr()
        self.matrix_IU = coo_matrix((self.list_UI_rat, (self.list_UI_iid, self.list_UI_uid)), shape=(self.matrix_II.shape[0], self.matrix_UU.shape[0])).tocsr()
        """ UT / TU mat """
        self.matrix_UT = coo_matrix((self.list_UT_rat, (self.list_UT_uid, self.list_UT_tid)), shape=(self.matrix_UU.shape[0], self.matrix_TT.shape[0])).tocsr()
        self.matrix_TU = coo_matrix((self.list_UT_rat, (self.list_UT_tid, self.list_UT_uid)), shape=(self.matrix_TT.shape[0], self.matrix_UU.shape[0])).tocsr()
        """ IT / TI mat """
        self.matrix_IT = coo_matrix((self.list_IT_rat, (self.list_IT_iid, self.list_IT_tid)), shape=(self.matrix_II.shape[0], self.matrix_TT.shape[0])).tocsr()
        self.matrix_TI = coo_matrix((self.list_IT_rat, (self.list_IT_tid, self.list_IT_iid)), shape=(self.matrix_TT.shape[0], self.matrix_II.shape[0])).tocsr()

        """ 測試用，可刪除 """
        # print('-------------------------- Make -- All -- Mat -----------------------')
        # print('UI',self.matrix_UI.todense())
        # print('IU',self.matrix_IU.todense())
        # print('UT',self.matrix_UT.todense())
        # print('TU',self.matrix_TU.todense())
        # print('IT',self.matrix_IT.todense())
        # print('TI',self.matrix_TI.todense())


    """ ---------------------------------- Vertex management ---------------------------------- """

    """ --- Column-normalization --- """
    def ColumnNomalization(self, matrix):
        valVeretxIDMax = matrix.shape[0]
        vecOne = np.ones((1, valVeretxIDMax))
        """ 取出每行的維度與index"""
        vecColDegree = vecOne * matrix
        vecVal = vecColDegree[vecColDegree.nonzero()].tolist()
        (vecPosX, vecPosY) = vecColDegree.nonzero()
        """ 將所有的值改為 1 / 維度  (PageRank )"""
        vecVal = [1./float(x) for x in vecVal]
        # print('[ColumnNomalization]vecVal',vecVal)
        vecPosY = vecPosY.tolist()
        """ 做出對角矩陣 值為 1 / 每行的維度 """
        matrixD = coo_matrix((vecVal, (vecPosY, vecPosY)), shape=(valVeretxIDMax, valVeretxIDMax))
        """ 將原來矩陣所有的值做 normalization ( 值 *  1 / 每行的維度 = 對於行的比重?) """
        # print('[ColumnNomalization] matrixD',matrixD.todense())
        matrixRet = matrix * matrixD
        # print('[ColumnNomalization] matrixRet',matrixRet.todense())
        return matrixRet

    """ --- Row-normalization --- """
    def RowNomalization(self, matrix):
        valVeretxIDMax = matrix.shape[1]
        vecOne = np.ones((valVeretxIDMax, 1))
        vecRowDegree = matrix * vecOne
        vecVal = vecRowDegree[vecRowDegree.nonzero()].tolist()
        (vecPosX, vecPosY) = vecRowDegree.nonzero()
        vecVal = [1./x for x in vecVal]
        vecPosX = vecPosX.tolist()

        matrixD = coo_matrix((vecVal, (vecPosX, vecPosX)), shape=(valVeretxIDMax, valVeretxIDMax))

        matrixRet = matrixD * matrix
        return matrixRet

    """ --- Symmetric-normalization --- (未建立)"""
    def SymmetricNomalization(self, matrix):
        valVeretxIDMax = matrix.shape[1]
        vecOne = np.ones((valVeretxIDMax, 1))
        vecRowDegree = matrix * vecOne
        vecVal = vecRowDegree[vecRowDegree.nonzero()].tolist()
        (vecPosX, vecPosY) = vecRowDegree.nonzero()
        vecVal = [1. / (x ** 0.5) for x in vecVal]
        vecPosX = vecPosX.tolist()

        matrixD = coo_matrix((vecVal, (vecPosX, vecPosX)), shape=(valVeretxIDMax, valVeretxIDMax))

        matrixRet = matrixD * matrix * matrixD
        return matrixRet

    """ sigmoid function """
    def sigmoid(self, s):
        theta = 1.0 / (1.0 + np.exp(-s))
        return theta

    """ sigmf Matrix將矩陣數值轉為0.5~1之間的值，若無相關則為0 """
    def sigmfMatrix(self, matX):
        vecVal = matX[matX.nonzero()]
        """ 非零的值超過1個才做，否則就使用原矩陣即可，使用.shape[1]避免矩陣格式不同而讀不到nnz的問題 """
        if vecVal.shape[1] > 0:
            vecVal = vecVal.tolist()[0]    # 轉為list後面才可計算sigmoid
            (vecPosX, vecPosY) = matX.nonzero()
            (m, n) = matX.shape
            vecAverage = np.average(vecVal)
            valSigmoidScale = np.std(np.array(vecVal))
            """ avoid x/0 """
            if valSigmoidScale == 0:
                valSigmoidScale = 1
            else:
                """ remove outlier and calculate new std """
                stdVecVal = [x for x in vecVal if x < (valSigmoidScale + vecAverage)]
                valSigmoidScale = np.std(np.array(stdVecVal))
                if valSigmoidScale == 0:
                    valSigmoidScale = 1
            """ sigmoid """
            vecVal = self.sigmoid(np.array(vecVal) / valSigmoidScale)
            matX = coo_matrix((vecVal, (vecPosX, vecPosY)), shape= (m, n))
        return matX

    """
          --- Update the transition matrix by normalizing the adjacent matrix. ---
          --- valOPcode has three value : {1 : ColumnNomalization; 2 : SymmetricNomalization; 3 : RowNomalization} ---
          --- Nete that processing this function may take much time, so update periodically ---"""
    def UpdateTransMatrix(self):
        self.matrix_UU = self.sigmfMatrix(self.matrix_UU)
        self.matrix_II = self.sigmfMatrix(self.matrix_II)
        self.matrix_TT = self.sigmfMatrix(self.matrix_TT)
        self.matrix_UI = self.sigmfMatrix(self.matrix_UI)
        self.matrix_IU = self.sigmfMatrix(self.matrix_IU)
        self.matrix_UT = self.sigmfMatrix(self.matrix_UT)
        self.matrix_TU = self.sigmfMatrix(self.matrix_TU)
        self.matrix_IT = self.sigmfMatrix(self.matrix_IT)
        self.matrix_TI = self.sigmfMatrix(self.matrix_TI)

        """ 將矩陣乘以各自的權重後合併
           [ matrix_UU ] [ matrix_UI ]  [ matrix_UT ]
           [ matrix_IU ] [ matrix_II ]  [ matrix_IT ]
           [ matrix_TU ] [ matrix_TI ]  [ matrix_TT ]
           """
        temp_rU = hstack([self.matrix_UU * self.valUU, self.matrix_UI * self.valUI, self.matrix_UT * self.valUT])
        temp_rI = hstack([self.matrix_IU * self.valIU, self.matrix_II * self.valII, self.matrix_IT * self.valIT])
        temp_rT = hstack([self.matrix_TU * (1-self.valUU-self.valIU), self.matrix_TI * (1-self.valUI-self.valII), self.matrix_TT * (1-self.valUT-self.valIT)])
        self.matrixTrans = vstack([temp_rU, temp_rI, temp_rT])
        self.log.debug('[UpdateTransMatrix] matrixTrans')
        self.log.debug(self.matrixTrans.todense())

        """ 使用真矩陣 """
        self.valUserNum_MatrixTrans = self.valUsersNum
        self.valItemNum_MatrixTrans = self.valItemsNum
        self.valTagNum_MatrixTrans = self.valTagsNum
        """ 使用真矩陣 """

        valVertexIDMax = self.matrixTrans.shape
        matrixI = eye(valVertexIDMax[0], valVertexIDMax[1])

        """ Column-Nomalization (for PageRank) """
        self.objInfRank.matrixColNormTrans = self.valDiffusionRate * self.ColumnNomalization(self.matrixTrans) + (1 - self.valDiffusionRate) * matrixI
        """ Symmetric Nomalization (for InfRank) """
        self.objInfRank.matrixSymNormTrans = self.valDiffusionRate * self.SymmetricNomalization(self.matrixTrans) + (1 - self.valDiffusionRate) * matrixI

        # clear matrixI;
        # clear obj.matrixTrans;
        """ 以下暫時不用 """
        # matrixRowNorm = self.RowNomalization(self.matrixTrans)
        # # self.objInfRank = self.objInfRank.InfluenceProcess(matrixRowNorm)
        # self.objInfRank.InfluenceProcess(matrixRowNorm)

    """ 處理Query 資料以及推薦排序 """
    def ExecRankingProcess(self, vecQueryUID, vecQueryIID, vecQueryTID, resultType, normType, k):
        valQueryUserSize = len(vecQueryUID)
        valQueryItemSize = len(vecQueryIID)
        valQueryTagSize = len(vecQueryTID)

        self.log.debug('[ExecRankingProcess] UserNum {0}'.format(self.valUserNum_MatrixTrans))
        self.log.debug('[ExecRankingProcess] ItemNum {0}'.format(self.valItemNum_MatrixTrans))
        self.log.debug('[ExecRankingProcess] TagNum {0}'.format(self.valTagNum_MatrixTrans))
        valTransMatrixSize = self.valUserNum_MatrixTrans + self.valItemNum_MatrixTrans + self.valTagNum_MatrixTrans

        """  Start to Construct the Query vector of initial state : vecInitialState  """
        vecInitialIndex = np.zeros((valQueryUserSize + valQueryItemSize + valQueryTagSize))
        vecInitialValue = np.zeros((valQueryUserSize + valQueryItemSize + valQueryTagSize))

        for i in range(valQueryUserSize):
            vecInitialIndex[i] = vecQueryUID[i]
            vecInitialValue[i] = self.valQueryUser * 1 / valQueryUserSize

        for i in range(valQueryItemSize):
            vecInitialIndex[valQueryUserSize + i] = self.valUserNum_MatrixTrans + vecQueryIID[i]
            vecInitialValue[valQueryUserSize + i] = self.valQueryItem * 1 / valQueryItemSize

        for i in range(valQueryTagSize):
            vecInitialIndex[valQueryUserSize + valQueryItemSize + i] = self.valUserNum_MatrixTrans + self.valItemNum_MatrixTrans + vecQueryTID[i]
            vecInitialValue[valQueryUserSize + valQueryItemSize + i] = self.valQueryTag * 1 / valQueryTagSize

        vecInitialState = coo_matrix((vecInitialValue.tolist(), (vecInitialIndex.tolist(), np.zeros(len(vecInitialValue)).tolist())), shape=(valTransMatrixSize, 1)).tocsr()
        self.vecInitialState = vecInitialState / vecInitialState.sum()  # normalization
        """  Finished Construct the vector of initial state : vecInitialState  """

        val_prStrat = time.time()
        self.vecInfRankVertexScore = self.objInfRank.DiffusionProcess(self.vecInitialState, self.vecInitialState, normType)
        val_prStop = time.time()
        self.log.info('[ExecRankingProcess] PR process time= {0}'.format(val_prStop-val_prStrat))

        """ Starting Output the result from 'vecInfRankVertexScore': rankResultList  """
        if resultType == 1:
            """ user """
            """ 只取使用者這一段來處理 """
            vecResultScore = self.vecInfRankVertexScore.toarray()[0:self.valUserNum_MatrixTrans]
            print(vecResultScore)
            """ 取得排序後的index """
            self.vecSortedIndex_InfRank = sorted(range(len(vecResultScore)), reverse=True , key = lambda k:vecResultScore[k])
            print(self.vecSortedIndex_InfRank)
            if valQueryUserSize > 0:
                rankResultList = self.vecSortedIndex_InfRank[0+valQueryUserSize : min(k+valQueryUserSize, self.valUserNum_MatrixTrans)]
            else:
                rankResultList = self.vecSortedIndex_InfRank[0 : min(k, self.valUserNum_MatrixTrans)]
        elif resultType == 2:
            """ item """
            vecResultScore = self.vecInfRankVertexScore.toarray()[self.valUserNum_MatrixTrans : self.valUserNum_MatrixTrans+self.valItemNum_MatrixTrans]
            """ 取得排序後的index """
            self.vecSortedIndex_InfRank = sorted(range(len(vecResultScore)), reverse=True , key = lambda k:vecResultScore[k])

            if valQueryItemSize > 0:
                rankResultList = self.vecSortedIndex_InfRank[0+valQueryItemSize : min(k+valQueryItemSize, self.valItemNum_MatrixTrans)]
            else:
                rankResultList = self.vecSortedIndex_InfRank[0 : min(k, self.valItemNum_MatrixTrans)]
        elif resultType == 3:
            """ tag """
            vecResultScore = self.vecInfRankVertexScore.toarray()[self.valUserNum_MatrixTrans+self.valItemNum_MatrixTrans : valTransMatrixSize]
            """ 取得排序後的index """
            self.vecSortedIndex_InfRank = sorted(range(len(vecResultScore)), reverse=True , key = lambda k:vecResultScore[k])

            if valQueryTagSize > 0:
                rankResultList = self.vecSortedIndex_InfRank[0+valQueryTagSize : min(k+valQueryTagSize, self.valTagNum_MatrixTrans)]
            else:
                rankResultList = self.vecSortedIndex_InfRank[0 : min(k, self.valTagNum_MatrixTrans)]
        """ Finish Output the result from 'vecInfRankVertexScore': rankResultList  """
        # print('看分數', vecResultScore[rankResultList]#看分數)
        # print(rankResultList)
        return rankResultList


    def UpdateParam(self, valIterationNum, valDampFactor, valDiffusionRate, valInfDiffRate, valInfDampFactor):
        self.valIterationNum = valIterationNum
        self.valDampFactor = valDampFactor
        self.valDiffusionRate = valDiffusionRate
        self.valInfDiffRate = valInfDiffRate
        self.valInfDampFactor = valInfDampFactor
        self.objInfRank = self.objInfRank.ModifyParam(valInfDampFactor, valDampFactor, valDiffusionRate, valIterationNum)
