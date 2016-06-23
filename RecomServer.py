# -*- coding:utf-8 -*-
__author__ = 'c11tch'
from GraphMng import GraphMng
from loadDataFromLocal import loadDataFromLocal
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import copy
import logging
import logging.config


class localThTest:

    dictUser2UID = {}
    dictUID2User = {}
    dictTag2TID = {}
    dictTID2Tag = {}
    dictItem2IID = {}
    dictIID2Item = {}
    valUsersNum = 0
    valTagsNum = 0
    valItemsNum = 0

    """ For manage copy graph using by thread
        = graph obj =    -graph ogl-      - graph 1 -      - graph 2 -
        =   state   =    -  None   -      -  ready  -      - updating -
        = usr queue =    -  None   -      -queue has val-  - queue.empty()=TRUE-
    """
    graphs = []
    num_of_graphs = 2
    graphState = []
    graphsOn = False

    """ 製作GM物件以管理Graph """
    GraphMng = GraphMng()
    """ 讀檔並存於localTest.list_locData中 """
    # list_locData = loadDataFromLocal(r'train.dat')
    list_locData = loadDataFromLocal(r'new_train.dat')

    """ start a process exec pool """
    pool = ThreadPoolExecutor(11)

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

    """
    ---------------------------------- To Generate Graphs ----------------------------------
    將讀檔後的資料存入Graphs
    The input format is [[user , item , rating , tag],
                         [user , item , rating , tag],
                         [user , item , rating , tag]]
    """
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
        """ --------------------------- End make_Graph_from_Data() ------------------------- """

    """ Combine all query information to a list """
    def makaRequestList(self, vecQueryUID, vecQueryIID, vecQueryTID, resultType):
        aRequest = []
        aRequest.append(vecQueryUID)
        aRequest.append(vecQueryIID)
        aRequest.append(vecQueryTID)
        aRequest.append(resultType)
        return aRequest

    """ --------------------------- Start Process Service ------------------------- """
    def startProcess(self):
        """ init service """
        """ copy graph obj to graphs list """
        for i in range(self.num_of_graphs):
            self.graphs.append(copy.copy(self.GraphMng))
            self.graphState.append('ready')

        """ Start a Server Socket """
        import zmq
        import json
        import threading

        """ Bind host * and port 8888 """
        host = '192.168.4.216'
        port = 8888
        url_router = "tcp://%s:%s" %(host, port)
        url_worker = 'inproc://ping-workers'
        worker_num = 10

        """
        Router and dealer will auto allocate Kpy's requests to Tch's workers by following scheme.
             -client req-     -client req-     -client req-
                  |                |                |
                  |________________|________________|
                                   |
                               - router -
                               -  code  -
                               - dealer -
                  _________________|_________________
                  |                |                |
                  |                |                |
              - worker -       - worker -       - worker -
        """
        context = zmq.Context()

        router = context.socket(zmq.ROUTER)
        router.bind(url_router)

        workers = context.socket(zmq.DEALER)
        workers.bind(url_worker)

        """ Define worker's job """
        def worker (name, url_worker, context):
            self.log.warning('worker {0} start'.format(name))
            """ connect to router """
            worker = context.socket(zmq.REP)
            worker.connect(url_worker)
            while True :
                try :
                    """ Receive a JSON encode dict contains { job ID , REQ , DATA } """
                    request_d = worker.recv_json()
                    request_dict = json.loads(request_d)
                    id, req_type, data =(request_dict['ID'],request_dict['REQ'],request_dict['DATA'])
                    if len(data) <= 4:
                        self.log.info('[worker]worker {0} recv job ID:{1} REQ:{2} DATA:{3}'.format(name, id, req_type, data))
                    else:
                        self.log.info('[worker]worker {0} recv job ID:{1} REQ:{2} DATA: Data is too long to show'.format(name, id, req_type))
                    """ init reply task dict """
                    reply_task = {
                        'ID' : id,
                        'REQ': req_type,
                    }
                    self.log.info('-' * 80)

                    """ Graphs has been initialized """
                    if self.graphsOn:
                        if req_type == 'R':
                            """ job REQ R is Recommendation """
                            if len(data) == 4 and isinstance(data[0],list) and isinstance(data[1],list) and isinstance(data[2],list) and isinstance(data[3],int) :
                                result = self.recomth(data)
                                reply = self.analyzeResult(result, 2)
                            else:
                                reply = 'Data format error. [[user], [item], [tag] ,rlt type] ' \
                                        'yours: {0}'.format(data)
                        elif req_type == 'U':
                            """ job REQ U is Update graphs """
                            bef_u_num, bef_i_num, bef_t_num = self.GraphMng.valUsersNum, self.GraphMng.valItemsNum, self.GraphMng.valTagsNum
                            self.getUpdateMatRequest(data)
                            added_u = self.GraphMng.valUsersNum - bef_u_num
                            added_i = self.GraphMng.valItemsNum - bef_i_num
                            added_t = self.GraphMng.valTagsNum - bef_t_num
                            reply = 'Added user : {0}  item : {1}  tag : {2}'.format(added_u, added_i, added_t)
                        elif req_type == 'I':
                            """ job REQ I is Init graphs """
                            reply = 'Graph has been initial, stop using req_type I to send task.'
                        elif req_type == 'S':
                            """ job REQ S is query graph states """
                            reply = 'Added user : {0}  item : {1}  tag : {2}'.format(self.GraphMng.valUsersNum, self.GraphMng.valItemsNum, self.GraphMng.valTagsNum)
                        else:
                            reply = 'What are u doing!!!!!!'
                    else:
                        if req_type == 'I':
                            """ get init job """
                            self.getUpdateMatRequest(data)
                            reply = 'Start Init Graph'
                            """ open graphs up """
                            self.graphsOn = True
                        else:
                            """ graph is not open up """
                            reply = 'Graphs need to be initial'
                    """ reply task message to client """
                    reply_task['RESULT'] = reply
                    task_json_encode = json.dumps(reply_task)
                    worker.send_json(task_json_encode)
                    self.log.info('[startProcess]worker {0} reply job{1} : {2}'.format( name, id, reply_task ))
                    self.log.info('-' * 80)
                except TypeError as err:
                    self.log.error('worker {0} TypeError : {1}'.format((name, err)))
                    break
                except :
                    self.log.error('worker {0} error'.format(name))
                    break
            worker.close()
        """ start workers """
        for i in range(worker_num):
            thread = threading.Thread(target=worker, args=(i, url_worker, context))
            thread.start()
        """ I dont know what is this... """
        zmq.device(zmq.QUEUE, router, workers)
        router.close()
        workers.close()
        context.term()
        """ --------------------------- End startProcess() ------------------------- """

    """
    Process update task...
    Here will get the relation of user, item and tags in a list [[1, 4, 3, 2],
                                                                 [2, 5, 3, 2],
                                                                 [2, 5, 3, 7],
                                                                 [3, 6, 3, 6]]
    """
    def getUpdateMatRequest(self, relation_data):
        self.makeGraphFromData(relation_data)

        """ Update latest graph first g2 > g1 """
        opposite_graphs = [ i for i in range(self.num_of_graphs)]
        opposite_graphs.reverse()
        for gIdx in opposite_graphs:
            if self.graphState[gIdx] == 'ready' :
                """ Change the flag to updating """
                self.graphState[gIdx] = 'updating'

                t = True
                while t:
                    """ Waiting for all threads who using this graph exec finished """
                    if self.graphs[gIdx].userQueue.empty():
                        """ updating current graph """
                        self.graphs[gIdx] = copy.copy(self.GraphMng)
                        t = False
                """ Update finished and graph reopen """
                self.graphState[gIdx] = 'ready'
    """ End get Update Mat Request() """

    def analyzeResult(self, resultsIndex, resultType):
        self.log.debug('[analyzeResult] resultType = {0}'.format(resultType))
        if resultType == 1:
            self.log.debug('[analyzeResult] recommend users : ')
            result = [self.dictUID2User[index] for index in resultsIndex]
        elif resultType == 2:
            self.log.debug('[analyzeResult] recommend items : ')
            result = [self.dictIID2Item[index] for index in resultsIndex]
        else:
            self.log.debug('[analyzeResult] recommend tags : ')
            result = [self.dictTID2Tag[index] for index in resultsIndex]
        self.log.debug('{0}'.format(result))
        return result

    def recomth(self, lst_job):
        lst_UID, lst_IID, lst_TID, resultType = (lst_job[0], lst_job[1], lst_job[2], lst_job[3])
        """ 比對代表數值 """
        vecQueryUID = np.array(lst_UID)
        vecQueryUID = [self.dictUser2UID[index] for index in vecQueryUID]
        vecQueryIID = np.array(lst_IID)
        vecQueryIID = [self.dictItem2IID[index] for index in vecQueryIID]
        vecQueryTID = np.array(lst_TID)
        vecQueryTID = [self.dictTag2TID[index] for index in vecQueryTID]
        normType = 1
        k = 10
        t = True
        while t:
            for gIdx in range(len(self.graphState)):
                """ Checking graph is ready """
                if self.graphState[gIdx] == 'ready':
                    """ using userQueue to tell graph updater some thread using this graph """
                    self.graphs[gIdx].userQueue.put(1)
                    """ Process PageRank """
                    resultsIndex = self.graphs[gIdx].ExecRankingProcess(vecQueryUID, vecQueryIID, vecQueryTID, resultType, normType, k)
                    """ For Check threads are synchronize """
                    import time
                    self.log.debug('[recomth] use graph {0} now time= {1}'.format(gIdx,time.localtime(time.time())))
                    t = False
                    self.graphs[gIdx].userQueue.get()
                    break
        """ 為了測試 thread 拖時間 """
        # MATRIX_SIZE = 1000
        # a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE)
        # b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE)
        # c_cpu = np.dot(a_cpu, b_cpu)

        return resultsIndex
    """ End recom for thread version """


    def __init__(self):
        self.log = logging.getLogger('[RS]')
        self.log.setLevel(logging.DEBUG)

        file_hdlr = logging.FileHandler('recomServer.log')
        file_hdlr.setLevel(logging.DEBUG)

        console_hdlr =logging.StreamHandler()
        console_hdlr.setLevel(logging.INFO)

        formatter = logging.Formatter('%(levelname)-8s - %(asctime)s - %(name)-12s - %(message)s')
        file_hdlr.setFormatter(formatter)
        console_hdlr.setFormatter(formatter)

        self.log.addHandler(file_hdlr)
        self.log.addHandler(console_hdlr)

        self.log.debug('startProcess')
        # self.list_locData = self.makeGraphFromData(self.list_locData) # load data from local
        self.startProcess()

if __name__ == '__main__':
    localThTest = localThTest()