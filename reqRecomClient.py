# -*- coding:utf-8 -*-
import zmq
import json
from concurrent.futures import ThreadPoolExecutor
from loadDataFromLocal import loadDataFromLocal
import recomclient

pool = ThreadPoolExecutor(11)

def reqRecom(id = 0):
    host = '192.168.4.216'
    port = 8888
    url_socket = "tcp://%s:%s" %(host, port)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect( url_socket )

    poll = zmq.Poller()
    poll.register(socket, zmq.POLLIN)

    task = {
        'ID' : id,
        'REQ': 'R',
        'DATA': [[2], [], [4], 2]
    }
    task_json_encode = json.dumps(task)
    socket.send_json(task_json_encode)
    socks = dict(poll.poll())
    if socks.get(socket) == zmq.POLLIN:
        ret = socket.recv_json()
        import time
        print( 'now time= {0}'.format(time.localtime(time.time())))
        print(ret)
        print('-'*25)

def reqUpd(data, req, id = 0):
    host = '192.168.4.216'
    port = 8888
    url_socket = "tcp://%s:%s" %(host, port)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect( url_socket )

    poll = zmq.Poller()
    poll.register(socket, zmq.POLLIN)

    task = {
        'ID' : id,
        'REQ': req,
        'DATA': data
            # [[1, 4, 3, 2],
            #     [2, 5, 3, 2],
            #     [2, 5, 3, 7],
            #     [3, 6, 3, 6]]
    }
    task_json_encode = json.dumps(task)
    socket.send_json(task_json_encode)
    print(task_json_encode)
    socks = dict(poll.poll())
    if socks.get(socket) == zmq.POLLIN:
        ret = socket.recv_json()
        import time
        print( 'now time = {0}'.format(time.localtime(time.time())))
        print(ret)
        print('-'* 80)

def test():
    print('-'* 80)


    data = loadDataFromLocal(r'train.dat')
    obj_upd_job = pool.submit(reqUpd, data, 'I', 'I01')

    # data = ['tesssssssssst']
    # obj_upd_job = pool.submit(reqUpd, data, 'S', 'S01')


    for i in range(1,2):
        obj_recom_job = pool.submit(reqRecom, 'R'+str(i))

    # data = [[1, 4, 3, 2],
    #         [2, 5, 3, 2],
    #         [2, 5, 3, 7],
    #         [3, 6, 3, 6]]
    # obj_upd_job = pool.submit(reqUpd, data, 'U', 'U01')

    # print(recomclient.recomtask([[2], [], [4], 2]))


if __name__ == '__main__':
    test()
