__author__ = 'c11tch'

def loadDataFromLocal(path):
    """ open data and store it in a list by case [  [train data 1 (line 1 )] , [train data 2] , [train data n]  ] """
    with open(path) as localData:
        listLocData = []
        for data in localData:
            d = data.split()
            listLocData.append(d)
    localData.close()
    return listLocData

def writeDataToLocal(path, list_wanna_write):
    """ writing data from given list to given local path """
    with open(path,'w') as write_file:
        for line in list_wanna_write:
            write_file.writelines(str(line)+":"+str(list_wanna_write[line])+'\n')
    write_file.close()