# -*- coding: utf-8 -*-
"""
k-近邻算法进行车辆评测

Created on Wed Aug  9 14:25:59 2017

@author: cfd
"""
from numpy import *
import operator 
#数据处理
#数字化处理
def KeyToNum(line):
    #将一行的字符串变成对应的数字
    numline = [] 
    switcher = {
            'vhigh'  : 4 ,
            'high'   : 3 ,
            'med'    : 2 ,
            'low'    : 1 ,
            '5more'  : 5 ,
            'more'   : 6 ,
            'small'  : 1 ,
            'big'    : 3 ,
            'unacc'  : 1 ,
            'acc'    : 2 ,
            'good'   : 3 ,
            'vgood'  : 4 ,          
    }
    for key in line:
        numline.append(switcher.get(key, key))
    return numline
#读数据集并保存到矩阵中
def filecarmatrix(filepath):
    fr = open(filepath)
    arraylines = fr.readlines()
    numberlines = len(arraylines)
    returnMat = zeros((numberlines, 6)) #生成零矩阵
    classLabelVector = []
    index = 0;        
    for line in arraylines:
        line = line.strip()#移除字符串头尾指定的字符（默认为空格）
        listfromline = line.split(',')
        numline = KeyToNum(listfromline) #将数据进行数字化处理
        returnMat[index, :] = numline[0:6]#将特征数据量传入矩阵
        classLabelVector.append(int(numline[-1]))
        index += 1
    return returnMat, classLabelVector
        
    
#归一化处理数据
'''
归一化数值
newValue = (oldValue - min) / (max - min)
'''
def autoNorm(DataSet):
    minVals = DataSet.min(0 )#将每列中的最小值放在变量minVals中
    maxVals = DataSet.max(0) #将每列中的最小值放在变量minVals中
    ranges = maxVals - minVals #将每列中的最小值放在变量minVals中
    normDataSet = zeros(shape(DataSet)) #生成一个与dataSet相同的零矩阵
    m = DataSet.shape[0] #求出dataSet列长度
    normDataSet = DataSet - tile(minVals, (m, 1)) #求出oldValue - min
    normDataSet = normDataSet / tile(ranges, (m,1)) #求出归一化数值
    return normDataSet, ranges, minVals

#分类器制作
def classifyCar(CarData, DataSet, Labels, k):
    DataSetSize = DataSet.shape[0] #获取矩阵第一纬度的长度
    DiffMat = tile(CarData, (DataSetSize, 1)) - DataSet
    sqDiffMat = DiffMat**2
    sqDistances = sqDiffMat**0.5
    distances = sqDistances.sum(axis=1) #矩阵行相加。生成新矩阵
    sortedDistIndicies = distances.argsort() #返回矩阵中的数组从小到大的下标值，返回新矩阵
    classCount = {}  #初始化新字典
    for i in range(k):
        voterLabel = Labels[sortedDistIndicies[i]]
        classCount[voterLabel] = classCount.get(voterLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True) #排序
    return sortedClassCount[0][0]

#编写测试代码
def CarEvaClassTest():
    basePer = 0.1 #测试基数，选取文本中10%的数据进行测试
    CarDataMat, CarLabels = filecarmatrix(r'D:\Learning\DataSet\car.txt')
    normMat, ranges, minVals = autoNorm(CarDataMat) #进行数据归一化
    m = normMat.shape[0]  #读取数据的列长度
    numTestVecs = int(m * basePer) #确定测试的数量
    errorCount = 0.0  #记录错误数量的变量
    for i in range(numTestVecs):  #进行循环测试
        result = classifyCar(normMat[i, :], normMat[numTestVecs:m, :], \
                             CarLabels[numTestVecs:m], 6) #通过分类器进行判断
        print('the classifer came back with: %d, the real answer is %d' \
              % (result, CarLabels[i]))
        if(result != CarLabels[i]): #比较判断数据和实际数据，并且打印
            errorCount += 1 #错误计数
    print('the total error rate is %f' % (errorCount/float(numTestVecs)))#打印错误率

#交互方法
def CarEvaluation():
    resultList = ['unacceptable', 'accept', 'good', 'very good']
    buying = input('How much is this car? Options: vhigh, high, med, low\n')
    maint = input('How much is the maintenance of the car? Options: vhigh, high, med, low\n')
    doors = input('How many doors does this car have? Options: 2, 3, 4, 5more\n')
    person = input('How many people can this car hold? Options:  2, 4, more\n')
    lug_boot = input('How big is the trunk of this car? Options: small, med, big\n')
    safety = input('How safe is the car? Options: low, med, high\n')
    characteristic = [buying, maint, doors, person, lug_boot, safety]
    CarDataMat, CarLabels = filecarmatrix(r'D:\Learning\DataSet\car.txt')
    normMat, ranges, minVals = autoNorm(CarDataMat) #进行数据归一化
    inArr = array(list(map(int, KeyToNum(characteristic)))) #通过map函数将，KeytoNum生成的列表内容全部转换为数字
    print(KeyToNum(characteristic))
    print(inArr)
    Result= classifyCar((inArr - minVals) / ranges, normMat, CarLabels, 6) 
    print('You will probably like this car:', resultList[Result - 1])
#print(KeyToNum(['vhigh','vhigh',2,2,'small','low', 'unacc']))

#returnMat, classLabelVector = filecarmatrix(r'D:\Learning\DataSet\car.txt')
#print(returnMat)
#CarEvaClassTest()
CarEvaluation()