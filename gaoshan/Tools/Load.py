# coding=utf-8
'''
Created on 2019年1月15日

@author: master
'''
import  os
import numpy as np 
from sklearn.model_selection import train_test_split
from __builtin__ import str
from keras.preprocessing.sequence import pad_sequences

PATH = '../data'
EMBEDDING_DIM = 100

#统计句子的最长字数
def getWordsCount():
    list = []
    getFileName(PATH, list)
    fileListByFilter = [enum for enum in list if '.txt' in enum]
    max = 0
    for txt in fileListByFilter:
        with open(txt) as f:
            str = f.read().decode('utf-8').replace('\n','')
        sentences = cut_sentence(str)
        for sentence in sentences:
            num = len(sentence)
            if max < num:
                max = num
    return max
 
def getFileName(path,list=[]):  
    for file in os.listdir(path):
        filepath = os.path.join(path,file)
        if os.path.isdir(filepath):
            getFileName(filepath,list)
        else:
            list.append(filepath);

def cut_sentence(words): 
    lines = []
    temp = []
    cutlist =  ',.!?:;~。!?:;~'.decode('utf8')
    for word in words:
        if word in cutlist:
            temp.append(word);
            lines.append("".join(temp))
            temp = []
        else:
            temp.append(word)
    return lines


def cutSentence(doc):
    lines = []
    temp = []
    cutlist =  ',.!?:;~。!?:;~'.decode('utf8')
    for item in doc:
        word = item[0]
        
        
        if word in cutlist:
            temp.append(item)
            lines.append(temp)
            temp = []
        else:
            temp.append(item)
    return lines 

def getData():
    filelist = []
    wordlist = []
    getFileName(PATH, filelist)
    filelist = [enum for enum in filelist if '.txt' in enum]
    lines = []
    for filename in filelist:  
    #按字组织数据
        
        with open(filename) as f:
            str = f.read().decode('utf-8')
            wordlist = list(str)
            wordtoid = zip(wordlist,range(len(wordlist)))
        biaozhu = {}
        annname = filename.replace(".txt",".ann")
        with open(annname) as f:
            for line in f.readlines():
                try:
                    lineArray = line.split("\t")
                    lineArray2 = lineArray[1].split(" ")
                    type = lineArray2[0]
                    start =  int(lineArray2[1])
                    end = int(lineArray2[2])
                    for i in range(start,end):
                        biaozhu[i] = type
                except Exception :
                    pass
        temp = []        
        for item in wordtoid:
            
            word = item[0]
            offset = item[1]
            if offset in biaozhu.keys():
                temp.append((word,biaozhu[offset]))
            else:
                temp.append((word,'o'))
        lines.extend(cutSentence(temp))
    return lines

def getWordDic():
    list = []
    wordlist = []
    getFileName(PATH, list)
    list = [enum for enum in list if '.txt' in enum]
    for filename in list:
        with open(filename) as f:
            str = f.read().decode('utf-8')
            for word in str:
               
                wordlist.append(word)
    wordlist = set(wordlist)
    
    wordtoid = dict(zip(wordlist,range(len(wordlist))))
    return wordtoid    

def getType():
    list = []
    typelist = []
    getFileName(PATH, list)
    list = [enum for enum in list if '.ann' in enum] 
    for filename in list:
        with open(filename) as f:
            for line in f.readlines():
                lineArray = line.split("\t")
                lineArray2 = lineArray[1].split(" ")
                type = lineArray2[0]
                typelist.append(type)
    typelist.append('o')
    typelist = set(typelist)
    
    typetoid = dict(zip(typelist,range(len(typelist))))
    
    return typetoid
def prepareData():
    wordtoid = getWordDic()
    typetoid = getType()
    data = getData()
    x_train,x_test,y_train,y_test = getTrainAndTestData(wordtoid, typetoid, data)
    embedding = getRandomEmbedding(wordtoid)
    return x_train,x_test,y_train,y_test,embedding,wordtoid,typetoid
    
    
def getTrainAndTestData(wordtoid,typetoid,data):
    dx = []
    dy = []
    for sentence in data :
        temp_x = []
        temp_y = []
        for item in sentence:
            word = wordtoid[item[0]]
            type = typetoid[item[1]]
            temp_x.append(word)
            temp_y.append(type)
        dx.append(temp_x)
        dy.append(temp_y)
    maxlen = np.max(map(len, dx))
    x_train, x_test, y_train, y_test = train_test_split(dx, dy, test_size=0.3, random_state=42)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    y_train = pad_sequences(y_train, maxlen=maxlen,value=-1)
    y_test = pad_sequences(y_test, maxlen=maxlen,value=-1)
    y_train = np.expand_dims(y_train, 2)
    y_test = np.expand_dims(y_test, 2)
    return x_train,x_test,y_train,y_test
              
def getRandomEmbedding(wordtoid): 
    max_token = len(wordtoid)
    embedding_matrix = np.zeros((max_token, EMBEDDING_DIM))
    return embedding_matrix       

if __name__ == '__main__':
    wordtoid = getWordDic()
    typetoid = getType()
    data = getData()  
    getTrainAndTestData(wordtoid, typetoid, data)    
 
             
             
         
         
              
             
             
     
     
          
         
    
