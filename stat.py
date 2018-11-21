# -*- coding: UTF-8 -*-
import os

frequency1 = {}
frequency2 = {}

def prestat(filename):
    fin = open(filename, 'r')
    seclen = 0
    sec = 0
    maxlen = 0
    maxlen2 = 0
    i = 0
    for line in fin:
        utlen = 0
        if line.strip() != '':
            seclen = seclen + 1
            for word in line.split():
                if word not in frequency1:
                    frequency1[word] = 1
                else:
                    frequency1[word] = frequency1[word] + 1
                utlen = utlen + 1
            if utlen >= maxlen:
                maxlen = utlen
                index = i
        else:
            if str(seclen) not in frequency2:
                frequency2[str(seclen)] = 1
            else:
                frequency2[str(seclen)] = frequency2[str(seclen)] + 1
            if seclen >= maxlen2:
                maxlen2 = seclen
                index2 = i
            seclen = 0
            sec = sec + 1
            
        i = i + 1

    print(str(maxlen)+'\t'+str(index))
    tmpdict1 = sorted(frequency1.items(), key=lambda item:item[1], reverse=True)
    for key,value in tmpdict1:
        print(str(key)+'\t'+str(value))
    tmpdict2 = sorted(frequency2.items(), key=lambda item:int(item[1]), reverse=True)
    for key,value in tmpdict2:
        print(str(key)+'\t'+str(value))
    print(str(maxlen2)+'\t'+str(index2))
    
prestat('../train-data/train-s.txt')
