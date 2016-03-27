#-*- coding:utf-8 -*-
def auc():
    data = []
    count = [0,0]
    with open('aucdata') as f:
        for line in f.readlines():
            label1, lable2, score = line.strip().split('\t')
            label = 0 if label1 == '1' else 1
            score = float(score)
            count[label] += 1
            data.append([score, label]) 
     
    data = sorted(data, key = lambda x:x[0], reverse = True)
    
    auc = 0 
    tp = 0
    fp = 0
    tppre = 0
    fppre = 0
    for sort_data in data:
        if sort_data[1] == 0:
            fp = fp + 1
        else:
            tp = tp + 1
        auc = auc + (fp - fppre) * (tp + tppre)
        fppre = fp
        tppre = tp
    print 1.0 * auc /(fp * tp * 2)
def auc1():
    data = []
    count = [0,0]
    with open('aucdata') as f:
        for line in f.readlines():
            label1, label2, score = line.strip().split('\t')
            label = 0 if label1 == '1' else 1
            score = float(score)
            count[label] += 1
            data.append([score, label])
         
        data = sorted(data,key=lambda x:x[0],reverse=True)
        n = len(data)
        k = 0
        for i in xrange(n):
            if data[i][1] == 1:
                k += n-i
        print (k-count[1]*(count[1]+1)/2.0)/(count[0]*count[1])
                 
if __name__ == '__main__':
    auc()
