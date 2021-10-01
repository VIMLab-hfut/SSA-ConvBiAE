import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def return_data(data, windowLen, add):  #data 是原数据，windowLen 是滑窗大小，add 是用多少个奇异值来还原原始数据
    seriesLen = len(data)
    #print(seriesLen,"seriesLen")
    K = seriesLen - windowLen + 1
    #print(K,"K")
    X = np.zeros((windowLen, K))
    #print(X,"X")
    for i in range(K):
        X[:, i] = data[i:i+windowLen]
    #print(X,"X2")

    U, sigma, VT = np.linalg.svd(X, full_matrices=False)
    for i in range(VT.shape[0]):
        VT[i, :] *= sigma[i]
    A = VT
    #print(A,"A")
    rec = np.zeros((windowLen, seriesLen))
    for i in range(windowLen):
        for j in range(windowLen - 1):
            for m in range(j + 1):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (j + 1)
        for j in range(windowLen - 1, seriesLen - windowLen + 1):
            for m in range(windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= windowLen
        for j in range(seriesLen - windowLen + 1, seriesLen):
            for m in range(j - seriesLen + windowLen, windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (seriesLen - j)
    print(rec.shape,"rec")
    #print(rec[1,:],"rec[1,:]")
    #r1 = np.sum(rec[:add,:], axis=0)  #
    #r2 = np.sum(rec[add:windowLen, :], axis=0)

    r1 = rec[0,:]
    r2 = rec[1,:]
    r3 = rec[2,:]
    r4 = rec[3,:]
    r5 = rec[4,:]
    r6 = rec[5,:]
    #r2 = np.sum(rec[1:windowLen, :], axis=0)


    return r1,r2 ,r3,r4,r5,r6


