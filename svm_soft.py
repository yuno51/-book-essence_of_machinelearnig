import numpy as np
from operator import itemgetter

class SVC:
    def __init__(self, C=1):
        self.C = C

    def fit(self, X, y,selections = None):
        a = np.zeros(X.shape[0])
        ay = 0
        ayx = np.zeros(X.shape[1])
        yx = y.reshape(-1,1) * X
        indices = np.arange(X.shape[0])
        while True:
            ydf = y * (1- np.dot(yx, ayx.T))
            iydf = np.c_[indices, ydf]
            i = int(min(iydf[((a>0)&(y>0))|((a<self.C)&(y<0))], key = itemgetter(1))[0])
            j = int(max(iydf[((a>0)&(y<0))|((a<self.C)&(y>0))], key = itemgetter(1))[0])
            if ydf[i] >= ydf[j]:
                break
            ay2 = ay - y[i]*a[i] -y[j]*a[j] #k != i,j
            ayx2 = ayx - y[i]*a[i]*X[i,:] -y[j]*a[j]*X[j,:] #k != i,j
            ai = (1 -y[i]*y[j] +y[i] * np.dot(X[i,:]-X[j,:], X[j,:]*ay2-ayx2)) / ((X[i,:]-X[j,:])**2).sum()
            if ai <= 0:
                ai = 0
            elif ai > self.C:
                ai = self.C
            aj = y[j] * (-ai*y[i] - ay2) 
            if aj <= 0:
                aj = 0
                ai = y[i] * (-aj*y[j] - ay2)
            elif aj > self.C:
                aj = self.C
                ai = y[i] * (-aj*y[j] - ay2)
            ay += y[i]*(ai-a[i]) + y[j]*(aj-a[j])
            ayx += y[i]*(ai-a[i])*X[i,:] + y[j]*(aj-a[j])*X[j,:]
            if a[i] == ai:
                break
            a[i] = ai
            a[j] = aj
            self.a_ = a
            ind = a != 0. #True,Falseの行列
            self.w_ = ((a[ind]*y[ind]).reshape(-1,1) * X[ind,:]).sum(axis = 0)
            self.w0_ = (y[ind]- np.dot(X[ind,:],self.w_)).sum() / ind.sum()
    
    def predict(self, X):
        return np.sign(self.w0_ + np.dot(X, self.w_))