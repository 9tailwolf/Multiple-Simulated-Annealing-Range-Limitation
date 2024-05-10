from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy

class SVR_Model():
    def __init__(self,C=2,gamma=1,epsilon=0.1):
        self.svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        df = pd.read_csv("data/log.csv")
        x, y = [],[]
        for i in range(df.shape[0]):
            x.append([df.iloc[i]['A'], df.iloc[i]['B'], df.iloc[i]['C'], df.iloc[i]['D'], df.iloc[i]['E']])
            y.append(df.iloc[i]['score'])

        xt,xv,yt,yv = train_test_split(x,y)
        self.svr_rbf.fit(xt,yt)
        self.loss = mean_squared_error(self.svr_rbf.predict(xv),yv)
        print("Loss : ", self.loss)
    
    def output(self,x):
        if type(x[0])==numpy.float64:
            x = [x]

        return self.svr_rbf.predict(x)

    def best_loss(self):
        return self.loss