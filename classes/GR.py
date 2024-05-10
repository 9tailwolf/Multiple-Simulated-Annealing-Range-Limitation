from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy

class GR_Model():
    def __init__(self,n_estimators=100,max_depth=50,learning_rate=0.1):
        self.gr = GradientBoostingRegressor(loss='squared_error',n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
        df = pd.read_csv("data/log.csv")
        x, y = [],[]
        for i in range(df.shape[0]):
            x.append([df.iloc[i]['A'], df.iloc[i]['B'], df.iloc[i]['C'], df.iloc[i]['D'], df.iloc[i]['E']])
            y.append(df.iloc[i]['score'])

        xt,xv,yt,yv = train_test_split(x,y)
        self.gr.fit(xt,yt)
        self.loss = mean_squared_error(self.gr.predict(xv),yv)
        print("Loss :", self.loss)
    
    def output(self,x):
        if type(x[0])==numpy.float64 or type(x[0])==float:
            x = [x]
        return self.gr.predict(x)
    
    def best_loss(self):
        return self.loss
