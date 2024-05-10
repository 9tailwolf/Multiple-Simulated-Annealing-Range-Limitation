import pandas as pd

from classes.NN import NN, NN_BN_Model
from classes.SVR import SVR_Model
from classes.SA import Simulated_Annealing

if __name__=='__main__':
    data = {i:[] for i in "ABCDE"}

    for i in range(5):
        model = NN_BN_Model(hidden_layer=64)
        obj = NN(model)
        m,_,x = Simulated_Annealing(obj,200,200,True)
        for a in "ABCDE":
            data[a].append(x[ord(a)-ord('A')])

    for i in range(5):
        obj = SVR_Model(C=2,gamma=1,epsilon=0.1)
        m,_,x = Simulated_Annealing(obj,200,200,True)
        for a in "ABCDE":
            data[a].append(x[ord(a)-ord('A')])

    df = pd.DataFrame(data)
    df.to_csv("./data/query.csv",index=False)

