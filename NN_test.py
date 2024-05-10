from classes.NN import NN
from classes.NN import NN_BN_model
if __name__=='__main__':
    res = []
    for _ in range(10):
        model = NN_BN_model(64)
        res.append(NN(model).best_loss())
    print(sum(res),min(res))
