from classes.SVR import SVR_Model
if __name__=='__main__':
    res = []
    for _ in range(10):
        res.append(SVR_Model(C=2,gamma=1,epsilon=0.1).best_loss())
    print(sum(res)/10,min(res))