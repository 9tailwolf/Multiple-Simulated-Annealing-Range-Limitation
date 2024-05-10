from classes.GR import GR_Model
if __name__=='__main__':
    res = []
    for _ in range(10):
        res.append(GR_Model(n_estimators=50, max_depth=100, learning_rate=0.1).best_loss())
    print(sum(res)/10, min(res))
