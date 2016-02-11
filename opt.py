import model
import load
import input_data
from hyperopt import fmin, tpe, hp
import numpy
numpy.set_printoptions(threshold='nan')

space = [hp.quniform('lr', 0.00001, 1, 0.00001),
         hp.quniform('bs', 100, 10000, 100),
         hp.quniform('fhl', 10, 200, 10),
         hp.quniform('shl', 10, 200, 10)]

data = load.read_data_sets ('data','labels',0.3,0.1, num = 1000000)

model.train_once (data, 0.02, 50000, 0, ermul=10, H1=200, H2=100)

# def cost ((lr, bs, fhl, shl)):
#     return model.train_once (data, lr, int(bs), 0, int(fhl), int(shl), 31, 1) #(data, 0.003, 5000, 0, 150, 50, 31, 1)

# best = fmin(cost,
#             space,
#             algo=tpe.suggest,
#             max_evals=1000)

# print best
