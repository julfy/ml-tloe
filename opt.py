import model
import transform
import load
import numpy

# from hyperopt import fmin, tpe, hp
# space = [hp.quniform('lr', 0.00001, 1, 0.00001),
#          hp.quniform('bs', 100, 10000, 100),
#          hp.quniform('fhl', 10, 200, 10),
#          hp.quniform('shl', 10, 200, 10)]

numpy.set_printoptions(threshold='nan')
numpy.set_printoptions(precision=2)

transform.transform_data ("/home/bogdan/work/repos/ml-tloe/serps/results/*", 'expanded', 10000)

data = load.read_data_sets ('expanded/*',0.3,0.1, num = 00000);

model.create ( H1=1, H2=50 )

# model.train (data, learning_rate=0.001, batch_size=100000, lmbda=0, ermul=10000, restore=False)

model.run(data)

################################################################################
# def cost ((lr, bs, fhl, shl)):
#     return model.train_once (data, lr, int(bs), 0, int(fhl), int(shl), 31, 1) #(data, 0.003, 5000, 0, 150, 50, 31, 1)

# best = fmin(cost,
#             space,
#             algo=tpe.suggest,
#             max_evals=1000)

# print best
