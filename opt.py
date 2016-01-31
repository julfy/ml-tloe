import model
import load

data = load.read_data_sets ('data','labels',0.3,0.1)

model.train_once (data, 0.001, 10, 0, 50, 30, 784, 10)
