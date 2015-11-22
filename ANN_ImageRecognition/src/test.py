import conv
from conv import Network
from conv import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = conv.load_data_shared()
mini_batch_size = 1
net = Network([
    FullyConnectedLayer(n_in=16, n_out=100),
    SoftmaxLayer(n_in=100, n_out=4)], mini_batch_size)

#net = Network([
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 4, 4),
#                  filter_shape=(8, 1, 1, 1),
#                  poolsize=(2, 2)),
#    FullyConnectedLayer(n_in=8*2*2, n_out=60),
#    SoftmaxLayer(n_in=60,n_out=4)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.05, validation_data, test_data)