Data for training and testing: MNIST data -> mnist.pkl.gz

Data loading: mnist_loader.load_data_wrapper -> load_data -> return (training_data, validation_data, test_data)

Initialisation of structure of the network, biases and weights: Network([784, 30, 10]) -> __init__

Training and testing: net.SGD -> update_mini_batch -> backprop -> sigmoid -> cost_derivative -> update weights and biases -> evaluate -> feedforward -> print
