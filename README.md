### Hello digits
- Trains a Neural Network python class using data from the mnist dataset
- Tests the trained NN with:
  - test data from mnist
  - imgs drawn from from paint app in windows 10

#### Dataset
- The full dataset in CSV format is available at: https://pjreddie.com/projects/mnist-in-csv/
- Training dataset is saved in path: "mnist_dataset/mnist_train.csv"
- Test dataset is saved in path: "mnist_dataset/mnist_test.csv"

#### Attributes:
- i_nodes: Represents number of nodes in the input layer of the network
  - 28by28/784 for the case of mnist dataset
- h_nodes: Represents number of nodes in the hidden layer 
  - The number is chosen arbitrarily 
- o_nodes: Represents number of nodes in the hidden layer 
  - 10 nodes: Each node represents each numnerical value 0 to 9
- Activation: A class representing a differntiable  activation function.

#### Methods
- foward_propagation: takes in an input list and returns an output of the network
- backward_propagation: 
  - takes in an inputs list,a target list, and a learning rate
  - uses gradient descent to refine weights of the network

  
#### Initialization of weights
- weights are initialised based mormal dist, mean= o, std=1/sqrt( number of incoming links)

#### References
1. Rashid, T 2016, Make Your Own Neural Network.
