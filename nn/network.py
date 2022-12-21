import numpy as np

class Neural_Network:
    """
    A class used to represent a Neural Network with three layers:
        Input layer, Hidden layer, Output layer

    Data Attributes
    ----------
    i_nodes : int
        number of nodes in the input layer of the network
    h_nodes : int
        number of nodes in the hidden layer of the network
    o_nodes : int
        number of nodes in the output layer of the network
    Activation: class
        a class that represents an activation function
        the class should have two methods:
            1. func: the activation function
            2. der: the derivative of the activation function

    Methods
    -------
    foward_propagation(inputs_list)
        Parameters
        ----------
        inputs_list: features of a dataset
        
        returns:
            inputs as a transposed 2D numpy array, 
            hidden_outputs:
              a dot product of input_hidden_layer weights and inputs
              the dot product is passed through an activation function
            final_outputs:
                a dot product of hidden_output_layer weights and hidden_outputs
                the dot product is passed through an activation function
    
    backward_propagation(inputs_list, targets_list,lr)
        Parameters
        ----------
        inputs_list:  features of a dataset
        targets_list: desired values to be predicted
        lr: learning rate of network
        
        Uses gradient descent to refine weights of the network
    
    reset_weights:
       Resets weights of a neural network
    
    """
    
    def __init__(self,i_nodes, h_nodes, o_nodes, Activation):
        #set number of node in each input
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes
        
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # 
        # w_ih: link weight matrix between input and hidden layer,  
        self.w_ih = np.random.normal(0.0,pow(self.h_nodes, -0.5),(self.h_nodes,self.i_nodes)) 
        # w_ho: link weight matrix between hidden layer and output layer,  
        self.w_oh = np.random.normal(0.0,pow(self.o_nodes, -0.5),(self.o_nodes,self.h_nodes)) 
        
        # activation function:
        self.activation = Activation()
        
    def foward_propagation(self,inputs_list):
        #convert inputs to a 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        
        #compress inputs
        output_func = lambda w, x: self.activation.func(np.dot(w, x))

        #calculate signals emerging from the hidden layer
        hidden_outputs = output_func(self.w_ih, inputs)

        #calculate signals emerging from the final output layer
        final_outputs =  output_func(self.w_oh,hidden_outputs)

        return inputs, hidden_outputs, final_outputs
    
    def backward_propagation(self, inputs_list, targets_list,lr):
        #return final output of signals
        inputs,hidden_outputs, final_outputs = self.foward_propagation(inputs_list)
        
        final_inputs = np.dot(self.w_oh,hidden_outputs)
        
        hidden_inputs = np.dot(self.w_ih, inputs)
        
        #convert targets to a 2D array
        targets = np.array(targets_list, ndmin=2).T
        
        #output layer error = target - final_outputs
        output_errors = targets - final_outputs
        
        # hidden layer error = output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.w_oh.T, output_errors)
        
        update_weights = lambda err, y, x: lr*np.dot(err*self.activation.der(y), x.T)

        #For weights between output layer and hidden layer use output_errors
        #update the weights for the links between the hidden and output layers
        self.w_oh += update_weights(output_errors,final_inputs,hidden_outputs)
        
        #For weights between hidden layer and input layer use input_errors
        #update the weights for the links between the input and hidden layers
        self.w_ih += update_weights(hidden_errors,hidden_inputs,inputs)
        
    def reset_weights(self):
        # w_ih: link weight matrix between input and hidden layer,  
        self.w_ih = np.random.normal(0.0,pow(self.h_nodes, -0.5),(self.h_nodes,self.i_nodes)) 
        # w_ho: link weight matrix between hidden layer and output layer,  
        self.w_oh = np.random.normal(0.0,pow(self.o_nodes, -0.5),(self.o_nodes,self.h_nodes)) 