import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import theano
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import cPickle as pickle
import warnings
from sklearn import metrics
import os
import sys

# Counter for decaying learning rate per epoch
counter = 0

# Random number generator
rand_gen = np.random.RandomState(123)
theano_rng = RandomStreams(rand_gen.randint(2*30))

class Layer:

    def __init__(self,input,n_in,n_out,activation=T.tanh,dropout=0):
        '''Creates additional hidden and output layers to NeuralNetwrok'''
        
        # Initialize weights
        W = np.asarray(rand_gen.uniform(
                  low=-np.sqrt(6./(n_in+n_out)),
                  high=np.sqrt(6./(n_in+n_out)),
                  size=(n_in,n_out)),dtype=theano.config.floatX)
        
        self.W = theano.shared(value=W,name='W')
        
        # Initialize biases for this layer
        self.bias = theano.shared(value=np.zeros((n_out,),dtype=theano.config.floatX),name='bias')
        
        # Learnable parameters for this layer
        self.params = [self.W,self.bias]

        # Actvation outputs for this layer
        if dropout:
            self.output = activation(T.dot(input,self.W*2)+self.bias)
        else:
            self.output = activation(T.dot(input,self.W)+self.bias)


class NeuralNetwork:

    def __init__(self,input,n_vis,n_hid_1,n_hid_2,n_hid_3,n_out,labels,noise,training):
        '''
        Creates a multi-layer NeuralNetwork with specified network architecture
        '''

        # Input data
        self.x = input
        
        # Output labels representing driver or passenger
        self.labels = labels
        
        # Preserve units on validation and test data
        if not training:

            # Layer 1
            self.first_hidden_layer = Layer(self.x,n_vis,n_hid_1)
            
            # Layer 2
            self.second_hidden_layer = Layer(self.first_hidden_layer.output,n_hid_1,n_hid_2)
            
            # Layer 3
            self.third_hidden_layer = Layer(self.second_hidden_layer.output,n_hid_2,n_hid_3)
            
            # Ouput layer
            self.output_layer = Layer(self.third_hidden_layer.output,n_hid_3,n_out,activation=T.nnet.softmax)
          
        # Dropout hidden units on training data
        else:
            
            # Layer 1
            self.first_hidden_layer = Layer(self.x,n_vis,n_hid_1,dropout=1)
            
            # Layer 2
            self.second_hidden_layer = Layer(self.corrupt(self.first_hidden_layer.output,noise),n_hid_1,n_hid_2,dropout=1)
           
            # Layer 3
            self.third_hidden_layer = Layer(self.corrupt(self.second_hidden_layer.output,noise),n_hid_2,n_hid_3,dropout=1)

            # Output layer
            self.output_layer = Layer(self.corrupt(self.third_hidden_layer.output,noise),n_hid_3,n_out,activation=T.nnet.softmax)
            
        # Model parameters
        self.params = self.first_hidden_layer.params+self.second_hidden_layer.params+self.third_hidden_layer.params+self.output_layer.params


    def updateModel(self,lr,decay,factor):
        '''
        Updates model parameters and returns the error on a mini-batch
        '''
        # CrossEntropy error
        CE = T.mean(-T.sum(self.labels*T.log(self.output_layer.output),axis=1))
        
        # Weight-decay
        regularization = decay*((self.first_hidden_layer.W**2).sum()+(self.second_hidden_layer.W**2).sum()+(self.output_layer.W**2).sum()
                                +(self.third_hidden_layer.W**2).sum())
        
        CE += regularization
        
        # Gradients
        grad_params = [T.grad(CE,parameter) for parameter in self.params]

        global counter
        counter += 1
        # Decaying rate
        lr = lr/(1+factor*counter)

        # Update parameters
        updates = [(p,p-lr*g) for p,g in zip(self.params,grad_params)]
        
        return CE,updates


    def getAccuracy(self,target):
        '''Returns accuracy of model'''
        
        # Get models prediction
        prediction = T.argmax(self.output_layer.output,axis=1)
        
        # Mean error rate
        return prediction,T.mean(T.neq(prediction,T.argmax(target,axis=1)))*100

    def corrupt(self,x,noise):
        '''Adds random noise to data'''

        return theano_rng.binomial(size=x.shape,n=1,p=1-noise)*x

def relu(input):
    '''
    Performs a rectified linear activation on the input
    '''
    return T.maximum(0,input)

def fit(dataset,classes,test_data,test_labels,n_hid_1,n_hid_2,n_hid_3,lr,factor,decay,noise,batch_size,num_epochs):
    '''
    Trains the NeuralNetwork and reports the percentage of miscalssified examples
    
    Parameters:
    
    @dataset -> training dataset (number of examples x number of features)
    @classes -> labels for training data (number of examples x number of classes)
    @test_data -> test dataset (same format as dataset)
    @test_labels -> labels for test data (same format as classes)
    @noise -> corruption factor for dropout
    @n_hid_1 -> number of hidden units in first layer
    @n_hid_2 -> number of hidden units in second layer 
    @n_hid_3 -> number of hidden units in third layer
    @lr -> learning rate
    @decay -> weight decay (model performs L2 regularization)
    @num_epochs -> number of epochs to train the model
    @batch_size -> size of batch training for parameter update
    @factor -> decaying constant for learning rate
    '''
    warnings.simplefilter('ignore', DeprecationWarning)
    sys.stdout.write('Training a Neural Network...\n')
    
    # Convert classes from numpy array to theano tensor
    labels = theano.shared(classes)
    
    # Create training,validaton,and test sets
    n_cases = dataset.shape[0]   
    
    # Use 80% for training
    d_train = dataset[:int(0.8*n_cases),:]
    l_train = labels[:int(0.8*n_cases)]
    
    # Use 20% for validation 
    d_valid = dataset[int(0.8*n_cases):,:]
    l_valid = labels[int(0.8*n_cases):]
    
    # Test on a new dataset
    d_test = test_data
    l_test = theano.shared(test_labels)


    # Batch index
    index = T.lscalar()
    
    # Batch sizes for training,validation and test sets
    train_num_batches = d_train.shape[0]//batch_size
    valid_num_batches = d_valid.shape[0]//batch_size
    test_num_batches = d_test.shape[0]//batch_size

    # Convert to theano tensor
    data_train = theano.shared(d_train)
    data_valid = theano.shared(d_valid)
    data_test = theano.shared(d_test)
    
    
    # Symbolic variable to represent dataset and labels 
    x = T.dmatrix('x')
    y = T.dmatrix('y')

    # Symbolic scalar to dropout hidden units on training only
    z = T.lscalar('z')
    
    # Create a neural network class with desired number of hidden units
    model = NeuralNetwork(x,dataset.shape[1],n_hid_1,n_hid_2,n_hid_3,classes.shape[1],y,noise,z)
    
    # Errors
    error,updates = model.updateModel(lr,decay,factor)
    
    # Training function
    train_model = function([index],error,updates=updates,givens={x:data_train[index*batch_size:batch_size*(index+1)],
                                                                 y:l_train[index*batch_size:batch_size*(index+1)],
                                                                 z:index**0})
    
    # Report accuracy on training set
    train_acc = function([],outputs=model.getAccuracy(y),givens={x:data_train,y:l_train})
    
    # Validating function 
    validate_model = function([index],error,givens={x:data_valid[index*batch_size:batch_size*(index+1)],
                                                    y:l_valid[index*batch_size:batch_size*(index+1)],
                                                    z:index*0})
    
    # Report accuracy on validation set
    valid_acc = function([],outputs=model.getAccuracy(y),givens={x:data_valid,y:l_valid})
    
    # Testing function
    test_model = function([index],error,givens={x:data_test[index*batch_size:batch_size*(index+1)],
                                                y:l_test[index*batch_size:batch_size*(index+1)],
                                                z:index*0})
    
    # Report accuracy on test set
    test_acc = function([],outputs=model.getAccuracy(y),givens={x:data_test,y:l_test})

    # Training,validation,and test errors per epoch
    training_err = np.zeros((num_epochs,1))
    validation_err = np.zeros((num_epochs,1))
    test_err = np.zeros((num_epochs,1))
    
    
    # Mini-Batch Stochastic Gradient Descent (MBSGD)
    for epoch in xrange(num_epochs):
        sys.stdout.write('Epoch %d\n'%(epoch))
        
        # Get error on datasets
        training_err[epoch] = np.mean([train_model(batch) for batch in xrange(train_num_batches)])
        validation_err[epoch] = np.mean([validate_model(batch) for batch in xrange(valid_num_batches)])
        test_err[epoch] = np.mean([test_model(batch) for batch in xrange(test_num_batches)])
        sys.stdout.write('Training Error:%.5f \tValidation Error:%.5f\tTest Error:%.5f\n'%(training_err[epoch],validation_err[epoch],test_err[epoch]))
    
        # Get accuracy on datasets
        train_set_acc = 100-train_acc()[1]
        valid_set_acc = 100-valid_acc()[1]
        test_set_acc = 100-test_acc()[1]
        sys.stdout.write('TrainingAccuracy: %.2f\tValidationAccuracy: %.2f\tTestAccuracy: %.2f\n'
                     %(train_set_acc,valid_set_acc,test_set_acc))

    # Classification Report
    valid_pred = valid_acc()[0]
    valid_expected = np.argmax(classes[int(0.8*n_cases):],axis=1)
    valid_report = metrics.classification_report(valid_expected,valid_pred)
    print 'Validation Report'
    print valid_report

    test_pred = test_acc()[0]
    test_expected = np.argmax(test_labels,axis=1)
    test_report = metrics.classification_report(test_expected,test_pred)
    print 'Test Report'
    print test_report

    # Plot cross-entropy error per epoch
    plt.figure(1)
    plt.plot(range(num_epochs),training_err,'r',label='Training')
    plt.plot(range(num_epochs),validation_err,'b',label='Validation')
    plt.plot(range(num_epochs),test_err,'g',label='Test')
    plt.xlabel('Epochs');plt.ylabel('Error')
    plt.legend()
    plt.show()
