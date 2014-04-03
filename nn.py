# Name: Mohamed Temraz
# Email: temraz11@gmail.com
# Description: A Neural Network implementation

import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
import cPickle as pickle
import sys

class NN:
    def __init__(self,data,n_hid=100,n_out=10):
        self.data = np.mat(data[:,:-1])
        self.labels = data[:,-1]

        #Train on 80% of data and use validation set for parameter selection
        cases = int(0.8*self.data.shape[0])
        self.train_data = self.data[:cases,:]
        self.valid_data = self.data[cases:,:]
        
        #Used for final accuracy evaluation
        self.train_class = self.labels[:cases]
        self.valid_class = self.labels[cases:]

        #reformat labels
        num_cases = self.data.shape[0]
        labels = np.zeros((num_cases,10))
        indices = list(self.labels.astype(int))
        labels[range(num_cases),indices] = 1.0
        self.train_labels = labels[:cases,:]
        self.valid_labels = labels[cases:,:]

        #visible units
        self.n_vis = self.data.shape[1]
        #hidden units
        self.n_hid = n_hid
        #outputs = 10 classes
        self.n_out = n_out
        #visible-hidden weights
        self.w_vh = np.mat(0.01 * np.random.randn(self.n_vis+1,self.n_hid))
        #hidden-output weights
        self.w_ho = np.mat(0.01 * np.random.randn(self.n_hid+1,self.n_out))

    def train(self,alpha=0.0005,decay=0.0005,momentum=0.9,epochs=500):
        dw_vh = np.zeros(self.w_vh.shape)
        dw_ho = np.zeros(self.w_ho.shape)
        n = self.train_data.shape[0]
        train_CE = np.zeros(epochs)
        valid_CE = np.zeros(epochs)
        for i in xrange(epochs):
            #Forward pass
            #Layer1
            v_transform = np.hstack((np.ones((n,1)),self.train_data))
            h_forward = v_transform * self.w_vh
            h_output = sigmoid(h_forward)
            #Layer2
            h_transform = np.hstack((np.ones((n,1)),h_output))
            o_forward = h_transform * self.w_ho 
            o_output = softmax(o_forward)

            #CrossEntropy Error
            train_CE[i] = -np.mean(np.sum(np.multiply(self.train_labels,np.log(o_output)),axis=1))

            #Backpropagate error derivatives
            dEdout = o_output - self.train_labels 
            dEdw_ho = h_transform.transpose() * dEdout
            dEdtemp1 = dEdout*self.w_ho.transpose() 
            dEdtemp2 = np.multiply(h_transform,1-h_transform) 
            dEdh = np.multiply(dEdtemp1,dEdtemp2)
            dEdw_vh = v_transform.transpose() * dEdh[:,1:]

            #Deriavtives
            dw_vh = momentum*dw_vh - alpha*dEdw_vh - decay*self.w_vh
            dw_ho = momentum*dw_ho - alpha*dEdw_ho - decay*self.w_ho
            
            #Weight updates
            self.w_vh += dw_vh
            self.w_ho += dw_ho
            
            #Evaluation
            valid_CE[i],v_output = self.evaluate(self.valid_data,self.valid_labels)

            #Outout error per epoch as you train
            sys.stdout.write('\rEpoch: %d\n\tTrain_CE: %.6f\tValid_CE: %.6f\n'%(i,train_CE[i],valid_CE[i]))

        #Get Model Accuracy
        train_acc = self.accuracy(o_output,self.train_class)
        valid_acc = self.accuracy(v_output,self.valid_class)
        sys.stdout.write('\rTraining Accuracy: %.2f\tValidation Accuracy: %.2f'%(train_acc,valid_acc))
        sys.stdout.flush()

    def evaluate(self,test_data,test_labels):
        n = test_data.shape[0]
        v_transform = np.hstack((np.ones((n,1)),test_data))
        h_forward = v_transform * self.w_vh
        h_output = sigmoid(h_forward)    
        h_transform = np.hstack((np.ones((n,1)),h_output))
        o_forward = h_transform * self.w_ho 
        o_output = softmax(o_forward)
        return -np.mean(np.sum(np.multiply(test_labels,np.log(o_output)),axis=1)),o_output

    def accuracy(self,model_output,target):
        prediction = np.array(np.argmax(model_output,axis=1)).flatten()
        result = (target.transpose()==prediction)
        return (np.sum(result)/float(result.shape[0]))*100
        
    def plot(self,epochs,CE_train,CE_valid):
        plt.plot(epochs,CE_train,'r',label='Training')
        plt.plot(epochs,CE_valid,'g',label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('CrossEntropy')
        plt.legend()
        plt.show()
    
    def save_parameters(self,parameters):
        pickle.dump(parameters,open('weights.txt','wb'))

def predict_label(x):
    return np.argmax(x,axis=1)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1)

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def loadData(file):
    data = map(list,arff.loadarff(file)[0])
    return np.array(data).astype(float)

#if __name__ == '__main__':
    #Commands used for training
    #data = loadData('train-digits.arff')
    #model = NN(data)
    #model.train()
    #model.save_parameters([model.w_vh,model.w_ho])
