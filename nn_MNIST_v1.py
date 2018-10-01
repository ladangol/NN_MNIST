#neural network for handwritten digits in the dataset mnist
import numpy as np
import scipy.io as sio  #for loadmat and minimize
from scipy.optimize import minimize
from time import time
from sklearn.metrics import confusion_matrix


######Reading and Preprocessing Data
######
def preprocess():
    mat = sio.loadmat('mnist_all')
    # the type of mat is a dictionary, were the keys are the
    #matlab variables and values are the objects assigned to otherwise
    #variables

    #Here the keys are (mat.keys()):
    #'train0', 'test0', 'train1', 'test1', 'train2', 'test2', 'train3', 'test3', 'train4', 'test4', 'train5', 'test5', 'train6', 'test6', 'train7', 'test7', 'train8', 'test8', 'train9', 'test9'
    #each key contains the images for that number
    #for example train0 contains 5923 examples of digit 0
    # The test data (test0 to test9) in overall is 10000 examples
    # And the train data is 60000 examples

    # the number of columns for each data example is 784 columns
    train_preprocess = np.zeros(shape= (50000, 784))
    train_labels = np.zeros(shape = (50000,)) # array pf 50000 elements
    test_preprocess = np.zeros(shape=(10000, 784))
    test_labels = np.zeros(shape = (10000, ))
    # We take 1000 records of each train data to the validation set
    # Therefore 10(number of train sets) * 1000 = 10000
    validation_preprocess = np.zeros(shape=(10000, 784))
    validation_labels= np.zeros(shape=(10000,))

    train_index = 0
    test_index = 0
    validation_len = 0
    valid_offset = 1000
    for key in mat:
        if 'train' in key:
            label = key[-1]  #key is for example train0 and key[-1] is 0
            values = mat.get(key)
            rownum = values.shape[0]  # or len (values)
            permindices  = np.random.permutation(rownum)
            endindex = rownum - valid_offset
            train_preprocess[train_index: train_index+endindex] = values[permindices[valid_offset:], :]
            train_labels[train_index: train_index+ endindex] = label
            train_index += endindex
            validation_preprocess[validation_len:validation_len + 1000] = values[permindices[0:1000], :]
            #validation_len += 1000
            validation_labels[validation_len:validation_len + 1000] = label
            validation_len += 1000
        if 'test' in key:
            label = key[-1]
            values = mat.get(key)
            test_preprocess[test_index: test_index+len(values)] = values
            test_labels[test_index: test_index+len(values)] = label
            test_index += len(values)

    #Permutation
    permindices = np.random.permutation(len(train_preprocess))
    train_data = train_preprocess[permindices]
    train_labels = train_labels[permindices]

    permindices = np.random.permutation(len(validation_preprocess))
    valid_data = validation_preprocess[permindices]
    validation_labels = validation_labels[permindices]

    permindices = np.random.permutation(len(test_preprocess))
    test_data = test_preprocess[permindices]
    test_labels = test_labels[permindices]

    ##Normalizing the values in the train, test and validation sets
    # the values are between 0 to 255, 0 is white 255 is Noir and
    #any other number is a shade of a gray (gris)
    train_data = np.double(train_data)/255.0
    test_data = np.double(test_data)/255.0
    valid_data = np.double(valid_data)/255.0


    ##Now removing useless features
    features_count = train_data.shape[1]
    #Note that train data is being chosen randomly, therefore at each run
    #there might be different features depending on data

    min_elements = np.amin(train_data, axis = 0) #select the lowest numbers in min(), min()
    max_elements = np.amax(train_data, axis = 0) #select the maximum of each element in rows max(), max(),

    columns_to_delete = []

    for i in range(features_count):
        if(min_elements[i] == max_elements[i]):
            columns_to_delete.append(i)
            #print ('deleting column ' + str(i))
    #Removing useless features
    #print ('to be deleted' + str(len(columns_to_delete)))
    #print (columns_to_delete)
    train_data = np.delete(train_data, columns_to_delete, axis = 1)
    valid_data = np.delete(valid_data, columns_to_delete, axis = 1)
    test_data = np.delete(test_data, columns_to_delete, axis = 1)

    return train_data, test_data, valid_data, train_labels, test_labels, validation_labels

#print ('tarin_data:' + str(train_data.shape[0]) + ' and ' + str(train_data.shape[1]))
#print ('valid_data:' + str(valid_data.shape[0]) + ' and ' + str(valid_data.shape[1]))

##########preprocessing done

####Weight initialization
def initializeWeights(n_in, n_out):

    #n_in is fan in: number of nodes in the previous layer
    #n_out is fan out: number of nodes in the next layer

    #  initializeWeights return the random weights for Neural Network
    # the weight initialization depends on the activation function
    # You can read about it here http://www.junlulocky.com/actfuncoverview
    #I use the weight initialization for the sigmod activation function


    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    low = -4 * np.sqrt(6)/np.sqrt(n_in+n_out+1)
    W = np.random.uniform(low, -1*low, [n_out, (n_in+1)])
    return W


##SIGMOD function (Actvation fnction)
def sigmod(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    res = 1.0 / (1 + np.exp(-1.0 * z))
    return res

def one_K_coding_scheme(training_label,n_class):
    size = np.size(training_label)
    out = np.zeros((size, n_class), dtype=np.int)
    for i in range(size):
        index = int(training_label[i])
        out[i][index] = 1
    return out

##NNObjFunction  we try to minimize otherwise
def nnObjfun(params, *args):
    """ Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
     % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices."""

    """
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer.
    """
    objective_value = 0
    objective_grad = np.array([])
    (n_input, n_hidden, n_class, train_data, train_labels, lambdaa) = args
    #print (str(n_input) + ' class ' + str(n_class))
    #print ('params length is' + str(len(params)))
    W1 = params[0:(n_hidden * (n_input+1))].reshape(n_hidden, (n_input+1))
    W2 = params[(n_hidden*(n_input+1)):].reshape(n_class, (n_hidden+1))

    data_count = train_data.shape[0]
    labels = one_K_coding_scheme(train_labels, n_class)

    #feed forward returns data with bias
    data, hidden_layer_out, output_layer = feed_forward(train_data, W1, W2)
    labels = labels.transpose()

    #Output_layer_out has n_class*number of input data (50000) dimension
    #labels have n_class * number of input data dimention

    #Log likelihood error for each data:

    log_likelihood_error = -1 * ((labels * np.log(output_layer)) + ((1-labels) * np.log(1 - output_layer)))
    log_likelihood_fun = np.sum(log_likelihood_error[:]) / data_count

    #Adding Regularization

    regularization = (lambdaa/(2*data_count)) * (np.sum(W1**2) + np.sum(W2**2))
    objective_value = log_likelihood_fun + regularization

    #W2_error
    #output_layer = n_class * input
    # so need the transpose of labels
    output_delta = output_layer - labels  #labels should be input * n_class
    W2_error = np.dot(output_delta ,hidden_layer_out.T)
    W2_grad = (W2_error + lambdaa * (W2)) / (data_count)

    #W1_error
    hidden_delta = ((1-hidden_layer_out)*hidden_layer_out) * np.dot(W2.T,output_delta)
    W1_error = np.dot(hidden_delta,data.T)
    #print ('W1_error matrix size is %d * %d' % (W1_error.shape[0], W1_error.shape[1]))
    #n_hidden+1 * (features + bias)
    W1_error = W1_error[:-1, :]  #ignoring input_bias of hidden layers
    #n_hidden * (features + bias)

    #print ('W1_error matrix size is %d * %d' % (W1_error.shape[0], W1_error.shape[1]))
    W1_grad = (W1_error + (lambdaa * W1)) / (data_count)

    objective_grad = np.concatenate((W1_grad.flatten(), W2_grad.flatten()))
    print ('Objective Value: ' + str(objective_value))
    return (objective_value, objective_grad)

 ##Feedforward

def feed_forward(train_data, W1, W2):
     input_layer = train_data.T   #remember number of features is the size of the input, so I need to transpose the matrix
     col = input_layer.shape[1]  #or np.size(train_daya,1)
     #input_bias = np.ones((col,), dtype=int) will produce error that the array do not match
     input_bias = np.ones((1, col), dtype = int)
     #print ('input_layer size before adding bias is %d * %d' %(len(input_layer), col))
     input_layer = np.concatenate((input_layer, input_bias), axis = 0)
     #print ('input_layer size before adding bias is %d * %d' %(len(input_layer), input_layer.shape[1]))
     #print ('W1 size = %d * %d' %(W1.shape[0], W1.shape[1]))
     #print ('Input Layer size = %d * %d' %(input_layer.shape[0], input_layer.shape[1]))
     #so it should be wj1*x1 + wj2*x2 + wj3*x3 + ... wj(d+1)*x(d+1)
     hidden_layer_in = np.dot(W1, input_layer)

     hidden_layer_out = sigmod(hidden_layer_in)

     #Now add bias
     col = hidden_layer_out.shape[1]
     hidden_bias = np.ones((1, col), dtype = int)
     hidden_layer_out = np.concatenate((hidden_layer_out, hidden_bias), axis = 0)

     output_layer_in = np.dot(W2, hidden_layer_out)

     output_layer = sigmod(output_layer_in)
     #print ('Output Layer size = %d * %d' %(output_layer.shape[0], output_layer.shape[1]))
     return input_layer, hidden_layer_out, output_layer


def one_of_K_coding(train_labels, n_class):
    labels = np.zeros((len(train_labels), n_class), dtype = int)
    for i in range(len(train_labels)):
        col = train_labels[i]
        labels[i][col] = 1
    return labels

def nnPredict(W1, W2, data):

    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""
    labels = np.array([])
    data, hidden_out, output_out = feed_forward(data, W1, W2)
    #print ('Output Layer size = %d * %d' %(output_out.shape[0], output_out.shape[1]))
    labels = np.argmax(output_out, axis = 0)
    return labels


##Neural Network Parameters
train_data, test_data, valid_data, train_labels, test_labels, validation_labels = preprocess()

n_input = train_data.shape[1]  # this is the number of featues
#Number of nodes in the output layer
n_class = 10  # each for a digit from 0 to 9

#Number of nodes in the hidden layer
n_hidden = 28

#labmda which is for regularization
lambdaa = 0.1

n_in = n_input
n_out = n_hidden
W1 = initializeWeights(n_in, n_out)

n_in = n_hidden
n_out = n_class
W2 = initializeWeights(n_in, n_out)

#Note that during the initializeWeights we add the weights for bias nodes as well
#So the dimention is n_out * n_in+1

args= (n_input, n_hidden, n_class, train_data, train_labels, lambdaa)
initial_weights = np.concatenate((W1.flatten(), W2.flatten()))
#print (len(params))
#print ('W1 is %d * %d ' % (len(W1), W1.shape[1]))
#print ('W2 is %d * %d ' % (len(W2), W2.shape[1]))
opts = {'maxiter': 50}


###Training is started
T1 = time()
nn_params = minimize(nnObjfun, initial_weights, jac=True, args=args, method='CG', options=opts)
T2 = time()
###Training is done
# Reshape nnParams from 1D vector into w1 and w2 matrices
W1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
W2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(W1, W2, train_data)

# find the accuracy on Training Dataset

print('Training set Accuracy:' + str(100 * np.mean((predicted_label == train_labels).astype(float))) + '%')

predicted_label = nnPredict(W1, W2, valid_data)

# find the accuracy on Validation Dataset



print('Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_labels).astype(float))) + '%')
#
print('Validation Set Confusion Matrix:')
cm = confusion_matrix(validation_labels, predicted_label)
print (cm)

predicted_label = nnPredict(W1, W2, test_data)
#print("valid label size" + str(validation_labels.shape[0]))
print('Test set Accuracy:' + str(100 * np.mean((predicted_label == test_labels).astype(float))) + '%')
##
print('Test set Confusion Matrix:')
cm = confusion_matrix(test_labels, predicted_label)
print (cm)
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
print ('TP' +str(TP))
print('FP' + str(FP))
print('FN' + str(FN))
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print('Precision'+ str(precision))
print('Recall' + str(recall))
