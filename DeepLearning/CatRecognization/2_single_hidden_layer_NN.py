import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
from skimage import io,transform,color

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  #shape:(209,64,64,3)
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
    
    
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    
    
    
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    train_m=train_set_x_orig.shape[0]
    train_set_x_flatten=train_set_x_orig.reshape(train_m,-1).T  #shape:(64*64*3,209)
    train_set_x=train_set_x_flatten/255
    test_m=test_set_x_orig.shape[0]
    test_set_x_flatten=test_set_x_orig.reshape(test_m,-1).T
    test_set_x=test_set_x_flatten/255
    
    return train_set_x, train_set_y, test_set_x, test_set_y, classes

def sigmoid(z):
    return 1/(1+np.exp(-z))
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    return parameters
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return cache
def compute_cost(A2, Y):
    m = Y.shape[1]
    logprobs = Y*np.log(A2) + (1-Y)* np.log(1-A2)
    cost = -1/m * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    A1=cache['A1']
    A2=cache['A2']
    W2 = parameters["W2"]
    
    
    
    dZ2= A2 - Y
    dW2 = 1 / m * np.dot(dZ2,A1.T)
    db2 = 1 / m * np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1,2))
    dW1 = 1 / m * np.dot(dZ1,X.T)
    db1 = 1 / m * np.sum(dZ1,axis=1,keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
def nn_model(X, Y, n_h, num_iterations = 10000,learning_rate = 0.1,print_cost=False):
    n_x =X.shape[0]
    n_y = Y.shape[0]
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(0, num_iterations):
        cache = forward_propagation(X, parameters)
        cost = compute_cost(cache['A2'], Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads,learning_rate)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i+1000, cost))
    return parameters
def predict(parameters, X):
    cache=forward_propagation(X, parameters)
    A2=cache['A2']
    for i in range(A2.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if(A2[0][i]>0.5):
            A2[0][i]=1
        else:
            A2[0][i]=0
    return A2
def print_accuracy(Y,predict):
    m=Y.shape[1]
    accuracy=0
    for i in range(m):
        if(Y[0][i]==predict[0][i]):
            accuracy+=1
    accuracy=accuracy/m*100
    
    print("Total:"+str(m)+"\nAccuray:"+str(accuracy)+"%")
def print_result(X,predict):
    m=X.shape[1]
    for i in range(m//4):
        index = i
        plt.imshow(X[:,index].reshape((num_px, num_px, 3)))
        plt.show()
        print ("y = " + str(test_set_y[0,index]) +\
               ", you predicted that it is a \"" + \
               classes[int(predict[0][i])].decode("utf-8") +  "\" picture.")
def get_my_picture():
    my_x=[]
    path='./my_set'+'/*.jpg'
    collection=io.ImageCollection(path)
    for i in range(len(collection)):
        img=collection[i]
        if(len(img.shape)==3):
            img=img[:,:,:3]
            img=transform.resize(img,(num_px,num_px))
            my_x.append(img)
    my_x=np.array(my_x)
    num=my_x.shape[0]
    my_x=my_x.reshape(num,-1).T
    return my_x
def write_file(parameters):
    for key,value in parameters.items():
        parameters[key]=value.tolist()
    f=open('parameters.txt','w')
    f.write(json.dumps(parameters))
    f.close()
def read_file():
    #reading from the file
    f=open('parameters.txt','r')
    data=json.loads(f.read())
    for key,value in data.items():
        data[key]=np.array(value)
    f.close()
    return data

train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
num_px=64

#m=test_set_x.shape[1]
#for i in range(m):
#    x=test_set_x[:,i]
#    img=x.reshape((num_px,num_px,3))
#    io.imsave('./test_set/test'+str(i)+'.jpg',img)
#建立模型
READ_FILE=0
WRITE_FILE=0
if(READ_FILE==0):
    parameters=nn_model(train_set_x,train_set_y,5,num_iterations = 3000,learning_rate = 0.05, print_cost=True)
else:
    parameters=read_file()
if(WRITE_FILE==1):
    write_file(parameters)

test_prediction=predict(parameters,test_set_x)
train_prediction=predict(parameters,train_set_x)
#My data set
my_set_x=get_my_picture()
my_prediction=predict(parameters,my_set_x/255)

PRINT=0
if(PRINT==0):
    print_result(my_set_x,my_prediction)
elif(PRINT==1):
    print_result(test_set_x,test_prediction)
elif(PRINT==2):
    print_result(train_set_x,train_prediction)

print("Test_set")
print_accuracy(test_set_y,test_prediction)
print("Train_set")
print_accuracy(train_set_y,train_prediction)