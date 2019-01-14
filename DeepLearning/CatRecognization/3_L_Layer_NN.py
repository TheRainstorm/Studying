import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage import io,transform
import json

def sigmoid(z):
    return 1/(1+np.exp(-z))
def relu(z):
    return (np.abs(z)+z)/2
def leak_relu(z):
    return (np.abs(z)+z)/2+(np.abs(-z)+z)*0.005
def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)-1          # number of layers in the network

    for l in range(1, L+1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    return parameters 
def L_layer_forward_relu(A_pre,W,b):
    Z=np.matmul(W,A_pre)+b
    A=relu(Z)
    #A=sigmoid(Z)
    #A=leak_relu(Z)
    return A,Z
def L_layer_forward_sigmoid(A_pre,W,b):
    Z=np.matmul(W,A_pre)+b
    A=sigmoid(Z)
    return A,Z

def L_model_forward(X, parameters):
    L=len(parameters)//2

    cache_A=[X]#保证cache_A[l]为l层
    cache_Z=[X]#保证cache_Z[l]为l层
    A=X 
    for l in range(1,L):
        A,Z=L_layer_forward_relu(A,parameters['W' + str(l)],parameters['b' + str(l)])
        cache_A.append(A)
        cache_Z.append(Z)
        
    A,Z=L_layer_forward_sigmoid(A,parameters['W' + str(L)],parameters['b' + str(L)])
    cache_A.append(A)
    cache_Z.append(Z)
    caches=(cache_A,cache_Z)
    return A,caches


def compute_cost(AL, Y):
    m=Y.shape[1]
    return np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))*-1/m
def compute_cost_regulation(AL, Y,para,lambd):
    m=Y.shape[1]
    cost1=np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))*-1/m
    L=len(para)//2
    cost2=0
    for l in range(1,L+1):
        cost2+=np.sum(np.square(para['W'+str(l)]))*lambd/m/2
    return cost1+cost2
def sigmoid_backward(A,Y):
    dZ=A-Y
    return dZ
def sigmoid_backward2(dA,Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    
    return dZ
def relu_backward(dA,Z):
#    dg_Z=(Z>0)*1#神奇的实现
#    dZ=dA*dg_Z
    
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    return dZ

def leak_relu_backward(dA,Z):
    return (Z>0)*0.09+0.01
def linear_backward(dZ,W,A_prev):
    m=dZ.shape[1]
    dW=1/m*np.matmul(dZ,A_prev.T)
    dA_prev=np.matmul(W.T,dZ)
    db=1/m*np.sum(dZ,axis = 1 ,keepdims=True)
    
    return dA_prev,dW,db
def L_model_backward(AL, Y, parameters,caches):
    cache_A,cache_Z=caches
    L=len(cache_A)-1
    
    grads={}
    
    dZ=sigmoid_backward(AL,Y)
    dA_prev,dW,db=linear_backward(dZ,parameters['W'+str(L)],cache_A[L-1])
    grads['dW'+str(L)]=dW
    grads['db'+str(L)]=db
    for l in range(L-1,0,-1):
        dZ=relu_backward(dA_prev,cache_Z[l])
        #dZ=leak_relu_backward(dA_prev,cache_Z[l])
        #dZ=sigmoid_backward2(dA_prev,cache_Z[l])
        dA_prev,dW,db=linear_backward(dZ,parameters['W'+str(l)],cache_A[l-1])
        grads['dW'+str(l)]=dW
        grads['db'+str(l)]=db
    return grads

def update_parameters(parameters, grads, learning_rate):
    L=len(parameters)//2
    for l in range(1,L+1):
        parameters['W'+str(l)]=parameters['W'+str(l)]-learning_rate*grads['dW'+str(l)]
        parameters['b'+str(l)]=parameters['b'+str(l)]-learning_rate*grads['db'+str(l)]
    return parameters

def update_parameters_regulation(parameters, grads, learning_rate,m,lambd):
    L=len(parameters)//2
    for l in range(1,L+1):
        parameters['W'+str(l)]=(1-(lambd*learning_rate)/m)*parameters['W'+str(l)]-learning_rate*grads['dW'+str(l)]
        parameters['b'+str(l)]=parameters['b'+str(l)]-learning_rate*grads['db'+str(l)]
    return parameters


def L_Layer_Model(X, Y, layers_dims, cost_per=100,learning_rate = 0.01, num_iterations = 3000, lambd=0,print_cost=True,plot_cost=False):#lr was 0.009
    m=X.shape[1]
    costs = []                         # keep track of cost
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations+1):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        
        #cost = compute_cost(AL, Y)
        cost = compute_cost_regulation(AL, Y,parameters,lambd)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y,parameters, caches)
        
        # Update parameters.
        
        #parameters = update_parameters(parameters, grads, learning_rate)
        parameters = update_parameters_regulation(parameters, grads, learning_rate,m,lambd)
                
        # Print the cost every 100 training example
        if print_cost and i % cost_per == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if(i%100==0):
            costs.append(cost)
    if  plot_cost:   
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    return AL,parameters

'''use for check backward propagation'''
#def dictionry_to_vector(para):
#    theta=para['W1'].reshape(-1,1)
#    theta=np.concatenate((theta,para['b1'].reshape(-1,1)),axis=0)
#    for l in range(2,4):
#        theta=np.concatenate((theta,para['W'+str(l)].reshape(-1,1)),axis=0)
#        theta=np.concatenate((theta,para['b'+str(l)].reshape(-1,1)),axis=0)
#    return theta
#def vector_to_dictionary(theta):
#    dic={}
#    dic['W1']=theta[:30].reshape(3,10)
#    dic['b1']=theta[30:33].reshape(3,1)
#    dic['W2']=theta[33:42].reshape(3,3)
#    dic['b2']=theta[42:45].reshape(3,1)
#    dic['W3']=theta[45:48].reshape(1,3)
#    dic['b3']=theta[48:49].reshape(1,1)
#    return dic
#def grads_to_vector(grads):
#    dtheta=grads['dW1'].reshape(-1,1)
#    dtheta=np.concatenate((dtheta,grads['db1'].reshape(-1,1)),axis=0)
#    for l in range(2,4):
#        dtheta=np.concatenate((dtheta,grads['dW'+str(l)].reshape(-1,1)),axis=0)
#        dtheta=np.concatenate((dtheta,grads['db'+str(l)].reshape(-1,1)),axis=0)
#    return dtheta
#
#def backward_check():
#    np.random.seed(1)
#    X=np.random.rand(10,10)*0.01
#    Y=np.ones((1,10))
#    Y[:,5:]=np.zeros((1,5))
#    
#    paras=initialize_parameters_deep([10,3,3,1])
#    theta=dictionry_to_vector(paras)
#    
#    #compute dP true value
#    AL,caches=L_model_forward(X,paras)
#    grads=L_model_backward(AL,Y,paras,caches)
#    
#    dtheta=grads_to_vector(grads)
#    
#    #compute dP in approximate value
#    e=0.0000001
#    num_values=theta.shape[0]
#
#    dtheta_approx=np.zeros(theta.shape)
#    for i in range(num_values):
#        theta_plus=theta.copy()
#        theta_minus=theta.copy()
#        
#        theta_plus[i,0]=theta_plus[i,0]+e
#        theta_minus[i,0]=theta_minus[i,0]-e
#        
#        para_plus=vector_to_dictionary(theta_plus)
#        para_minus=vector_to_dictionary(theta_minus)
#        
#        
#        
#        AL1,_=L_model_forward(X,para_plus)
#        AL2,_=L_model_forward(X,para_minus)
#        
#        
#        cost_plus=compute_cost(AL1,Y)
#        cost_minus=compute_cost(AL2,Y)
#        
#        dtheta_approx[i,0]=(cost_plus-cost_minus)/(2*e)
#        
#        print(i)
#        print('real')
#        print(dtheta[i,0])
#        print('approx')
#        print(dtheta_approx[i,0])
#    
#    numerator = np.linalg.norm(dtheta - dtheta_approx)                                           # Step 1'
#    denominator = np.linalg.norm(dtheta) + np.linalg.norm(dtheta_approx)                                         # Step 2'
#    difference = numerator / denominator
#    
#    if difference > 1e-7:
#        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
#    else:
#        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
#    
#    return difference
#backward_check()

def predict(X,parameters):
    A,_=L_model_forward(X, parameters)
    return A>=0.5
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
def print_result(X,predict):
    m=X.shape[1]
    for i in range(m):
        index = i
        plt.imshow(X[:,index].reshape((num_px, num_px, 3)))
        plt.show()
        print ("y = " + str(test_set_y[0,index]) +\
               ", you predicted that it is a \"" + \
               classes[int(predict[0][i])].decode("utf-8") +  "\" picture.")
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
def print_accuracy(Y,predict):
    m=Y.shape[1]
    accuracy=0
    for i in range(m):
        if(Y[0][i]==predict[0][i]):
            accuracy+=1
    accuracy=accuracy/m*100
    print(str(accuracy)+"%")




np.random.seed(3)
train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
num_m=train_set_x.shape[0]
num_px=64
layers_dims=[num_m,20,7,5,3,1]#20,7,5




READ_FILE=0
WRITE_FILE=1
if(READ_FILE==0):
    predict_train,parameters=L_Layer_Model(train_set_x,train_set_y,layers_dims,cost_per=100,\
        learning_rate = 0.0075, num_iterations = 4500, lambd=5,print_cost=True,plot_cost=True)
else:
    parameters=read_file()
if(WRITE_FILE==1):
    write_file(parameters)

predict_test=predict(test_set_x,parameters)
predict_train=predict(train_set_x,parameters)

print_accuracy(train_set_y,predict_train)
print_accuracy(test_set_y,predict_test)



my_set_x=get_my_picture()
my_prediction=predict(my_set_x/255,parameters)

PRINT=1
if(PRINT==0):
    print_result(my_set_x,my_prediction)
elif(PRINT==1):
    print_result(test_set_x,predict_test)


