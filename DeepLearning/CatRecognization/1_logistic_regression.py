import numpy as np
import matplotlib.pyplot as plt
import h5py

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
def initialize_with_zeros(dim):
    return np.zeros((dim,1)),0
def propagate(w,b,X,Y):
    m=Y.shape[1]
    #forward propagation
    Z=np.matmul(w.T,X)+b
    A=sigmoid(Z)
    #cost
    cost=-1*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m
    #backward propagation
    dw=np.matmul(X,(A-Y).T)/m
    db=np.sum(A-Y)/m
    grads={"dw":dw,"db":db}
    return grads,cost
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads,cost=propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # update rule (≈ 2 lines of code)
        w=w-dw*learning_rate
        b=b-db*learning_rate
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,"b": b}
    grads = {"dw": dw,"db": db}
    
    return params, grads, costs
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    assert(w.shape[0]==X.shape[0])
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A=sigmoid(np.matmul(w.T,X)+b)
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if(A[0][i]>0.4):
            Y_prediction[0][i]=1
        else:
            Y_prediction[0][i]=0
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w,b=initialize_with_zeros(X_train.shape[0])
    
    params,grads,costs=optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w=params['w']
    b=params['b']
    
    Y_prediction_train=predict(w,b,X_train)
    Y_prediction_test=predict(w,b,X_test)
    
    # Print train/test Errors
    train_accuracy=1 - np.mean(np.abs(Y_prediction_train - Y_train))
    test_accuracy=1 - np.mean(np.abs(Y_prediction_test - Y_test))
    print("train accuracy: {} %".format(train_accuracy* 100))
    print("test accuracy: {} %".format(test_accuracy * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations,
         "train_accuracy":train_accuracy,
         "test_accuracy":test_accuracy}
    
    return d


train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
num_px=64
#一次模型
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=200, learning_rate = 0.005, print_cost = True)


##研究iteration次数
#iteration=np.arange(1,2000,step=100)
#accuracy=[]
#for i in iteration:
#    num_iterations=int(i)
#    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate = 0.005, print_cost = False)
#    accuracy.append(d['test_accuracy'])
#
#plt.plot(np.array(iteration),np.array(accuracy))
#
#plt.ylabel('test_accuracy')
#plt.xlabel('num_iteration')
#plt.legend()
#plt.show()



#输出test图片和预测
#print(test_m)
#for i in range(0,20):
#    index = i
#    plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
#    plt.show()
#    print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")
##输出训练图片
#for i in range(0,50):
#    index = i
#    plt.imshow(train_set_x[:,index].reshape((num_px, num_px, 3)))
#    plt.show()
#    print ("y = " + str(train_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_train"][0,index])].decode("utf-8") +  "\" picture.")


##使用自己的图片
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

my_predict_y=predict(d['w'],d['b'],my_x/255)

my_predict_y=np.ravel(my_predict_y)
for i in range(len(collection)):
    io.imshow(collection[i])
    plt.show()
    print ("you predicted that it is a \"" + classes[int(my_predict_y[i])].decode("utf-8") +  "\" picture.")
#    