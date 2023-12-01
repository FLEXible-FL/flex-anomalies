from models import AutoEncoder, DeepCNN_LSTM,IsolationForest, ClusterAnomaly,PCA_Anomaly
from utils.metrics import *
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import scipy.io
import numpy as np


def load_data_ODDS(file_path):  # data from https://odds.cs.stonybrook.edu/ (.mat)
    mat = scipy.io.loadmat(file_path) 
    #print(mat.keys())
    data = {'X':mat['X'].tolist(),'y':mat['y'].tolist()}
   
    df = pd.DataFrame.from_dict(data)
    df = df.sample(frac=1).reset_index(drop=True) 
    return df

def split_data(X,y,split_size = 0.30):
    # Generate train-test splits
    return train_test_split(X,y, test_size = split_size, random_state=42)
    

def model_AutoEncoder(input_dim,epochs,batch_size,neurons,hidden_act):
      
    checkpoint_filepath = 'tmp/checkpoint'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_filepath,
                                save_weights_only=False,
                                monitor='val_loss',
                                 mode='min',
                                save_best_only=True)

   
    model = AutoEncoder(input_dim= input_dim,
                        neurons= neurons,hidden_act= hidden_act,
                        callbacks=[model_checkpoint_callback],
                        epochs= epochs, batch_size= batch_size
                        )
    
                 
    return model 



def model_IForest():
    model = IsolationForest()
    return model


def model_Cluster(n_clusters):
     model = ClusterAnomaly(n_clusters = n_clusters)
     return model

    

def model_PCA(n_components):
    model = PCA_Anomaly(n_components = n_components)
    return model 

def model_CNN_LSTM():
    pass

    

if __name__ == '__main__':
    #model_name = input("please enter a list with the name of models (AutoEncoder, IForest, Cluster, CNN_LSTM, PCA)")
    data_path ="datasets/shuttle.mat"

    df = load_data_ODDS(data_path)    # shape = (n_samples,2), n_samples = samples number  and 2: X(list of attributes by sample), y (sample label)
    X = np.array(df['X'].tolist())
    y = np.array(df['y'].tolist())
    split_size = 0.30   
    X_train, X_test, y_train, y_test = split_data(X,y,split_size)
    # define parameters 
    EPOCHS = 100
    BATCH_SIZE = 32     
    input_dim = X.shape[1]
    n_clusters = 4
    n_components = 4

    f = []
    c_model1 = model_AutoEncoder(input_dim, EPOCHS,BATCH_SIZE,neurons=[8,4,8],hidden_act= ["relu","relu","relu"])
    c_model1.fit(X_train,y_train)
    c_model1.evaluate(X_test,y_test)  
    f = f + [["AutoEncoder"] + c_model1.result_metrics_]
    
    c_model2 = model_IForest()  
    c_model2.fit(X_train,y_train)
    c_model2.evaluate(X_test,y_test) 
    f = f + [["IForest"] + c_model2.result_metrics_]
    
    c_model3 = model_Cluster(n_clusters)  
    c_model3.fit(X_train,y_train)
    c_model3.evaluate(X_test,y_test)  
    f = f + [["Cluster"] + c_model3.result_metrics_]
    
    c_model4 = model_PCA(n_components)  
    c_model4.fit(X_train,y_train)
    c_model4.evaluate(X_test,y_test)  
    f = f + [["PCA"] + c_model4.result_metrics_]
    
    #c_model5 = model_CNN_LSTM()
    
    df = pd.DataFrame(f,columns = [" Models ", " Accuracy ", " Precision ", " F1 ", " Recall ", " AUC_ROC "])
    print(df) 
    
   
 
    # print(c_model1.labels_[:10])  
    # print(c_model1.d_scores_[:10])   
    # print(c_model1.decision_function(X_train)[:10])
    
    