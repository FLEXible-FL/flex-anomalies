from models import AutoEncoder, DeepCNN_LSTM,IsolationForest, ClusterAnomaly
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_data():
    data = arff.loadarff("./datasets/Mammography.arff")   
    df = pd.DataFrame(data[0])
    df = df.sample(frac=1).reset_index(drop=True)
    return df 

def split_data():
    data = load_data()
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    y = y.apply(lambda x: 0 if x == b'-1' else 1)  # 1 = outliers, 0 = inliers
    
    # Generate train-test splits
    return train_test_split(X,y, test_size=0.30, random_state=42)
    

def define_model_AutoEncoder(X):
    EPOCHS = 100
    BATCH_SIZE = 32
    FEATURE_DIMENSION = X.shape[1]

    checkpoint_filepath = 'tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_filepath,
                                save_weights_only=False,
                                monitor='val_loss',
                                 mode='min',
                                save_best_only=True)

   
    model = AutoEncoder(input_dim= FEATURE_DIMENSION,
                        neurons=[8,4,8],hidden_act=["relu","relu","relu"],
                        callbacks=[model_checkpoint_callback],
                        epochs=EPOCHS
                        )
    return model 

def run_AutoEncoder():
    X_train, X_test, y_train, y_test = split_data()
    
    # define model
    c_model = define_model_AutoEncoder(X_train)  
   
    c_model.fit(X_train.to_numpy(),X_train.to_numpy())
    print(c_model.history_)
    c_model.evaluate(X_test.to_numpy(),y_test.to_numpy())

def run_IForest():
    X_train, X_test, y_train, y_test = split_data()
    c_model = IsolationForest()
    c_model.fit(X_train.to_numpy(),y_train.to_numpy())
    c_model.evaluate(X_test.to_numpy(),y_test.to_numpy())   

    print(c_model.labels_[:10])  
    print(c_model.d_scores_[:10])   
    print(c_model.decision_function(X_train)[:10])


def run_Cluster():
    X_train, X_test, y_train, y_test = split_data()
    c_model = ClusterAnomaly(n_clusters = 4)
    c_model.fit(X_train.to_numpy(),y_train.to_numpy())
    c_model.evaluate(X_test.to_numpy(),y_test.to_numpy())   

    print(c_model.labels_[:10]) 
    print(c_model.d_scores_[:10])
    print(c_model.decision_function(X_train)[:10])


if __name__ == '__main__':
    #run_AutoEncoder()
    #run_IForest()
    run_Cluster()