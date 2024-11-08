import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def data_preprocessing(data,percentage):

    

    u = data.iloc[:,:10]
    k = data.iloc[:,-1]
    # Generate noise for each column
    noise = np.random.normal(0, u.std(), u.shape) * percentage
    # Add noise to the DataFrame
    u_noisy = u + noise
    data_noisy = pd.concat([u_noisy, k], axis=1)
    # Create a StandardScaler instance
    scaler = StandardScaler()
    # Normalize the DataFrame and create a new DataFrame
    data_noisy = pd.DataFrame(scaler.fit_transform(data_noisy), columns= data_noisy.columns)
    data = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)
    #splitting data set
    u_tr_n, u_te_n, k_tr, k_te = train_test_split(data_noisy.iloc[:,:-1], data_noisy.iloc[:,-1], test_size=0.2, random_state=42)
    u_tr,u_te = train_test_split(data.iloc[:,:-1],test_size=0.2,random_state=42)
    
    # shifting the orginal values by 25% fro prior mean
    shift_train = k_tr*0.25
    shift_test = k_te*0.25
    k_tr_pr = k_tr + shift_train
    k_te_pr = k_te + shift_test
    
    return u_tr_n, u_te_n, u_tr, u_te, k_tr, k_te, k_tr_pr, k_te_pr
