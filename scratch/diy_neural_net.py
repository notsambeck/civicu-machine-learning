'''
use height and weight as X1, X2 to predict gender y
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../lessons/shared-resources/heights_weights_genders.csv')

df['Gender'] = (df.Gender == 'Male').astype(int)

W = np.random.randn(2).T


def predict_gender(sample):
    height, weight = df[['Height', 'Weight']].iloc[sample]
    x = np.array([height, weight])
    logistic = np.dot(x, W)
    y_pred = sigmoid(logistic)
    return log_loss(y_pred)

def log_loss(y_prob, y):
    return y*np.l
