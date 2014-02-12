import numpy as np
import pandas as pd

def readMNISTdata(type):
    
    data = np.matrix()
    princomps = getPCA()
    
    digits = range(10)
    for digit in digits:
	digit_data = pd.read_csv(type + str(digit) + '.csv', header=None)


    return data

def getPCA():
    princomps = pd.read_csv('Q.csv', header=None)
    return princomps



