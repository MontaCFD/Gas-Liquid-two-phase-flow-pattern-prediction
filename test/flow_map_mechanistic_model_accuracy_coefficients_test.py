import numpy as np 
import joblib
import pandas as pd
from tabulate import tabulate
import sys, os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Ignore the UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    
sys.path.insert(0, '../src')
from FP_prediction import *
from thermodynamic_properties import *

# input parameter settings
best = True    # best configuration 
#best = False   # standard configuration
if (best):
    print('best configuration')
else:
    print('standart configuration')
p    = 3      # 0: Shoham, 1:Suni, 2: Suni, 3:Robe, 4:Meng

# Get the path to the venv directory
venv_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the path to the data directory
data_dir = os.path.join(venv_dir, 'data')
# Data sources
file_names = ['validation_horizontal_pipe_1986_shoham.csv',
              'validation_horizontal_pipe_1986_shoham-d1.csv', 
              'validation_horizontal_pipe_1986_shoham-d2.csv',
              'validation_horizontal_pipe_1987_Suni.csv',
              'validation_horizontal_pipe_1987_Suni-d1.csv',
              'validation_horizontal_pipe_1987_Suni-d2.csv', 
              'validation_horizontal_pipe_1997_Robe.csv', 
              'validation_horizontal_pipe_1999_Meng.csv']

# coefficients and factors 
# source doi:10.1016/j.cherd.2011.08.009
# S-NS: lamda = 0.25  S is predicted to 100 % and lamda = 7.0   NS is predicted to 100 %
# standard scenario 
# B-NB: alpha factor 
# Annular- Non annular hL/d between 0.35 and 0.5 
standard_configuration_list = [(1.0, 0.35, 1) for _ in file_names]
# best scenario to reach the best overall accuracy compared with experimtental data points :) 
#best_configuration_list     = [(0.53,0.42,1.5), (0.71,0.38,1.33), (0.25, 0.42, 1.0), (0.25, 0.47, 0.67), (1.54, 0.33,0.5), (0.71, 0.25, 0.5)]
#best_configuration_list     = [(0.71,0.42,1.5), (0.72,0.42,1.5), (0.25, 0.42, 1.0), (0.25, 0.47, 0.67), (1.54, 0.33,0.5), (0.71, 0.25, 0.5)]
# aus Shoham and Suni
best_configuration_list     = [(0.71,0.42,1.5), (0.71,0.42,1.5), (0.72,0.42,1.5), (0.25, 0.47, 0.83), (0.25, 0.42, 1.0),(0.25, 0.42, 1.0), (1.54, 0.33,0.5), (0.71, 0.25, 0.5)]
best_configuration_list     = [(0.71,0.42,1.5), (0.71,0.42,1.5), (0.72,0.42,1.5), (0.25, 0.47, 0.83), (0.25, 0.47, 0.83),(0.25, 0.47, 0.83), (1.54, 0.33,0.5), (0.71, 0.25, 0.5)]

# Create a subdirectory called "FlowMap" if it doesn't exist
if not os.path.exists("FlowMap"):
    os.makedirs("FlowMap")
    
#read_file    = file_names[p]
read_file = os.path.join(data_dir, file_names[p])


if (best):
    lamda, beta, gamma = best_configuration_list[p]
else:
    lamda, beta, gamma = standard_configuration_list[p]

# Load the data from the CSV file
df = pd.read_csv(read_file)
# Initialize counters for correct predictions and instances of each flow pattern
correct_SS = 0
correct_SW = 0
correct_A = 0
correct_I = 0
correct_DB = 0
total_correct = 0

instances_SS = (df['Flow_label'] == 0).sum()
instances_SW = (df['Flow_label'] == 1).sum()
instances_A = (df['Flow_label'] == 2).sum()
instances_I = (df['Flow_label'] == 3).sum()
instances_DB = (df['Flow_label'] == 4).sum()

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Extract the input values
    d = row['ID']
    A = pi / 4. * d ** 2.
    theta = row['Ang']
    U_sl = row['Vsl']
    U_sg = row['Vsg']
    rho_l = row['DenL']
    rho_g = row['DenG']
    mu_l = row['VisL']
    mu_g = row['VisG']
    
    # Calculate the mass flow rates and void fraction
    m_g = U_sg * A * rho_g
    m_l = U_sl * A * rho_l
    x = m_g / (m_g + m_l)
    s = sqrt(1.0 - x * (1.0 - rho_l / rho_g))
    epsilon = 1.0 / (1.0 + s * (1.0 - x) / x * (rho_g / rho_l))
    
    # Predict the flow pattern using the FPT_Horizontal_Pipe_test function
    ptt, FP, Fr_sl, Fr_sg = FPT_Horizontal_Pipe_test(U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, d, theta, lamda, gamma, beta)
    
    # Check if the prediction matches the actual flow pattern
    if ptt == row['Flow_label']:
        total_correct += 1
        if row['Flow_label'] == 0:
            correct_SS += 1
        elif row['Flow_label'] == 1:
            correct_SW += 1
        elif row['Flow_label'] == 2:
            correct_A += 1
        elif row['Flow_label'] == 3:
            correct_I += 1
        elif row['Flow_label'] == 4:
            correct_DB += 1

# Calculate the accuracy scores
accuracy_SS = correct_SS / instances_SS * 100
accuracy_SW = correct_SW / instances_SW * 100
accuracy_A = correct_A / instances_A * 100
accuracy_I = correct_I / instances_I * 100
accuracy_DB = correct_DB / instances_DB * 100
overall_accuracy = total_correct / len(df) * 100

# Create a table of accuracy scores
table = [
    ['Flow Pattern', 'Instances', 'Correct Predictions', 'Accuracy (%)'],
    ['SS', instances_SS, correct_SS, f"{accuracy_SS:.2f}"],
    ['SW', instances_SW, correct_SW, f"{accuracy_SW:.2f}"],
    ['A', instances_A, correct_A, f"{accuracy_A:.2f}"],
    ['I', instances_I, correct_I, f"{accuracy_I:.2f}"],
    ['DB', instances_DB, correct_DB, f"{accuracy_DB:.2f}"],
    ['Overall', len(df), total_correct, f"{overall_accuracy:.2f}"]
]

# Print the table
print(tabulate(table, headers='firstrow', tablefmt='grid'))