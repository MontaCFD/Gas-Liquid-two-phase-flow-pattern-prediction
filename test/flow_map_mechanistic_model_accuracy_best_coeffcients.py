import numpy as np 
import joblib
import pandas as pd
from tabulate import tabulate
import sys, os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
from itertools import product

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Ignore the UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    
import numpy as np 
from math import sqrt, pi, radians, sin, cos, tan, log, exp, pow
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
import joblib
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Ignore the UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, '../src')
from FP_prediction import *
from thermodynamic_properties import *




# Define the ranges for each parameter
lamda_values = np.linspace(0.25, 2.0, 20)  # 5 points between 0.25 and 7.0
gamma_values = np.linspace(0.5, 2.0, 10)  # 5 points between 0.25 and 1.0
beta_values = np.linspace(0.25, 0.5, 10)  # 5 points between 0.35 and 0.5


# Create a list of all possible combinations of values
param_combinations = list(product(lamda_values, gamma_values, beta_values))


# Initialize variables to store the best combination and its corresponding accuracy for each flow pattern
best_accuracy = {'SS': 0, 'SW': 0, 'A': 0, 'I': 0, 'DB': 0}
best_params = {'SS': {}, 'SW': {}, 'A': {}, 'I': {}, 'DB': {}}

# Initialize variables to store the best overall combination and its corresponding overall accuracy
best_overall_accuracy = 0
best_overall_params = {}


# Get the path to the notebooks directory
notebooks_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to the project root by going up one level from notebooks
project_root_dir = os.path.dirname(notebooks_dir)

# Construct the path to the data directory
data_dir = os.path.join(project_root_dir, 'data')

# File names
file_names = ['validation_horizontal_pipe_1986_shoham.csv',
              'validation_horizontal_pipe_1986_shoham-d1.csv', 
              'validation_horizontal_pipe_1986_shoham-d2.csv',
              'validation_horizontal_pipe_1987_Suni.csv',
              'validation_horizontal_pipe_1987_Suni-d1.csv',
              'validation_horizontal_pipe_1987_Suni-d2.csv', 
              'validation_horizontal_pipe_1997_Robe.csv', 
              'validation_horizontal_pipe_1999_Meng.csv']

# Assume read_file and model_trained are chosen
p = 3
#read_file    = file_names[p]
read_file = os.path.join(data_dir, file_names[p])
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

for lamda, gamma, beta in param_combinations:
    # Iterate through each row in the DataFrame
    # reset count 
    total_correct = 0
    correct_SS    = 0
    correct_SW    = 0
    correct_A     = 0
    correct_I     = 0
    correct_DB    = 0
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

    # Update best accuracy and parameters if current accuracy is better
    if accuracy_SS > best_accuracy['SS']:
        best_accuracy['SS'] = accuracy_SS
        best_params['SS'] = {'lamda': lamda, 'gamma': gamma, 'beta': beta}
    
    if accuracy_SW > best_accuracy['SW']:
        best_accuracy['SW'] = accuracy_SW
        best_params['SW'] = {'lamda': lamda, 'gamma': gamma, 'beta': beta}
    
    if accuracy_A > best_accuracy['A']:
        best_accuracy['A'] = accuracy_A
        best_params['A'] = {'lamda': lamda, 'gamma': gamma, 'beta': beta}
    
    if accuracy_I > best_accuracy['I']:
        best_accuracy['I'] = accuracy_I
        best_params['I'] = {'lamda': lamda, 'gamma': gamma, 'beta': beta}
    
    if accuracy_DB > best_accuracy['DB']:
        best_accuracy['DB'] = accuracy_DB
        best_params['DB'] = {'lamda': lamda, 'gamma': gamma, 'beta': beta}

    # Update best overall accuracy and parameters if current overall accuracy is better
    if (accuracy_SW > 70.0 and overall_accuracy > best_overall_accuracy) or instances_SW ==0 and overall_accuracy > best_overall_accuracy:
        best_overall_accuracy = overall_accuracy
        best_overall_params = {'lamda': lamda, 'gamma': gamma, 'beta': beta}

# Output best accuracy and parameters for each flow pattern
for flow_pattern, accuracy in best_accuracy.items():
    print(f"Best accuracy for {flow_pattern}: {accuracy}")
    print(f"Best parameters for {flow_pattern}: {best_params[flow_pattern]}")

# Output best overall accuracy and parameters
print(f"Best overall accuracy: {best_overall_accuracy}")
print(f"Best overall parameters: {best_overall_params}")

# Create a table of accuracy scores
table = [
    ['Flow Pattern', 'Best Accuracy (%)', 'Best Parameters'],
    ['SS', best_accuracy['SS'], best_params['SS']],
    ['SW', best_accuracy['SW'], best_params['SW']],
    ['A', best_accuracy['A'], best_params['A']],
    ['I', best_accuracy['I'], best_params['I']],
    ['DB', best_accuracy['DB'], best_params['DB']],
    ['Overall', best_overall_accuracy, best_overall_params]
]

# Print the table
print(tabulate(table, headers='firstrow', tablefmt='grid'))
