#! /usr/bin/env python3
'''===============================================================================================================
 - Author: M. Guesmi
 - Adress: Chair of process engineering TU Dresden, Helmholtzstraße 10, 01069 Dresden, Germany
 - Summary: organize experimental from different sources and prepare it for machine learning model
=================================================================================================================='''

import pandas as pd
import os, sys


# print current directory
print(os.getcwd())

# Order: Author, P, T, Type of liquid, Type of Gas, DenL, DenG, VisL, VisG, ST, ID, Roughness, Ang, L/D, Vsl, Vsg, Flow_label
# to remove: P, T, Type of liquid, Type of Gas, Roughness, L/D
# after remove: Author, DenL, DenG, VisL, VisG, ST, ID, Ang, Vsl, Vsg, Flow_label
# desired order: Author, Vsl, Vsg, DenL, DenG, VisL, VisG, ST, ID, Ang,  Flow_label
# Read the CSV file
df1 = pd.read_csv('dataset.csv')
# Define the interval of inclination  [a, b]
a = -10. 
b =  10.
# List of columns to remove
columns_to_remove = ['P' , 'T', 'Type of liquid', 'Type of Gas', 'Roughness', 'L/D']

# Drop the specified columns
df1 = df1.drop(columns=columns_to_remove)

# Define the desired column order
desired_order = ['Author', 'Vsl', 'Vsg', 'DenL', 'DenG', 'VisL', 'VisG', 'ST', 'ID', 'Ang',  'Flow_label']

# Reorder the columns
df1 = df1[desired_order]
# now filter data --> Interesting for us is the slightly inclined upward or downward

print(f'Angle of inclination: [{a}°, +{b}°]')
# Filter out rows where the value in ColumnX is outside the interval [a, b]
df1 = df1[(df1['Ang'] >= a) & (df1['Ang'] <= b)]

# evaluate number of classes 
# Define a list of flow patterns
target_values = ['SS', 'SW', 'A', 'I', 'DB', 'B']

# Initialize an empty list to store frequencies
frequencies = []

# Calculate the frequency for each target value
for target_value in target_values:
    frequency = df1['Flow_label'].value_counts().get(target_value, 0)
    frequencies.append((target_value, frequency))

# Write the frequencies to a text file
with open('frequencies_flow_labels.txt', 'w') as FP_frequency_file:
    for target_value, frequency in frequencies:
        FP_frequency_file.write(f"Value: {target_value}, Frequency: {frequency}\n")
        print(f"Value: {target_value}, Frequency: {frequency}\n")
FP_frequency_file.close()

# create table with data repartition: Origin - Labels 
# Write the frequencies to a text file
with open('origin-labels-frequency.txt', 'w') as FP_frequency_file:
    for target_value, frequency in frequencies:
        FP_frequency_file.write(f"Value: {target_value}, Frequency: {frequency}\n")
        print(f"Value: {target_value}, Frequency: {frequency}\n")
FP_frequency_file.close()
# Save the modified DataFrame to a new CSV file (optional)
df1.to_csv('desired_dataset.csv', index=False)
df1.to_csv('desired_dataset.txt', index=False)

# Get the Size of Each File and print it in output file
output_file = open('file_sizes.txt', 'w')

# organizing data depending on it's origin 
# Define the directory path
# Define the directory name
directory_name = 'data_origins'
print(os.getcwd())
# Create the directory in the current working folder
directory_path = os.path.join(os.getcwd(), directory_name)
# Check if the directory already exists
if not os.path.exists(directory_path):
    # Create the directory
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")
else:
    print(f"Directory '{directory_path}' already exists.")
# Load the CSV file
df2 = df1 
#df2 = pd.read_csv('desired_dataset.csv')
# Get unique values from the first column
unique_values = df2.iloc[:, 0].unique()

# Iterate through unique values and split data
total_points = 0.
for value in unique_values:
    # Create a new DataFrame containing only rows with the current value
    subset = df2[df2.iloc[:, 0] == value]
    
    # Save the subset to a new CSV file named after the value
    file_name = f'{directory_path}/{value}.csv'
    subset.to_csv(file_name, index=False)
    # find values distribution in each data set 
    # Calculate the frequency for each target value
    # Initialize an empty list to store frequencies
    frequencies_subset = []
    for target_value in target_values:
        frequency = subset['Flow_label'].value_counts().get(target_value, 0)
        frequencies_subset.append((target_value, frequency))
    # Write the frequencies to a text file
    with open(f'{directory_path}/{value}-Imbalance.txt', 'w') as FP_frequency_file:
        print(f'{value} \n')
        for target_value, frequency in frequencies_subset:
            FP_frequency_file.write(f"Value: {target_value}, Frequency: {frequency}\n")
            print(f"Value: {target_value}, Frequency: {frequency}\n")
    FP_frequency_file.close()
    

    with open(file_name) as f:
        num_lines = sum(1 for line in f) - 1 # -1 because of the head line
        total_points = total_points + num_lines
        output_file.write(f'{file_name} performed {num_lines}  experimental data points\n')

# print number of total points 
output_file.write(f'total number of investigated data points {total_points} \n')        
# close output file
output_file.close()



# Step 1: Read the CSV file
df = pd.read_csv('desired_dataset.csv') # dataset with desired inclination angle range [-10°, +10°]

# Step 2:
#  Extract features and patterns
patterns = df['Flow_label']
sources = df['Author']

column_name = 'Flow_label'  # Replace with the actual column name

# Define your mapping of strings to numbers
string_to_number = {
    'SS': 0,
    'SW': 1,
    'A' : 2,
    'I' : 3,
    'DB': 4,
    'B' : 4
}

# Loop through the column and replace strings with numbers
for idx, value in enumerate(df[column_name]):
    if value in string_to_number:
        df.at[idx, column_name] = string_to_number[value]

# save desired dataset with flow label replaced as numbers
df.to_csv('desired_dataset_inclined_pipes.csv', index=False)
# Filter sources to be kept vor the validation 
# sources 
# > 
source =['1982_Ovad','1985_Piu-','1986_Koub', '1987_Suni', '1997_N. V','1997_Robe', '1999_Meng', '2001_Mana', '2001_Mata', '2003_ Pla', '2003_Abdu']
# 1986_Koub, 1997_Robe and 1999_Meng are used to validate the models 
s1 = 3
s2 = 5
s3 = 6

# Define the sources to exclude
excluded_sources = [source[s1], source[s2], source[s3]]  # List of sources to exclude
#excluded_sources = [] 
# Create a condition for exclusion
condition = df['Author'].isin(excluded_sources)

# Create a DataFrame with the excluded data
excluded_data = df[condition]

# remaining data for training and validation
#df = df[df['Author'] != source[s1]]
#df = df[df['Author'] != source[s2]]
#df = df[df['Author'] != source[s3]]
training_test_data = df[~condition]


# Create the directory in the current working folder
subfolder      = 'Horizontal-slightly-inclined-pipes'
subfolder_path = os.path.join(os.getcwd(), subfolder)
# Check if the directory already exists
if not os.path.exists(subfolder_path):
    # Create the directory
    os.makedirs(subfolder_path)
    print(f"Directory '{subfolder_path}' created successfully.")
else:
    print(f"Directory '{subfolder_path}' already exists.")

# Change to the subfolder
os.chdir(subfolder_path)
# Save the training, test and validation data
excluded_data.to_csv('validation_data.csv', index=False)    
training_test_data.to_csv('training_test_data.csv', index=False)

# Count frequency of each pattern for each source
df_remaining   = training_test_data
pattern_counts = df_remaining.groupby(['Author', 'Flow_label']).size().reset_index(name='Frequency')

# Organize data into a table
pattern_table = pattern_counts.pivot_table(index='Author', columns='Flow_label', values='Frequency', fill_value=0)

# Step 5: Add a row at the bottom with the sum of each pattern's frequencies
pattern_table.loc['Total'] = pattern_table.sum()

# output to file: resulting table 
output = open('resulting-table.txt','w')
# Print the resulting table
print(pattern_table)
output.write(f'{pattern_table}\n')
output.write(f'--------------------------------------\n')
# total number of experimental data points 
number_of_points = len(df_remaining['Flow_label'])
print(f'total number of investigated data points for the range [{a}°, {b}°] is: {number_of_points}\n')
output.write(f'total number of investigated data points for the range [{a}°, {b}°] is: {number_of_points}\n')
output.close()