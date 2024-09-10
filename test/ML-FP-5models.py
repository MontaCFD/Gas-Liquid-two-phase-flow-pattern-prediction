#! /usr/bin/env python3
'''===============================================================================================================
 - Author: M. Guesmi
 - Adress: Chair of process engineering TU Dresden, George-Bähr-Straße 3b, 01069 Dresden, Germany
 - Summary: test different machine learning models to classfiy flow pattern based on operational and geometrical features 
=================================================================================================================='''
#!python -m pip install seaborn
#!python -m pip install xgboost
#!python -m pip install install --upgrade scikit-learn imbalanced-learn
# 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# ML-Models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Evaluation metrics
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
# 
from sklearn.preprocessing import LabelBinarizer
import pickle
# Output, Display
from tabulate import tabulate
from IPython.display import display, HTML
# import joblib to import model to file
import joblib
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
# Ignore the UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


# Ignore all warnings
#warnings.filterwarnings('ignore') 
df = pd.read_csv('training_test_data.csv',sep=',')

df.describe()
df
# Calculate the correlation between each feature and the Flow pattern
correlation_df = pd.DataFrame(columns=['Feature', 'Correlation with FlowPattern'])

for feature in df.drop(columns=['Author', 'ST','Flow_label']).columns:
    #correlation    = df[feature].corr(df['Flow_label'])
    correlation = df[feature].corr(df['Flow_label'])
    print(f"Correlation between {feature} and Flow Pattern: {correlation}")
    correlation_df = pd.concat([correlation_df, pd.DataFrame({'Feature': [feature], 'Correlation with FlowPattern': [correlation]})], ignore_index=True)

# Generate a bar plot to visualize the correlation with the Flow pattern
plt.figure(figsize=(10, 6))
sns.barplot(x='Correlation with FlowPattern', y='Feature', data=correlation_df, palette='coolwarm')
plt.title('Correlation with FlowPattern Variable')
plt.show()
# Split the dataset into train and test sets
# Features
X = df.drop(columns=['Author', 'ST','Flow_label'], axis=1)    

# Target variable
y = df['Flow_label']                          
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = pd.concat([X_train, y_train], axis=1)
print('everything is fine sofar!')
# Imbalanced data --> use different techniques 
# Using different classifiers 
# Initialize different classifiers
classifiers = {
    'XGBClassifier': XGBClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=20000),
    'Support Vector Machine': SVC(max_iter=20000)
}
# Generate y_pred for each classifier
y_preds = {}
# Initialize a list to store the results of accuracy
results = []
models_accuracies=[]
# Train and evaluate each classifier
for clf_name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    y_preds[clf_name] = y_pred

    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred) * 100  # Multiply by 100 for percentage
    results.append([clf_name, f"{accuracy:.2f}%"])
    models_accuracies.append([clf_name, accuracy])

# Print results as a table
print('Mutliclass accuracy:')
print(tabulate(results, headers=["Model", "Accuracy"], tablefmt="pretty"))
# Initialize a KFold object with k=5 (5-fold cross-validation)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize a list to store results
results = []

# Perform k-fold cross-validation for each classifier
for clf_name, clf in classifiers.items():
    cv_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

        # Train the model on the training data
        clf.fit(X_train_cv, y_train_cv)

        # Make predictions on the test data
        y_pred_cv = clf.predict(X_test_cv)

        # Calculate accuracy and store the score
        accuracy = accuracy_score(y_test_cv, y_pred_cv)
        cv_scores.append(accuracy)
    
    # Calculate average cross-validation score
    average_score = sum(cv_scores) / len(cv_scores)
    
    # Store the result
    results.append([clf_name, average_score])


# Print the results as a table
print('Cross validation:')
print(tabulate(results, headers=['Classifier', 'Average Score'], tablefmt='pretty'))

# Define original labels and desired labels
original_labels = ['0', '1', '2', '3', '4']
desired_labels = ['SS', 'SW', 'A', 'I', 'DB']
#desired_labels = ['Smooth-Stratified', 'Wavy-Stratified', 'Annular', 'Slug', 'Bubbly']
# Create a dictionary to map original labels to desired labels
label_mapping = {original_labels[i]: desired_labels[i] for i in range(len(original_labels))}



# Initialize a dictionary to store confusion matrices
confusion_matrices = {}

for name, y_pred in y_preds.items():
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm

print('Confusion Matrix:')
# Plot the confusion matrices
for name, cm in confusion_matrices.items():
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=desired_labels, yticklabels=desired_labels) # Reverse yticklabels
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    # Create the confusion matrix using scikit-learn
    cm = confusion_matrix(y_test, y_pred)
    # Calculate row sums (total predictions for each class)
    row_sums = np.sum(cm, axis=1)
    # Calculate percentages
    percentages = cm / row_sums[:, np.newaxis] * 100  # Ensure division is element-wise
    #sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=desired_labels, yticklabels=desired_labels) # Reverse yticklabels

   
# Initialize a dictionary to store classification reports
classification_reports = {}
tables = []
for clf_name, y_pred in y_preds.items():
    # Generate the classification report
    report = classification_report(y_test, y_pred, target_names=desired_labels, zero_division=1, output_dict=True)
    
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # Convert the report dictionary to a table
    table = []
    for label, metrics in report.items():
        if isinstance(metrics, float):
            precision = recall = f1_score = support = metrics
        else:
            precision = metrics.get('precision', '')
            recall = metrics.get('recall', '')
            f1_score = metrics.get('f1-score', '')
            support = metrics.get('support', '')
        
        table.append([label, precision, recall, f1_score, support])
        
    # Add a header to the table
    headers = ["Label", "Precision", "Recall", "F1-Score", "Support"]    
    # Add classifier name and kappa score as title
    tables.append((clf_name, kappa, table, headers))
    
    
    

# Print the tables
for classifier, kappa, table, headers in tables:
    print(f"Classifier: {classifier}, Cohen's Kappa: {kappa}")
    print(tabulate(table, headers, tablefmt="pretty"))

    print("\n")
# Get class probabilities for the input data
model = classifiers['Decision Tree']
model = classifiers['XGBClassifier']
model = classifiers['Random Forest']
# Get class probabilities for the input data
predicted_probabilities = model.predict_proba(X_test)
# Display long output with scrolling
display(HTML("<div style='max-height: 300px; overflow-y: auto;'>"  + "</div>"))
print(model)
# Get the index of the maximum probability for each prediction
max_prob_indices = np.argmax(predicted_probabilities, axis=1)

# Get the corresponding patterns
corresponding_patterns = [y_pred[index] for index in max_prob_indices]

# Get the maximum probabilities
max_probabilities = [predicted_probabilities[i, index] for i, index in enumerate(max_prob_indices)]

# Print the list of maximum probabilities and their corresponding patterns
for pattern, probability in zip(corresponding_patterns, max_probabilities):
    print(f"Pattern: {pattern}, Max Probability: {probability:.4f}")
# list of models and their corresponding accuracies: models_accuracies
# Filter models with accuracy > 80%
selected_models = [model_info for model_info in models_accuracies if model_info[1] > 85]

# Print the selected models
for model_info in selected_models:
    print(f"Model: {model_info[0]}, Accuracy: {model_info[1]:.2f}%")

# Initialize a dictionary to store models with accuracy > 90%
high_accuracy_models = {}


# Iterate through the models
for model_name, model in classifiers.items():
    for model_info in selected_models:
        if (model_info[0] == model_name):
            high_accuracy_models[model_name] = model


# the selected models are already trained these models
# Load the new data set
validation_data = pd.read_csv('validation_data.csv')  
# Features
X_valid = df.drop(columns=['Author', 'ST', 'Flow_label'], axis=1)    

# Target variable
y_valid = df['Flow_label'] 

# Initialize a dictionary to store predictions
predictions_dict = {}

# Loop through the selected models
for model_name, model in high_accuracy_models.items():
    # Make predictions
    predictions = model.predict(X_valid)
    
    # Store predictions in the dictionary
    predictions_dict[model_name] = predictions

#  predictions stored in predictions_dict
for model_name, predictions in predictions_dict.items():
    accuracy = accuracy_score(y_valid, predictions)
    print(f"Accuracy for {model_name}: {accuracy*100 :.2f} %")
# Get class probabilities for the input data
model = classifiers['Decision Tree']
model = classifiers['XGBClassifier']
model = classifiers['Random Forest']
predicted_probabilities = model.predict_proba(X_test)
# Display long output with scrolling
display(HTML("<div style='max-height: 300px; overflow-y: auto;'>"  + "</div>"))
print(model)
# Get the index of the maximum probability for each prediction
max_prob_indices = np.argmax(predicted_probabilities, axis=1)

# Get the corresponding patterns
corresponding_patterns = [y_pred[index] for index in max_prob_indices]

# Get the maximum probabilities
max_probabilities = [predicted_probabilities[i, index] for i, index in enumerate(max_prob_indices)]

# Print the list of maximum probabilities and their corresponding patterns
# find the greatest probability, for which the pattern prediction is different from the actual
# Initialize variables to keep track of maximum false prediction probability
max_false_prediction_probability = 0.0
for pattern,actual, probability in zip(corresponding_patterns, y_test, max_probabilities):
    print(f"Pattern: {pattern}, {actual}, Max Probability: {probability:.2f}") 
    if pattern != actual:
        if probability > max_false_prediction_probability:
            max_false_prediction_probability = probability

if max_false_prediction_probability > 0.0:
    print(f"The maximum false prediction probability is {max_false_prediction_probability:.4f}")
else:
    print("No false predictions found.")


for name, model in classifiers.items():
    print(model)
    joblib.dump(model, name)