import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Setting the default rcParams to format text as LaTeX
rcParams['text.usetex'] = True

# Data extracted from the LaTeX table
data = {
    "Model": [
        "XGBClassifier", "XGBClassifier", "XGBClassifier", "XGBClassifier", 
        "Random Forest", "Random Forest", "Random Forest", "Random Forest", 
        "XGBClassifier", "Random Forest", "Decision Tree", "Decision Tree", 
        "Decision Tree", "Decision Tree", "Decision Tree", "Support Vector Machine", 
        "Support Vector Machine", "Logistic Regression", "Logistic Regression", 
        "Logistic Regression", "Logistic Regression", "Support Vector Machine", 
        "Logistic Regression", "Support Vector Machine", "Support Vector Machine"
    ],
    "Oversampler": [
        "SMOTE", "ADASYN", "BorderlineSMOTE", "RandomOverSampler", 
        "RandomOverSampler", "BorderlineSMOTE", "SVMSMOTE", "SMOTE", 
        "SVMSMOTE", "ADASYN", "SMOTE", "SVMSMOTE", 
        "BorderlineSMOTE", "ADASYN", "RandomOverSampler", "RandomOverSampler", 
        "SVMSMOTE", "RandomOverSampler", "SMOTE", 
        "SVMSMOTE", "ADASYN", "ADASYN", 
        "BorderlineSMOTE", "SMOTE", "BorderlineSMOTE"
    ],
    "Accuracy": [
        95.46, 95.32, 94.75, 94.61, 
        94.61, 94.61, 94.47, 94.33, 
        94.04, 93.90, 93.76, 93.48, 
        92.62, 92.48, 91.21, 67.38, 
        67.38, 67.23, 65.67, 
        65.25, 64.82, 64.68, 
        64.40, 63.26, 57.16
    ]
}

# Data without oversampling
no_oversampling_data = {
    "Model": [
        "Random Forest", 
        "XGBClassifier", 
        "Decision Tree", 
        "Support Vector Machine", 
        "Logistic Regression"
    ],
    "Accuracy": [
        94.75, 
        94.61, 
        91.77, 
        77.59, 
        74.33
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)
df_no_oversampling = pd.DataFrame(no_oversampling_data)

# Create a grouped bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", hue="Oversampler", data=df, errorbar=None)

# Add horizontal lines for each model without oversampling
oversamplers = df['Oversampler'].unique()
for i in range(len(df_no_oversampling)):
    model = df_no_oversampling['Model'][i]
    accuracy = df_no_oversampling['Accuracy'][i]
    for j in range(len(oversamplers)):
        plt.hlines(accuracy, i - 0.2 + (j-2) * 0.1, i - 0.2 + (j + 2) * 0.1, colors='black', linestyles='-')
        plt.text(i, accuracy, f'{accuracy}\\%', ha='center', va='bottom')


#plt.title(r'\textbf{Accuracy of Machine Learning Models with Various Oversampling Techniques}')
plt.ylabel(r'Accuracy (\%)', fontsize=12)
plt.xlabel('')
plt.xticks(rotation=0)
# Set y-axis to show steps of 5%
plt.yticks(range(0, 101, 5))
plt.tight_layout()

# Change the font size of the model names and accuracy
plt.tick_params(axis='both', which='major', labelsize=12)
# Add a custom legend
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], color='black', linestyle='-'))
labels.append('Without Oversampling')
plt.legend(handles, labels, ncol=2)


# Save the figure as a PDF
plt.savefig('accuracy_plot_ML_models_oversampling_configurations.pdf', format='pdf')

plt.show()
