import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set color palette to "colorblind"
sns.set_palette('colorblind')

Shoham = True
Suni   = True
#Suni   = False
# Original scenario1 and scenario2 after finding the combination
# (1,0.35 ,1)
if (Shoham):
    scenario1 = {'SS': 98.97, 'SW': 79.63, 'A': 73.68, 'I': 60.78, 'DB': 100, 'Overall': 77.97} # Shoham
    scenario2 = {'SS': 98.97, 'SW': 70.37, 'A': 91.23, 'I': 83.01, 'DB': 82.35, 'Overall': 86.33} # Shoham
    instances = {'SS (97 DP)': 97, 'SW (54 DP)': 54, 'A (57 DP)': 57, 'I (153 DP)': 153, 'DB (34 DP)': 34, 'Overall (395 DP)': 395}
if (Suni):
    scenario1 = {'SS': 58.82, 'A': 62.5, 'I': 75.44, 'DB': 40, 'Overall': 67.88}                # Suni
    # (0.61,1.5,0.42)
    scenario2 = {'SS': 52.94, 'A': 100, 'I': 83.63, 'DB': 66.67, 'Overall': 83.21}               # Suni
    instances = {'SS (17 DP)': 17, 'A (56 DP)': 56, 'I (171 DP)': 171, 'DB (30 DP)': 30, 'Overall (274 DP)': 274}



scenario1_vals = scenario1.values()
scenario2_vals = scenario2.values()
instances_values = list(instances.values())
print(instances_values)
labels = scenario1.keys()
labels = instances.keys()
# plot 
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 8))
rects1 = ax.bar(x - width/2, scenario1_vals, width, label=r'($\lambda = 1$, $\beta = 0.35$, $\gamma = 1$)')



    


# Save the figure
if (Shoham and not (Suni)):
    rects2 = ax.bar(x + width/2, scenario2_vals, width, label=r'($\lambda = 0.72$, $\beta = 0.42$, $\gamma = 1.5$)')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'Accuracy ($\%$)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    # Adjusting y-axis ticks
    ax.set_yticks(range(0, 101, 10))
    ax.set_title('Overall accuracy and accuracy by flow pattern \n compared against experimental data of Shoham (1982) for horizontal pipe', fontsize=16)
    ax.legend(loc='best', fontsize=14)
    fig.tight_layout()
    plt.savefig("comparison_plot_shoham_original_best.png")
    plt.savefig("comparison_plot_shoham_original_best.pdf")

if (Suni):
    rects2 = ax.bar(x + width/2, scenario2_vals, width, label=r'($\lambda = 0.25$, $\beta = 0.47$, $\gamma = 0.83$)')
    ax.set_ylabel(r'Accuracy ($\%$)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    # Adjusting y-axis ticks
    ax.set_yticks(range(0, 101, 10))
    ax.set_title('Overall accuracy and accuracy by flow pattern \n compared against experimental data of Suni (1987) for horizontal pipe', fontsize=16)
    ax.legend(loc='best', fontsize=14)
    fig.tight_layout()
    plt.savefig("comparison_plot_suni_original_best.png") 
    plt.savefig("comparison_plot_suni_original_best.pdf")


plt.show()


