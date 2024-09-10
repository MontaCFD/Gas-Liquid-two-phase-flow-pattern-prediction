import numpy as np 
from math import sqrt, pi, radians, sin, cos, tan, log, exp, pow
import joblib
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Ignore the UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    
# Get the path to the test directory
test_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to the project root by going up one level from test
project_root_dir = os.path.abspath(os.path.join(test_dir, os.pardir))

# Get the absolute path to the src directory
src_dir = os.path.join(project_root_dir, 'src')

# Add the src directory to the Python path
sys.path.append(src_dir)

# Verify that the src directory is in sys.path
print('sys.path:', sys.path)

# Import modules from the src directory
try:
    from FP_prediction import *
    from thermodynamic_properties import *
    print('Modules imported successfully.')
except ImportError as e:
    print(f'Error importing modules: {e}')
#----------------------------------------------------------------------------------------------------------------------
def filter_points(points):
    # Create a dictionary to store the sum of y-coordinates and count of points for each unique x-coordinate
    sum_y_count = {}
    
    # Create a dictionary to store the sum of y-coordinates for each unique x-coordinate
    sum_y_for_x = {}
    
    # Iterate through the points
    for x, y in points:
        # Check if the x-coordinate is already in the dictionary
        if x in sum_y_count:
            # Update the count of points
            sum_y_count[x] += 1
            
            # Update the sum of y-coordinates if y > 0.01
            if y > 0.01:
                sum_y_for_x[x] += y
        else:
            # Initialize the count of points
            sum_y_count[x] = 1
            
            # Initialize the sum of y-coordinates if y > 0.01
            if y > 0.01:
                sum_y_for_x[x] = y
            else:
                sum_y_for_x[x] = 0
    
    # Calculate the mean y-value for each unique x-coordinate
    mean_y_for_x = {x: sum_y_for_x[x] / sum_y_count[x] for x in sum_y_for_x}
    
    # Create a list of tuples with the unique x-coordinate and its corresponding mean y-value
    filtered_points = [(x, mean_y_for_x[x]) for x in sorted(mean_y_for_x.keys())]
    
    # Return the filtered points
    return filtered_points


# Idea: adaptive logspacing of relevant intervals and appending them to one list 
def logspacing(m,n):
    # Define the interval [a, b]
    #m = -3
    #n = 2
    if (m >= n):
        print('first int m should be < n') 
    else:
        a = 10**(m)
        b = 10**(n)
        #list = []
        # Define the specific linear spacings
        j  = m 
        # Initialize an empty list to store the discretized values
        discretized_values = []
        disc_list = []
        while (j < n):
            if ( j >= 0):
                spacing = 0.1 #/ 5.
            if (j > 0):
                spacing = 0.1 #/ 5.
                
            spacing = 10 ** (j-1) # / 5. 
            #spacing = .5 * spacing
            low = 10 ** (j)
            top = 10 ** (j + 1)
            values = np.linspace(low, top, int((top - low) / spacing) + 1)
            discretized_values.append(values)
            # update j 
            j = j + 1
        # Sort the values in ascending order
        disc_list = [item for sublist in discretized_values for item in sublist]

    return disc_list

# adaptive logspacing
def adaptive_logspace(start, stop, step):
    values = []
    current_value = start
    while current_value <= stop:
        values.append(current_value)
        current_value *= step
    return np.array(values)

# coefficients and factors 
# source doi:10.1016/j.cherd.2011.08.009
# S-NS: lamda = 0.25  S is predicted to 100 % and lamda = 7.0   NS is predicted to 100 %
# B-NB: alpha factor 
# Annular- Non annular hL/d between 0.35 and 0.5 


best = True
#best = False

# Trained models names
trained_models_names = ['XGBClassifier - oversampled Data', 'XGBClassifier', 'Random Forest', 'Decision Tree']
joblib_ml_files      = ['XGB_oversampled_data.joblib', 'XGBClassifier.joblib', 'RandomForest.joblib', 'DecisionTree.joblib']

# Data sources
# Get the path to the notebooks directory
notebooks_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to the project root by going up one level from notebooks
project_root_dir = os.path.dirname(notebooks_dir)

# Construct the path to the data directory
data_dir = os.path.join(project_root_dir, 'data')
# Data sources
file_names = ['validation_horizontal_pipe_1986_shoham.csv',
              'validation_horizontal_pipe_1986_shoham-d1.csv', 
              'validation_horizontal_pipe_1986_shoham-d2.csv',
              'validation_horizontal_pipe_1987_Suni.csv',
              'validation_horizontal_pipe_1987_Suni-d1.csv',
              'validation_horizontal_pipe_1987_Suni-d2.csv', 
              'validation_horizontal_pipe_1997_Robe.csv', 
              'validation_horizontal_pipe_1999_Meng.csv']




# Create a subdirectory called "FlowMap" if it doesn't exist
if not os.path.exists("FlowMap"):
    os.makedirs("FlowMap")

# read data file 
p = 0
#read_file    = file_names[p]
read_file = os.path.join(data_dir, file_names[p])

# Create the plot name by combining the filename without the .csv extension and the trained model name
plot_name = f"{os.path.splitext(read_file)[0]}" + f"standard-vs-best"
# Load the data from the CSV file
df = pd.read_csv(read_file)
# access the data in the DataFrame
rho_l = df['DenL'].values[0]
rho_g = df['DenG'].values[0]
mu_l  = df['VisL'].values[0]
mu_g  = df['VisG'].values[0]
g     = 9.81
ST    = df['ST'].values[0]
ID    = df['ID'].values[0]
Ang   = df['Ang'].values[0]

# Select configuration
best_configuration_list     = [(0.71,0.42,1.5), (0.71,0.42,1.5), (0.72,0.42,1.5), (0.25, 0.47, 0.83), (0.25, 0.47, 0.83),(0.25, 0.47, 0.83), (1.54, 0.33,0.5), (0.71, 0.25, 0.5)]
standard_configuration_list = [(1.0, 0.35, 1) for _ in file_names]


#
lamda1, beta1, gamma1 = standard_configuration_list[p]
lamda2, beta2, gamma2 = best_configuration_list[p]

# Lists of superficial velocites 
U_sg_List = logspacing(m=-2, n=2)
U_sl_List = logspacing(m=-3, n=1)
print(len(U_sg_List))
print(len(U_sl_List))
X          = []
Fr_sg_list = []
Fr_sl_list = []
# Transition lines 

T_standard    = [[], [], [], []]
 

T_best        = [[], [], [], []]
# Record the start time
start_time = time.time()

# Froude number coefficients for superficial velocity
#FL = sqrt(rho_l/(rho_l - rho_g)) * 1. / sqrt(g * ID * cos(radians(Ang))) 
#FG = sqrt(rho_g/(rho_l - rho_g)) * 1. / sqrt(g * ID * cos(radians(Ang)))
FL = 1.
FG = 1.
# Assuming U_sl_List and U_sg_List are lists of values
points = []
#k = 0 
for i in range(len(U_sg_List)):
    for j in range(len(U_sl_List)):
        #k = k + 1 
        #print(k)
        # Get superficial velocities
        U_sg = U_sg_List[i]
        U_sl = U_sl_List[j]

        # transition line 
        A   = pi / 4.0 * ID ** 2.0 
        m_g = U_sg * A * rho_g
        m_l = U_sl * A * rho_l
        x   = m_g / (m_g + m_l)
        s   = sqrt(1.0 - x * (1.0 - rho_l / rho_g))
        #print(x, s)
        epsilon = 1. / (1. + s * (1. - x) / x * rho_g / rho_l)
        # mechanistic model: standard configuration
        ptt, FP, Fr_sl, Fr_sg = FPT_Horizontal_Pipe_test(U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, ID, Ang,lamda1, gamma1, beta1)
         
        Fr_sg_list.append(Fr_sg)
        Fr_sl_list.append(Fr_sl)
        points.append((Fr_sg, Fr_sl, ptt))
        
        if (j == 0):
            b1   = ptt 
            #print('b=', b)
        else: 
            a1   = ptt 
            if (not (a1 == b1)):
                #print('a=', a, 'b=', b)
                # build midpoint 
                
                U1 = U_sg = U_sg_List[i]
                U2 = 0.5 * (U_sl_List[j - 1] + U_sl_List[j])
                # TODO iterate more to find the solution (Transition point with more accuracy!)
                delta = abs(U_sl_List[j] - U_sl_List[j - 1])

                U_min = min(U_sl_List[j - 1], U_sl_List[j])
                U_max = max(U_sl_List[j - 1], U_sl_List[j])
                U_m  = 0.5 * (U_min + U_max) 
                error = delta 
                while( error > delta / 50.):  
                    U_sl = U_m
                    m_g  = U_sg * A * rho_g
                    m_l  = U_sl * A * rho_l
                    x    = m_g / (m_g + m_l)
                    s    = sqrt(1.0 - x * (1.0 - rho_l / rho_g))
                    #print(x, s)
                    epsilon = 1. / (1. + s * (1. - x) / x * rho_g / rho_l)
                    # X = (U_sl, U_sg, rho_l, rho_g, mu_l, mu_g, ID, Ang)
                    data = (U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, ID, Ang)
                    ptt, FP, Fr_sl, Fr_sg = FPT_Horizontal_Pipe_test(U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, ID, Ang, lamda1, gamma1, beta1)
                    c1 = ptt 
                    if (not ((c1 == a1) or (b1 == a1) )):
                        break 
                    else:
                        if (c1 == b1):
                            U_min = U_m 
                        if (c1 == a1): 
                            U_max = U_m
                        
                    print(b1, c1, a1)
                    U_m   = 0.5 * (U_min + U_max)    
                    error =  abs(U_max - U_min)
                
                # now get the more accurate solution 
                U2 = U_m 
                
                # T1: S-NS Transitionline with empirical model
                if ((b1 == 0 and a1 == 3) or (b1 == 1 and a1 == 2) or (b1 == 1 and a1 == 3)):
                    T_standard[0].append((U1 * FG, U2 * FL))
                
                # T2: SS-SW Transitionline with empirical model
                if(b1 == 0 and a1 == 1):
                    T_standard[1].append((U1 * FG, U2 * FL))
                
                # T3: A-I/DB Transitionslinie with empirical model
                if(b1 == 2 and (a1 == 3 or a1 == 4)):
                    T_standard[2].append((U1 * FG, U2 * FL))
         
                # T4: I-DB Transitionsline with empirical model
                if (b1 == 3 and a1 == 4):
                    T_standard[3].append((U1 * FG, U2 * FL))
                        
            # store previous FP value 
            b1 = a1
                
                
        # Mechanistic model: best configuration 
        ptt, FP, Fr_sl, Fr_sg = FPT_Horizontal_Pipe_test(U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, ID, Ang, lamda2, gamma2, beta2)
       
        if (j == 0):
            b2   = ptt 
            #print('b=', b)
        else: 
            a2   = ptt 
            if (not (a2 == b2)):
                #print('a=', a, 'b=', b)
                # build midpoint 
                
                U1 = U_sg = U_sg_List[i]
                U3 = 0.5 * (U_sl_List[j - 1] + U_sl_List[j])
                # TODO iterate more to find the solution (Transition point with more accuracy!)
                delta = abs(U_sl_List[j] - U_sl_List[j - 1])

                U_min = min(U_sl_List[j - 1], U_sl_List[j])
                U_max = max(U_sl_List[j - 1], U_sl_List[j])
                U_m  = 0.5 * (U_min + U_max) 
                error = delta 
                while( error > delta / 50.):  
                    U_sl = U_m
                    m_g  = U_sg * A * rho_g
                    m_l  = U_sl * A * rho_l
                    x    = m_g / (m_g + m_l)
                    s    = sqrt(1.0 - x * (1.0 - rho_l / rho_g))
                    #print(x, s)
                    epsilon = 1. / (1. + s * (1. - x) / x * rho_g / rho_l)
                    # X = (U_sl, U_sg, rho_l, rho_g, mu_l, mu_g, ID, Ang)
                    X    = [[U_sl, U_sg, rho_l, rho_g, mu_l, mu_g, ID, Ang]]
                    
                    ptt, FP, Fr_sl, Fr_sg = FPT_Horizontal_Pipe_test(U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, ID, Ang, lamda2, gamma2, beta2)
                    c2   = ptt 
                    if (not ((c2 == a2) or (b2 == a2) )):
                        break 
                    else:
                        if (c2 == b2):
                            U_min = U_m 
                        if (c2 == a2): 
                            U_max = U_m
                        
                    print(b2, c2, a2)
                    U_m   = 0.5 * (U_min + U_max)    
                    error =  abs(U_max - U_min)
                
                # now get the more accurate solution 
                U3 = U_m 
                
                    
                # T1: S-NS Transitionline with ML model
                if ((b2 == 0 and a2 == 3) or (b2 == 1 and a2 == 2) or (b2 == 1 and a2 == 3)):
                    T_best[0].append((U1 * FG, U3 * FL))
                
                # T2: SS-SW Transitionline with ML model
                if(b2 == 0 and a2 == 1):
                    T_best[1].append((U1 * FG, U3 * FL))
                
                # T3: A-I/DB Transitionslinie with ML model
                if(b2 == 2 and (a2 == 3 or a2 == 4)):
                    T_best[2].append((U1 * FG, U3 * FL))
         
                # T4: I-DB Transitionsline with ML model
                if (b2 == 3 and a2 == 4):
                    T_best[3].append((U1 * FG, U3 * FL))
                        
            
            # store previous FP value   
            b2 = a2   
    

        
# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the result
print(f"Elapsed time: {elapsed_time} seconds")           
# ------------------------------------------------------------
# Initialize empty lists for each column
VSG_list = df['Vsg'].tolist()
VSL_list = df['Vsl'].tolist()
FP_list  = df['Flow_label'].tolist()
# Combine feature1_list and feature2_list into a list of points
validation_points = list(zip(VSG_list, VSL_list))

# Initialize a dictionary to store sublists for each pattern
validation_points_FP_lists = {}

# Iterate through the data and create sublists for each pattern
for pattern, point in zip(FP_list, validation_points):
    if pattern not in validation_points_FP_lists:
        validation_points_FP_lists[pattern] = []
    validation_points_FP_lists[pattern].append(point)

# --------------------------------------------------------------
         

label = [ 'S-NS', 'SS-SW', 'A-I/B', 'I-B']
FP_label = ['SS', 'SW', 'A', 'I', 'DB']
#print(TR)
#print(list(len(T_standard[i]) for i in range(4)))

# plot 
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })
# > Markers M[i] and colors C[i]
C = [
    'C0',  # Blue
    'C1',  # Orange
    'C2',  # Green
    'C3',  # Red
    'C4',  # Purple
    'C5',  # Brown
    'C6',  # Pink
    'C7',  # Gray
    'C8',  # Olive
    'C9',  # Cyan
    'C10', # Maroon
    'C11'  # Teal
]

M = ['s' ,'^' ,'o', 'D', '*', 'x','v' ,'D' ,'o' ,'h' ,'+' ]  # '*''x'

fig = plt.figure(figsize=(10, 10))
#plt.suptitle(r'Flow map of water-air flow in horizontal pipe  using Taitel et al. model (1976) and ML-Model XGBClassifier ', fontsize=14)

# Get the number of unique flow patterns
num_patterns = len(validation_points_FP_lists)

# Define the flow pattern indices and their corresponding labels
flow_patterns = {
    0: 'SS',
    1: 'SW',
    2: 'A',
    3: 'I',
    4: 'B'
}

# Plot the validation points with different colors and markers
for i in range(len(validation_points_FP_lists)):
    pattern, points = list(validation_points_FP_lists.items())[i]
    points = list(zip(*points))
    plt.scatter(points[0], points[1], label=flow_patterns[pattern], color=C[i], marker=M[i])

# Plot the empirical data
for p in range(len(T_standard)):
    # Separate x and y coordinates from the points list
    if T_standard[p]:  # Check if the sublist is not empty
        x1, y1 = zip(*T_standard[p])
    # Plot the empirical data with solid lines
    plt.plot(x1, y1, linestyle='solid', linewidth=3.5, marker='', markersize=3, color=C[5])
    
    

# Apply the filter_points function to each sublist in T_best
T_best_filtered = [filter_points(sublist) for sublist in T_best]

#T_best_filtered =    T_best
# Plot the best configuration data
for p in range(len(T_best)):
    # Separate x and y coordinates from the points list
    x2, y2 = zip(*T_best[p])
    
    # Plot the best configuration data with dotted lines
    plt.plot(x2, y2, linestyle='solid', linewidth=3.5, marker='', markersize=3, color=C[7])

# Add legends for the plots
# Define the label with LaTeX formatting
label1 = r'Taitel \& al. ($\lambda$ = ' +str(lamda1) + r', $\beta$ = ' + str(beta1) + r', $\gamma$ =' + str(gamma1) + r')'
label2 = r'Taitel \& al. ($\lambda$ = ' +str(lamda2) + r', $\beta$ = ' + str(beta2) + r', $\gamma$ =' + str(gamma2) + r')'   
plt.plot([], [], linestyle='solid', linewidth=3.5, marker='', markersize=3, color=C[5], label=label1)
plt.plot([], [], linestyle='solid', linewidth=3.5, marker='', markersize=3, color=C[7], label=label2)
                   
plt.xlabel(r' Superficial Gas velocity $U_{SG}$  $[\frac{m}{s}]$', size='12')
plt.ylabel(r' Superficial Liquid velocity $U_{SL}$  $[\frac{m}{s}]$', size='12')
plt.legend(loc='best', ncol=2, fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin=1e-2, xmax=1e2) 
plt.ylim(ymin=1e-3, ymax=1e1) 
plt.grid(True, which="both", ls="-", color='0.65', linewidth=0.5)    

# Save the plot as PNG and PDF files in the "FlowMap" subdirectory
plt.savefig(f"FlowMap/{plot_name}.png")
plt.savefig(f"FlowMap/{plot_name}.pdf")
plt.show()
    
    
    

