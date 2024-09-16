import pandas as pd
import matplotlib.pyplot as plt
import io

# Your data
data1 = """
Flow Pattern,Annular,Bubbly,I,SS,WS
1982 Ovad,374,268,1220,140,557
1985 Piu,31,0,103,0,0
1986 Koub,0,0,53,0,0
1997 N.V,26,13,52,5,0
2001 Mana,69,0,89,0,6
2001 Mata,14,54,54,4,28
2003 Ploa,8,34,159,8,8
"""
# Your data
data2 = """
Flow Pattern,Annular,Bubbly,I,SS,WS
1987 Suni,279,133,560,228,5
1997 Robe,7,0,156,0,41
1999 Meng,87,0,0,4,62
"""
# Create a pandas DataFrame
df1 = pd.read_csv(io.StringIO(data1), sep=',')
df2 = pd.read_csv(io.StringIO(data2), sep=',')

# Calculate the sum of each column
sum1 = df1.sum(numeric_only=True)
sum2 = df2.sum(numeric_only=True)
# Create a pie chart
plt.pie(sum1, labels=sum1.index, autopct='%1.1f%%')
plt.title('Training and test dataset')
plt.show()
#plt.savefig("Training_test_dataset.pdf")
plt.pie(sum2, labels=sum2.index, autopct='%1.1f%%')
plt.title('Additional validation of dataset')
plt.show()
#plt.savefig("Additional_Validation_dataset.pdf")
