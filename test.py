# importing pandas as pd
import pandas as pd
import numpy as np
import operator

# Creating row index values for our data frame
ind = pd.date_range('01 / 01 / 2000', periods=5, freq='12H')

# Creating a dataframe with 4 columns
# using "ind" as the index for our dataframe
df = pd.DataFrame({"A": [1, 2, 3, 4, 5],
				"B": [10, 20, 30, 40, 50],
				"C": [11, 22, 33, 44, 55],
				"D": [12, 24, 51, 36, 2]},
				index=ind)

# print(df["C"] > 34, df["C"].shift() <= 34)
# print('----------------------------------')
# print(df["C"] , df["C"].shift())

# df['X'] = np.where(np.logical_and((df['RCSI'] > 34), (df['C'].shift() <= 34)), 1, 0)

# print(df['X'])

students = {}
math = students["001"] = 1
science = students["002"] = 2
art = students["003"] = 3
social_science = students["004"] = 4
new_ma_val = max(students.items(), key=operator.itemgetter(1))[0]

print(new_ma_val, students[new_ma_val])

for i in range(10):
	with open('output-btc.txt', 'w') as f:
		f.writelines(str(i))

myfile = open('xyz.txt', 'w')
for line in range(10):
	rs = str(line) +''+ 'i'
	myfile.write("%s\n" % rs)
    
myfile.close()
    
