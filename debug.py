import numpy as np
import pdb;   # Add this line to set a breakpoint[]
# Create two lists
list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]
pdb.set_trace()
# Convert lists to NumPy arrays
array1 = np.array(list1)
array2 = np.array(list2)

# Sum the two arrays
result = array1 + array2

print("Resulting array:", result)