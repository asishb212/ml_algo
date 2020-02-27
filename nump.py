import numpy as np 
a=np.array([1,2,3])  #declares an array with type as numpy.ndarray
print(a.shape)  #prints size
#all the elements can be changed and can be brought by normal indexing
b = np.array([[1,2,3],[4,5,6]])   #2d array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0]) # [x,y] x is for lists y is for elements in the list
a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.][ 0.  0.]]"
b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"
c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.][ 7.  7.]]"
d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.][ 0.  1.]]"
arr=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# slicing van also be done here
sub_arr=arr[:2,1:3]  #rows x columns
#below are the same statements
print(arr[[0, 1, 2], [0, 1, 0]]) #prints (0,0) (1,1) (2,0)
poi=np.arange(4)  #gives array from 0 to 3
#math funcs + - * / sqrt mean can be called
#dot function is used to make matrix multiplication
l=np.array([1,5,8,6])
m=np.array([[1],[5],[6],[8]])
print(l.dot(m))  # both are same
print(np.dot(l,m))
print(np.sum(arr))  # Compute sum of all elements;
print(np.sum(arr, axis=0))  # Compute sum of each column;
print(np.sum(arr, axis=1))  # Compute sum of each row
arr.transpose()
fgt = np.empty_like(arr)   # Create an empty matrix with the same shape as arr
npoints = 20
slope = 2
offset = 3
x = np.arange(npoints)
y = slope * x + offset + np.random.normal(size=npoints)
print(y)
p = np.polyfit(x,y,2) 
print(p)
#gives a coefficients of polynomial according to given values 
p = np.poly1d([1, 2, 3])
print(np.poly1d(p))
#creates a polynomial