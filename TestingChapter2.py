import numpy
A=numpy.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(A.shape)
print(A)
A=A.reshape(8,1)
print(A)