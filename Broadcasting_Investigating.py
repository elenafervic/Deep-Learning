import numpy
A=numpy.array([1,2,3,4,5])
#print(A.shape)
#print(A)#So this first array 5 rows, so its a column vector. Cause it's shape is (5,). 

B=numpy.array([[1,2,3,4,5]])
#print(B.shape)#B is the equivalent of a row vector now, cause it has 1 row and 5 column.
#print(B)

A=numpy.expand_dims(A,axis=0)
C=numpy.concatenate([A]*2,axis=0)#Why does [A]*2 result in two arrays to be concatenated instead of an array with 2 arrays within?
C[0,0]=0
print(C)

D=numpy.random.random((2,3,4,5))
E=numpy.random.random((4,5))
F=numpy.maximum(D,E)
print("This is the correct answer",F)

E=numpy.expand_dims(E,axis=0)
E=numpy.concatenate([E]*3,axis=0)
E=numpy.expand_dims(E,axis=0)
E=numpy.concatenate([E]*2,axis=0)

print(F==numpy.maximum(D,E))

#I think its true.