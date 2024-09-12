import numpy
import math
#A=numpy.array([[[1,2],[3,4]],[[5,6],[7,8]]])
#print(A.shape)
#print(A)
#A=A.reshape(8,1)
#print(A)

w=[1,2]
b=[3,4]
a=w+b
print(a)

images=numpy.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]) #The length is 3
print(len(images))
print(math.ceil(len(images)))