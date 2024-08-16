import tensorflow as tf
class NaiveDense:#A blueprint for a naive dense layer, which when initialised makes an object (a naive dense layer).
    def __init__(self,input_size,output_size,activation):
        self.activation=activation#Initialises the properties of the object.
        w_shape=(input_size,output_size)
        w_initial_value=tf.random.uniform(w_shape,minval=0,maxval=1e-1)#Outputs random value from random distribution.
        self.W=tf.Variable(w_initial_value)

        b_shape=(output_size,)
        b_initial_value=tf.zeros(b_shape)
        self.b=tf.Variable(b_initial_value)

    def __call__(self,inputs):
        return (self.activation(tf.matmul(inputs,self.W)+self.b))
    
    def weights(self):
        return [self.W, self.b]#This method is created to enable easy retrieval of the weights. Although maybe in the main code you could use print(name_of_instance.W) or .b instead.
DenseLayer=NaiveDense(2,3,"Blah")

