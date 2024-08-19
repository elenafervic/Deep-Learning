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
    @property
    def weights(self):
        return [self.W, self.b]#This method is created to enable easy retrieval of the weights. Although maybe in the main code you could use print(name_of_instance.W) or .b instead.

class NaiveSequential():#Chains layers together. This is used to make the model, which means you don't have to call the layers each time.
    def __init__(self,layers):
        self.layers=layers#So the layers themselves are actually part of the object.

    def __call__(self,inputs):
        x=inputs
        for layer in self.layers:
            x=layer(x)#so the output of one layer goes on to be fed into the next layer. Do we need past values of x?
            return x
    @property
    def weights(self):
        print(1)

DenseLayer=NaiveDense(2,3,"Blah")

