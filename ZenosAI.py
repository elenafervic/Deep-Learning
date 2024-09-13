import tensorflow as tf
import math
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
    def weights(self):#Presumably this is done every time you call the weights method.
        weights=[]#Creates an empty list
        for layer in self.layers:
            weights+=layer.weights
        return weights
class BatchGenerator:
    def __init__(self, images,labels,batch_size=128):
        assert len(images)==len(labels)
        self.index=0
        self.images=images
        self.labels=labels
        self.batch_size=batch_size
        self.num_batches=math.ceil(len(images)/batch_size)#why don't we use self.batch_size and self.images?

    def next(self):
        images=self.images[self.index:self.index+self.batch_size]#I wonder if this is a shallow copy or not. In any case, it doesn' include the endpoint.
        labels=self.labels[self.index:self.index+self.batch_size]
        self.index+=self.batch_size
        return images,labels#Does this return a tuple?

def one_training_set(model,images_batch,labels_batch):
    with tf.GradientTape() as tape:
        #Make predictions
        predictions=model(images_batch)

        #Calculate the loss
        per_sample_losses=tf.keras.losses.sparse_categorical_crossentropy(labels_batch,predictions)#The predictions have are in the form of probability to have each label (eg in our case its a 1D array with 10 probabilities). Meanwhile the "labels" variable just have the correct number.
        average_loss=tf.reduce_mean(per_sample_losses)#Each pair of label prediction and label have a loss, so we average the loss to see how close the predictions are to reality.
        
    #Compute the gradient of the loss w.r.t current weights.
    gradients=tape.gradient(average_loss,model.weights)#We find d(loss)/d(weights)

    #Now we update the weights.
    update_weights(gradients,model.weights)
    return average_loss #Presumably we may use this later.
#This is one of the training loops for one batch. We will later do each batch multiple times (different epochs).

learning_rate=1e-3#This is a global variable because it is assumed the learning rate won't change during the learning process.
def update_weights(gradients,weights):#Must have defined th elearning rate already.
    for g, w in zip(gradients,weights):
        w.assign_sub(g*learning_rate)#assign_sub is the equivalent of -= for tensorflow variables (of which w is an example of)
        #This is a really simple optimizer (gradient descent), we move w a little in the direction of fastest descent (-gradient).
        
#Can now make a model:
model=NaiveSequential([
    NaiveDense(input_size=28*28,output_size=512,activation=tf.nn.relu),
    NaiveDense(input_size=512,output_size=10,activation=tf.nn.softmax)    
    ]) #We create a Naive sequential object called model with two deep layers.
assert len(model.weights)==4
print("Yes")
#print(model.weights)



