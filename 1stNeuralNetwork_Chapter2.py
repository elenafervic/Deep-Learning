from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images,train_labels), (test_images,test_labels)=mnist.load_data()
print(train_labels[0])
digit=train_images[0]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
