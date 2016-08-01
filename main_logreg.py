import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.io as sio
import os 
import random
#from sklearn.metrics import accuracy_score
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.stdout= open("output_logreg.txt","w")

#Download the dataset 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

print('Total size of training images is ', mnist.train.images.shape)
print('Total size of test images is ', mnist.test.images.shape)

t_sample_sizes = [100, 1000, 5000, 10000, 20000, 40000, 55000]
acc = [];
for idx, t_sample_size in enumerate(t_sample_sizes):

	#train a logistic regression based classifier from scikitlearn
	logreg = LogisticRegression()
	r_index = random.sample(range(len(mnist.train.labels)),t_sample_size)
	logreg.fit(mnist.train.images[r_index,:], mnist.train.labels[r_index]) 

	res = logreg.predict(mnist.test.images)
	accuracy = metrics.accuracy_score(mnist.test.labels, res)
	acc.append(metrics.accuracy_score(mnist.test.labels, res))
	images = logreg.coef_.reshape((10 ,28, 28))
	image = images[0,:,:];
	for i in range(1,10):
		image = np.c_[image,images[i,:,:]]
		
	
	print('Number of training samples : ', t_sample_size)
	print('Classification report ', metrics.classification_report(mnist.test.labels, res))
	print('Accuracy is ', accuracy)
	# print('Confusion Matrix ', metrics.confusion_matrix(mnist.test.labels, res))
	print("--------------------------------------------------------------")
	print()


	
	plt.imshow(image)
	plt.axis('off')
	fn = "weights_"+ str(t_sample_size) + ".png"
	plt.savefig(fn,bbox_inches='tight')
sys.stdout.close()

# print(acc)
# plt.plot(t_sample_sizes, acc, linewidth=2.0)
# plt.xlabel('Training sample size')
# plt.ylabel('Accuracy')
# plt.title('Accuracy with training sample size')
# plt.savefig('trvsacc.png',bbox_inches='tight')




