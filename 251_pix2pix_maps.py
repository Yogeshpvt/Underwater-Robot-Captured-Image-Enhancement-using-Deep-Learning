
from os import listdir
from numpy import asarray, load
from numpy import vstack
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot as plt
import numpy as np

# load all images in a directory into memory
# def load_images(path, size=(256,512)):
# 	src_list, tar_list = list(), list()
# 	# enumerate filenames in directory, assume all are images
# 	for filename in listdir(path):
# 		# load and resize the image
# 		pixels = load_img(path + filename, target_size=size)
# 		# convert to numpy array
# 		pixels = img_to_array(pixels)
# 		# split into satellite and map
# 		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
# 		src_list.append(sat_img)
# 		tar_list.append(map_img)
# 	return [asarray(src_list), asarray(tar_list)]

# # dataset path
# path = 'maps/train/'
# # load dataset
# [src_images, tar_images] = load_images(path)
# print('Loaded: ', src_images.shape, tar_images.shape)


# n_samples = 3
# for i in range(n_samples):
# 	pyplot.subplot(2, n_samples, 1 + i)
# 	pyplot.axis('off')
# 	pyplot.imshow(src_images[i].astype('uint8'))
# # plot target image
# for i in range(n_samples):
# 	pyplot.subplot(2, n_samples, 1 + n_samples + i)
# 	pyplot.axis('off')
# 	pyplot.imshow(tar_images[i].astype('uint8'))
# pyplot.show()


def load_images(path, size=(256,256)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)


# dataset path
path = 'C:/Users/YOGESH V/Desktop/BlockageAI/CGAN/underwater_imagenet/'

# load dataset A - Monet paintings
dataA_all = load_images(path + 'trainA/')
print('Loaded dataA: ', dataA_all.shape)

from sklearn.utils import resample
#To get a subset of all images, for faster training during demonstration
dataA = resample(dataA_all, 
                 replace=False,     
                 n_samples=1000,    
                 random_state=42) 

# load dataset B - Photos 
dataB_all = load_images(path +'trainB/')
print('Loaded dataB: ', dataB_all.shape)
#Get a subset of all images, for faster training during demonstration
#We could have just read the list of files and only load a subset, better memory management. 
dataB = resample(dataB_all, 
                 replace=False,     
                 n_samples=1000,    
                 random_state=42) 

# plot source images
n_samples = 3
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + i)
	plt.axis('off')
	plt.imshow(dataA[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + n_samples + i)
	plt.axis('off')
	plt.imshow(dataB[i].astype('uint8'))
plt.show()


#######################################


from pix2pix_model import define_discriminator, define_generator, define_gan, train
# define input shape based on the loaded dataset
image_shape = dataA.shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

#Define data
# load and prepare training images
data = [dataA, dataB]

def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

dataset = preprocess_data(data)

from datetime import datetime 
start1 = datetime.now()

#     d_model = load_model('discriminator')
#     g_model = load_model('generator')
#     gan_model= load_model('gan')

train(d_model, g_model, gan_model, dataset, n_epochs=15 , n_batch=1) 
#Reports parameters for each batch (total 1096) for each epoch.
#For 10 epochs we should see 10960

stop1 = datetime.now()
#Execution time of the model 
execution_time = stop1-start1
print("Execution time is: ", execution_time)

#Reports parameters for each batch (total 1096) for each epoch.
#For 10 epochs we should see 10960
################################################
#load the trained model and again train

# def load():
#     d_model = load_model('discriminator')
#     g_model = load_model('generator')
#     gan_model= load_model('gan')
#     gan.summary()
#     discriminator.summary()
#     generator.summary()

#     return gan, generator, discriminator




#################################################

 #Test trained model on a few images...
from keras.models import load_model
from numpy.random import randint
from matplotlib import pyplot
import cv2
import numpy as np
from numpy import vstack

model = load_model('model_g_005000.h5')

# plot source, generated and target images
def plot_images(src_img, gen_img):  
    images = vstack((src_img, gen_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    #cv2.imshow('A.jpg',images[0])
    A1 = (images[0])*255
    cv2.imwrite('O_12.jpg',A1)
    #cv2.imshow('B.jpg',images[1])
    B1 = (images[1])*255
    cv2.imwrite('G_12.jpg',B1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


src_image1 = cv2.imread('Original_12.jpg')
src_image1 = cv2.resize(src_image1,(256,256))
src_image2 = 2. * src_image1 / 255. - 1
src_image = np.reshape(src_image2 , (1,256,256,3))
 #src_image, tar_image = X1[ix], X2[ix]
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image)

# gen_img =np.copy(gen_image)
# src_img = np.copy(src_image)

# from keras.models import load_model
# from numpy.random import randint
# model = load_model('model_015000.h5')

# # plot source, generated and target images
# def plot_images(src_img, gen_img, tar_img):
#     images = vstack((src_img, gen_img, tar_img))
#  	# scale from [-1,1] to [0,1]
#  	images = (images + 1) / 2.0
#  	titles = ['Source', 'Generated', 'Expected']
#  	# plot images row by row
#  	for i in range(len(images)):
# 		# define subplot
# 		pyplot.subplot(1, 3, 1 + i)
# 		# turn off axis
# 		pyplot.axis('off')
# 		# plot raw pixel data
# 		pyplot.imshow(images[i])
# 		# show title
# 		pyplot.title(titles[i])
#  	pyplot.show()
    
    
# [X1, X2] = dataset
# # select random example
# ix = randint(0, len(X1), 1)
# src_image, tar_image = X1[ix], X2[ix]
# # generate image from source
# gen_image = model.predict(src_image)
# # plot all three images
# plot_images(src_image, gen_image, tar_image)



