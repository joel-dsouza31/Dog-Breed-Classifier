# Dog-Breed-Classifier
This is the repo of Dog breed classifier project in Udacity ML Nanodegree.

# Project Overview
The goal of the project is to build a machine learning model that can be used within web app to process real-world, user-supplied images. The algorithm has to perform two tasks:

<li>Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.</li>
<li>If supplied an image of a human, The code will identify that it is a human</li>
Convolutional Neural Networks are used to classify the breeds of dogs and humans. The solution involves three steps. 
<li>First, to detect human images, we can use existing algorithm like OpenCV’s implementation of Haar feature based cascade classifiers.</li> <li>Second, to detect dog-images we will use a pretrained VGG16 model.</li> <li>Finally, after the image is identified as dog/human, we can pass this image to an CNN model which will process the image and predict the breed that matches the best out of 133 breeds.</li>

# CNN Architecture built from scratch:

I used the 5 convolution layers all with the colvolution of kernel size = 3, stride = 1 and padding = 0
Relu activations are used after each convoltution layers except the last one.
Max pooling layers of 2×2 are applied.
Batch normalization is applied after each max pool layer.
Dropout is applied with the probability of 0.2.
First layer will take three inputs for RGB because the in_channel is 3 and produces 16 output, the next layer to be a convolutional layer with 16 filters.
Input = 224x224 RGB image
Kernel Size = 3x3
Padding = 1 for 3x3 kernel
MaxPooling = 2x2 with stride of 2 pixels, which will reduce the size of image and by the result the number of parameters will be half.
Activation Function = ReLU (No vanishing gradient, there will be very vevry small output for input very larg or very small).
Batch Normalization 2D is a technique to provide inputs that are zero mean or variance 1.

Layer 1: (3,16) input channels =3 , output channels = 16
Layer 2: (16,32) input channels = 16 , output channels = 32
Layer 3: (32,64) input channels =32 , output channels = 64
Layer 4: (64,128) input channels =64 , output channels = 128
Layer 5: (128,256) input channels =128 , output channels = 256

One fully connected layer with 9216 input channels and 133 output channel as dog breeds.

# Refinement - CNN model created with transfer learning
The CNN created from scratch have accuracy of 13%, Though it meets the benchmarking, the model can be significantly improved by using transfer learning. To create CNN with transfer learning, I have selected the Resnet architecture which is pre-trained on ImageNet dataset. The last convolutional output of Resnet is fed as input to our model. We only need to add a fully connected layer to produce 133-dimensional output (one for each dog category). The model performed extremely well when compared to CNN from scratch. The model accuracy improved to 67%
