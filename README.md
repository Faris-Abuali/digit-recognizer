# Digit Recognizer
### Building a Neural Network from Scratch (no Tensorflow/Pytorch, just numpy &amp; math)

## General Overview
During this semester, I have had some experience with Python and machine learning basics, but I am still new to computer vision. This project was a perfect introduction to techniques like artificial neural networks (ANN) using a classic dataset including pre-extracted features.

## Problem Statement & Objective
* My goal is to correctly identify digits from a dataset of tens of thousands of handwritten images.
* The dataset is provided by: MNIST ("Modified National Institute of Standards and Technology"). It is one of the most popular datasets of computer vision.
   * This classic dataset of handwritten images has served as the basis for benchmarking classification algorithms.
   * As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.


<img width="587" alt="img2" src="https://user-images.githubusercontent.com/54215462/174850145-5140e711-3da7-49d3-859c-9ca42402530c.png">

<img width="599" alt="img3" src="https://user-images.githubusercontent.com/54215462/174850313-f22b9856-500c-4f24-8dc6-e09e261b99d1.png">

## Math Explanation
I implemented a two-layer neural artificial network and trained it on the MNIST digit recognizer dataset. It’s meant to be implemented from scratch in Pyton using only numpy library – to deal with lists/matrices – and basic math. This helped me understand the underlying math of neural networks better.


![image](https://user-images.githubusercontent.com/54215462/174850549-1fa4cb7a-8a96-4d18-b26f-ef36fe776e60.png)
### Input: The neural network’s input will be a 28px*28px black & white image. This means each image should be represented as a matrix of 28*28 = 784px


![image](https://user-images.githubusercontent.com/54215462/174850620-9e67c046-7c00-4e1b-93ab-c26934647479.png)
### My neural network will have the following architecture
* Input layer will have 784 units/neurons corresponding to the 784 pixels in each 28x28 input image 
* One Hidden layer with 10 neurons
* Output layer with 10 output units (because there will be 10 possible classifications from 0,..9)


