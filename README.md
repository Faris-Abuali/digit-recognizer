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


<img width="329" alt="image" src="https://user-images.githubusercontent.com/54215462/174853072-9ea6f6a5-e4f3-49c8-9420-815d9a64894a.png">


<img width="512" alt="image" src="https://user-images.githubusercontent.com/54215462/174853190-5aecab3b-563b-4b2f-8bd5-9bcbbf50564b.png">

My neural network will have the following architecture
* Input layer will have 784 units/neurons corresponding to the 784 pixels in each 28x28 input image 
* One Hidden layer with 10 neurons
* Output layer with 10 output units (because there will be 10 possible classifications from 0,..9)


<img width="576" alt="image" src="https://user-images.githubusercontent.com/54215462/174853244-09a4c88c-9600-4b19-a9ae-cca61f69e9a3.png">


<img width="593" alt="image" src="https://user-images.githubusercontent.com/54215462/174853299-b19d3f08-f93b-4337-826e-ba02dd0efa30.png">


<img width="518" alt="image" src="https://user-images.githubusercontent.com/54215462/174853334-bd019130-912d-45ca-a266-f2247ad5696e.png">
<img width="599" alt="image" src="https://user-images.githubusercontent.com/54215462/174853373-a177016d-b892-442c-9d5b-10cb1f2883f7.png">
<img width="601" alt="image" src="https://user-images.githubusercontent.com/54215462/174853417-f770313b-5437-4375-b5bd-ce7eeec8bc46.png">
<img width="560" alt="image" src="https://user-images.githubusercontent.com/54215462/174853454-7da0060b-f6cd-417b-8641-1354cbe509ba.png">
<img width="324" alt="image" src="https://user-images.githubusercontent.com/54215462/174853472-dfe67fce-76f1-4dc3-a033-ac3c5af3bf3b.png">

