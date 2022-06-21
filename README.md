# Handwritten Digit Recognition
#### Building a Neural Network from Scratch (no Tensorflow/Pytorch, just numpy &amp; math)

## General Overview
During this semester, I have been studying the **Artificial Intelligence** curriculum. I have learnt some interesting topics, such as the **Artificial Neural Networks**. I have had some experience with Python and machine learning basics, but I am still new to computer vision. Building this project from scratch was a perfect introduction to _**Computer Vision**_ using artificial neural networks (ANN).

## What is Handwritten Digit Recognition?
The handwritten digit recognition is the ability of computers to recognize human handwritten digits. It is a hard task for the machine because handwritten digits are not perfect and can be made with many different flavors. The handwritten digit recognition is the solution to this problem which uses the image of a digit and recognizes the digit present in the image.

<hr />
I have made a PowerPoint presentation showing everything in detail. Kindly take a look at at it. It will help you grasp the concept much better :point_down:
<br />
:link: https://mega.nz/file/ilMmwajB#cvxWlmGEu969mOb8pTbj7WxT9hL4QTJ8UVK11OOs9i0


## 1. Problem Statement & Objective
* My goal is to correctly identify digits from a dataset of tens of thousands of handwritten images.
* The dataset is provided by: MNIST ("Modified National Institute of Standards and Technology"). It is one of the most popular datasets of computer vision.
   * This classic dataset of handwritten images has served as the basis for benchmarking classification algorithms.
   * As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.


<img width="587" alt="img2" src="https://user-images.githubusercontent.com/54215462/174850145-5140e711-3da7-49d3-859c-9ca42402530c.png">

<img width="599" alt="img3" src="https://user-images.githubusercontent.com/54215462/174850313-f22b9856-500c-4f24-8dc6-e09e261b99d1.png">

## 2. Math Explanation
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
<img width="595" alt="image" src="https://user-images.githubusercontent.com/54215462/174854176-0fa94878-c18f-4b4c-b2cb-cde88e4ce674.png">
<img width="560" alt="image" src="https://user-images.githubusercontent.com/54215462/174853454-7da0060b-f6cd-417b-8641-1354cbe509ba.png">
<img width="324" alt="image" src="https://user-images.githubusercontent.com/54215462/174853472-dfe67fce-76f1-4dc3-a033-ac3c5af3bf3b.png">
<img width="251" alt="image" src="https://user-images.githubusercontent.com/54215462/174853818-bea905d9-2416-43a9-b817-9436f000b180.png">
<img width="361" alt="image" src="https://user-images.githubusercontent.com/54215462/174853886-f84e9745-05ff-4290-a82e-1f9d65e7faea.png">

## 3. Coding it up
Here is a glimpse of the training set used to train our neural network:
<img width="599" alt="image" src="https://user-images.githubusercontent.com/54215462/174854320-072646ef-41f3-4acb-8d41-2ca0fab80dc7.png">
<br />
You can check out the file `code.py` for the complete code :grinning:

## 4. Results
<img width="600" alt="image" src="https://user-images.githubusercontent.com/54215462/174854726-60f6e1d0-9153-46a9-8162-565b99dc98d0.png">
<img width="599" alt="image" src="https://user-images.githubusercontent.com/54215462/174854765-e93b7016-7243-4110-8a27-401fab737fc0.png">
<img width="601" alt="image" src="https://user-images.githubusercontent.com/54215462/174854803-0d02b087-676c-4639-9a95-835570d8205c.png">
<img width="601" alt="image" src="https://user-images.githubusercontent.com/54215462/174854867-c88b0868-8709-4dfb-a521-0abbf215a040.png">
...
<img width="602" alt="image" src="https://user-images.githubusercontent.com/54215462/174854911-9061c8c7-60f1-4cec-abfc-dd1817e08739.png">
<img width="603" alt="image" src="https://user-images.githubusercontent.com/54215462/174854942-bd72217f-0549-4429-b59a-d0c5ce4f72b0.png">




#### Let's look at a couple of examples:
<img width="269" alt="image" src="https://user-images.githubusercontent.com/54215462/174855235-b416c89a-7964-4488-86aa-5e3d7f7f1b78.png">

<img width="192" alt="image" src="https://user-images.githubusercontent.com/54215462/174855280-3117ad7f-5d8b-44af-aa79-1b34db56411b.png">
<img width="195" alt="image" src="https://user-images.githubusercontent.com/54215462/174855128-2a795f1d-ba75-4949-8dc1-8b4d985dde25.png">
<img width="182" alt="image" src="https://user-images.githubusercontent.com/54215462/174855332-3f5826cc-538f-4e21-ae5d-073012102630.png">
<img width="177" alt="image" src="https://user-images.githubusercontent.com/54215462/174855364-d02c5c50-a449-4caa-a18d-0acfc830aabc.png">
<img width="606" alt="image" src="https://user-images.githubusercontent.com/54215462/174855433-69381952-a3d4-498e-b331-cf2545d4c016.png">



And that's it. If you found this repository interesting, kindly give it a star :star2::wink:
