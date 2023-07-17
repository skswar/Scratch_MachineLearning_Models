<div align="center">
<img src="https://raw.githubusercontent.com/skswar/Scratch_MachineLearning_Models/master/img/MLLogo.png" alt="ML Intro Logo"/>
</div>
<br/>
<h3 align="center">Exploring the Inner Workings of Machine Learning Algorithms through Scratch Implementation</h3>
<br/>

## Table of contents
* [Introduction](#introduction)
* [Stochastic Gradient Descent From Scratch](#stochastic-gradient-descent-from-scratch)
* [Logistic Regression From Scratch](#logistic-regression-from-scratch)
* [Kmeans and Kmeans++ From Scratch](#kmeans-and-kmeans++-from-scratch)
* [Principal Component Analysis (PCA) From Scratch](#principal-component-analysis-from-scratch)

<hr>

## Introduction
In the current landscape, numerous packages and libraries are readily available in the market to enhance productivity in the fields of Data Science, Machine Learning, and Data Analysis. However, it is common for users to adopt these tools without gaining a comprehensive understanding of the underlying algorithms, leading to suboptimal outcomes. Therefore, this repository was created with the objective of delving into key machine learning algorithms, exploring their fundamental concepts and true algorithms, and conducting comparative analyses with existing packages to ensure a better understanding of their functionality and correctness.

## Stochastic Gradient Descent From Scratch
Gradient descent is a fundamental algorithm in almost every application of machine learning. It plays a crucial role in minimizing the error between predicted and ground truth values, forming the foundation of machine learning as we know it today. This notebook focuses on implementing the gradient descent theory and loss function calculations. Initially, these functions are applied to a random data array, followed by their application to the well-known 'California Housing Dataset'. The accompanying images depict the reduction in loss over iterations. Notably, as the learning rate (lr) decreases, the algorithm takes longer to converge and sometimes possibly increases the likelihood of overfitting (thus more MSE on test data with lr 0.001). To validate the implementation, the functions are compared with the scikit-learn SGDRegressor, and it is observed that the results are quite similar. The comparison results are mentioned in detail in the notebook.

<p align="center">
<img src="https://raw.githubusercontent.com/skswar/Scratch_MachineLearning_Models/master/img/LR_RandomDataArray.png" width="250px" height="200px"/>
<img src="https://raw.githubusercontent.com/skswar/Scratch_MachineLearning_Models/master/img/LR_CaliforniaHousing.png" width="250px" height="200px"/>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/skswar/Scratch_MachineLearning_Models/master/img/SGD_MSE.png" width="50%"/></p>

**Link To Notebook**: [Stochastic_Gradient_Descent_Scratch.ipynb](<https://github.com/skswar/Scratch_MachineLearning_Models/blob/main/ML_Model_Scripts/Stochastic_Gradient_Descent_Scratch.ipynb>)

## Logistic Regression From Scratch
The logistic regression algorithm is a fundamental method for binary classification in machine learning. The concept of logistic regression is not only used in Machine Learning but also in many deep learning algorithms. The idea is similar to the Stochastic Gradient Descent as we mentioned earlier, except, in Logistic Regression we try to predict the pobability of input data to belong to a certain class. Thus Logistic Regression uses concept of cross-entropy function, which compares the predicted probability with the true class label to minimize the loss and then a sigmoid function which classifies the output. In this notebook, I have implemented gradient descent for logistic regression, cross entropy loss function and sigmoid function from scratch. Then the implemented algorithm is applied on the Breast Cancer Wisconsin Data Set and its performance is evaluated. The following images show the performance of the algorithm through Loss Function Graph and Confusion matrix. Finally in the notebook, I also talk about the interpretation of the weights given by logistic regression which is a bit different than linear regression weights. 

<p align="center">
<img src="https://raw.githubusercontent.com/skswar/Scratch_MachineLearning_Models/master/img/LogReg_BC.png" width="250px" height="200px"/>
<img src="https://raw.githubusercontent.com/skswar/Scratch_MachineLearning_Models/master/img/BC_ConfusionMtrix.png" width="250px" height="200px"/>
</p>

**Link To Notebook**: [Logistic_Regression_Scratch.ipynb](<https://github.com/skswar/Scratch_MachineLearning_Models/blob/main/ML_Model_Scripts/Logistic_Regression_Scratch.ipynb>)

## Kmeans and Kmeans++ From Scratch
In unsupervised machine learning clustering is one of the first technical concepts that we learn. And when we lean clustering, we learns Kmeans Algorithm. The Kmeans algorithm has developed and taken number of different forms/upgrades to improve its performance, clustering efficiency, and its ability to deal with higher dimensional data. The most primitive two version of this algorithm is Kmeans and Kmeans++ which are till date used extensively in different data science applications. In this notebook, I first implemented the Kemans and Kmeans++ algorithms from scratch. Then applied this algorithm to the famous Iris dataset yo evaluate its performance.   




