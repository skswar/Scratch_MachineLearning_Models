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
Gradient descent is a fundamental algorithm in almost every application of machine learning. It plays a crucial role in minimizing the error between predicted and ground truth values, forming the foundation of machine learning as we know it today. This notebook focuses on implementing the gradient descent theory and loss function calculations. Initially, these functions are applied to a random data array, followed by their application to the well-known 'California Housing Dataset'. The accompanying images depict the reduction in loss over iterations. Notably, as the learning rate decreases, the algorithm takes longer to converge and sometimes possibly increases the likelihood of overfitting. To validate the implementation, the functions are compared with the scikit-learn SGDRegressor, and it is observed that the results are quite similar. The comparison results are mentioned in detail in the notebook.

<p align="center">
<img src="https://raw.githubusercontent.com/skswar/Scratch_MachineLearning_Models/master/img/LR_RandomDataArray.png" width="400px" height="300px"/>
<img src="https://raw.githubusercontent.com/skswar/Scratch_MachineLearning_Models/master/img/LR_CaliforniaHousing.png" width="400px" height="300px"/>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/skswar/Scratch_MachineLearning_Models/master/img/SGD_MSE.png" width="70%"/></p>

**Link To Notebook**: [Stochastic_Gradient_Descent_Scratch](<https://github.com/skswar/Scratch_MachineLearning_Models/blob/main/ML_Model_Scripts/Stochastic_Gradient_Descent_Scratch.ipynb>)

## Logistic Regression From Scratch
test








