# Neural Network Charity Analysis

## Overview of the analysis
#### The purpose of this project is to help a Fundation predict where to make investments by using TensorFlow and a neural network or deep learning model to create a binary classifier. 
#### This binary classification model will be used to predict whether the organizations that might receive funds from the foundation will be successful based on the features in the dataset.  
#### In this sense, the first step is to preprocess the dataset using Pandas and Scikit-Learn's StandarScaler().
#### Finally, the binary classification model created should be optimized using TensorFlow to meet a target predictive accuracy higher than 75%. It is required that at least three attempts are made to meet this target.
#### Google Collab Notebook is used for this project. 



## Results
---
***ORIGINAL MODEL***

+ **Target variable:** IS_SUCCESSFUL
+ **Feature variables:** 
APPLICATION_TYPE            
AFFILIATION                 
CLASSIFICATION              
USE_CASE                     
ORGANIZATION                
STATUS                      
INCOME_AMT                   
SPECIAL_CONSIDERATIONS      
ASK_AMT                   
+ **Removed variables (non-beneficial):**
EIN, NAME
+ **Neurons:** 110
+ **Hidden Layers:** 2
+ **Output Layer:** 1

+ **Activation functions:**
Relu in the first and second layers. Sigmoid in the output layer
+ **Target model performance of 75% of accuracy:** Not achieved. Accuracy of 0.7257142663002014

![D21](https://github.com/Connectime4ever/Neural_Network_Charity_Analysis/blob/main/D21.png)

---

***NEW MODEL-FIRST ATTEMPT***
+ **Target variable:** IS_SUCCESSFUL
+ **Feature variables:** 
APPLICATION_TYPE      
AFFILIATION           
CLASSIFICATION        
USE_CASE              
ORGANIZATION          
INCOME_AMT            
ASK_AMT 
+ **Removed variables (non-beneficial):**
EIN, NAME, SPECIAL_CONSIDERATIONS,STATUS
+ **Neurons:** 216
+ **Hidden Layers:** 2
+ **Output Layer:** 1

![a1](https://github.com/Connectime4ever/Neural_Network_Charity_Analysis/blob/main/a1.png)


+ **Activation functions:**
 Relu in the first and second layers. Sigmoid in the output layer
+ **Additional Steps taken to increase model performance:**
    - More non-beneficial variables were taken off: SPECIAL_CONSIDERATIONS,STATUS
    - The bining of feature APPLICATION_TYPE was changed by increasing the number of values for each bin.
    -  Adding more neurons to the first and second layer (108 total each),following a rule of thumb: three times the number of features. Besides, adding more neurons could make the model smater,faster and more robust. 
+ **Target model performance of 75% of accuracy:** Not achieved. Accuracy of 0.7241982221603394

![a12](https://github.com/Connectime4ever/Neural_Network_Charity_Analysis/blob/main/a12.png)

---

***NEW MODEL-SECOND ATTEMPT***
+ **Target variable:** IS_SUCCESSFUL
+ **Feature variables:** 
APPLICATION_TYPE      
AFFILIATION           
CLASSIFICATION        
USE_CASE              
ORGANIZATION          
INCOME_AMT            
ASK_AMT 
+ **Removed variables (non-beneficial):**
EIN, NAME, SPECIAL_CONSIDERATIONS,STATUS
+ **Neurons:** 324
+ **Hidden Layers:** 3
+ **Output Layer:** 1

![a2](https://github.com/Connectime4ever/Neural_Network_Charity_Analysis/blob/main/a2.png)

+ **Activation functions:**
 Relu in the first and second layer. Sigmoid in the output layer
+ **Additional Steps taken to increase model performance:**
    -  Adding a new layer for a total of 324 neurons (108 each) for the reasons explained above. Besides, changing the structure of the model with a new layer can help the model identifying nonlinear characteristics of the input data with no additional data. It is considered that 3 layers should be enough even for more complex model, so only 3 are used in this case.  
    -  Increasing the number of epochs to the training regimen for a total of 200. This contributes to provide each neuron with more information about the input data, increasing the chances for the neurons to apply more effective weight coeficients.
+ **Target model performance of 75% of accuracy:** Not achieved. Accuracy of 0.722449004650116

![a21](https://github.com/Connectime4ever/Neural_Network_Charity_Analysis/blob/main/a21.png)

---

***NEW MODEL-THIRD ATTEMPT***
+ **Target variable:** IS_SUCCESSFUL
+ **Feature variables:** 
APPLICATION_TYPE      
AFFILIATION           
CLASSIFICATION        
USE_CASE              
ORGANIZATION          
INCOME_AMT            
ASK_AMT 
+ **Removed variables (non-beneficial):**
EIN, NAME, SPECIAL_CONSIDERATIONS,STATUS
+ **Neurons:** 324
+ **Hidden Layers:** 3
+ **Output Layer:** 1

![a3](https://github.com/Connectime4ever/Neural_Network_Charity_Analysis/blob/main/a3.png)

+ **Activation functions:**
 Leaky Relu for the three layers. Sigmoid in the output layer
+ **Steps taken to increase model performance:**
    -  Changing the activation function of the three layers to Leaky ReLU. Using this  higher complexity activation function could help avoiding the risk of ignoring lower complexity features.  
    - Increasing the number of epochs to the training regimen for a total of 500. This contributes to provide each neuron with more information about the input data, increasing the chances for the neurons to apply more effective weight coeficients. 
 + **Target model performance of 75% of accuracy:** Not achieved. Accuracy of 0.723498523235321

[a31](https://github.com/Connectime4ever/Neural_Network_Charity_Analysis/blob/main/a31.png)  

## Summary

 + None of the three attemps of optimizing the original deep learning model were successful neither improved the original model performance. All of them show an accuracy rate of 0.72 approximately, underperforming the target of 75% set for the model.  
 
 + It is recomended to try a different approach using support vector machines (SVMs) to solve this classificaton problem. Sometimes SVMs could overperform basic or deep learning models in binary classification cases like this one. Besides, its implementation requires less coding. 

 