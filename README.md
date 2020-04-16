# Churn-using-ANN
Predicting churn of a telecom company using an Artificial Neural Net.

# Data
The dataset used to build the Artificial Neural Network is taken from the online data science portal Kaggle. The dataset consists of roughly 3500 rows and 21 columns. Each row corresponds to a single entry mentioning the details of a single customer and whether that particular customer stayed with the company. The last column of the dataset ‘churn’ describes this. If ‘churn=True’ for a particular customer, then the customer discontinued the telecom services. Whereas if ‘churn=False’, then the customer continues the telecom services.

Data: https://www.kaggle.com/becksddf/churn-in-telecoms-dataset

# Objectives
To predict and rank the customers most likely to discontinue the telecom services. Doing this will enable the organization to focus on these customers and develop methods to retain them.
To provide valuable insight into which factors impact the churning decision of the customer the most. Using this, the organization can  focus on the development of these factors and withdraw its resources from the development of non essential factors.  

# Steps to build a robust model
Data Preprocessing
Building the ANN
Evaluating the ANN
Hyperparameter Tuning
Data Visualization

# Data Preprocessing
Data preprocessing/preparation/cleaning is the process of detecting and correcting (or removing) corrupt or inaccurate records from a dataset, or and refers to identifying incorrect, incomplete, irrelevant parts of the data and then modifying, replacing, or deleting the dirty or coarse data. Often real world data is full of noise. It is very important to get rid of the ‘garbage’ data as a ML model trained on ‘garbage’ data will perform very poorly in general.


# Building the ANN
ANN (Artificial Neural Networks) is an information processing paradigm that is inspired by the way the biological nervous system such as brain process information. It is composed of large number of highly interconnected processing elements(neurons) working in unison to solve a specific problem. ANN are mostly used for regression and classification problems. Whereas CNN (Convolutional Neural Networks) are used for tasks relating to computer vision. 


# Evaluating the ANN
The method that is used to evaluate the classifying model is K-fold Cross Validation. In k-fold cross-validation, the original dataset is randomly partitioned into k equal size subsets. Of the k subsets, a single subset is retained as the testing data, and the remaining k-1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsets used exactly once as the testing data.
--- As the Keras library is used to build the ANN, we need to wrap the ANN in a special scikit-learn wrapper. Doing this, the scikit-learn library can be used.


# Hyperparameter Tuning
Hyperparameter tuning for the fastest execution time and best accuracy rate is done by using a powerful technique called Grid Search. Grid search is a process that searches exhaustively through a manually specified subset of the hyperparameter space of the targeted algorithm. Another technique that can be used for hyperparameter tuning is the Random Search. 
