import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.wrappers.scikit_learn import KerasClassifier

#Import the dataset
data = pd.read_csv('data.csv')
data.drop(['area code','phone number','state'], axis=1, inplace=True)
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#Label encode categorical features
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])
x[:,2] = le.fit_transform(x[:,2])
y = le.fit_transform(y)

#Split data into x_train,x_test,y_train,y_test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=0)

#Perform standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)



#from keras.models import Sequential
#from keras.layers import Dense
#ann = Sequential() #Initialize

#Adding layers to the ANN
#ann.add(Dense(activation='relu',units=9,kernel_initializer='uniform',input_dim=17))
#ann.add(Dense(activation='relu',units=9,kernel_initializer='uniform'))
#ann.add(Dense(activation='sigmoid',units=1,kernel_initializer='uniform'))
   
#Compile ANN 
#ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fit ANN to training data
#ann.fit(x_train,y_train, batch_size=10, epochs=100)

#Evaluating ANN
def build_ann(lr=0.1, momentum=0.4):
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    ann = Sequential()   
    ann.add(Dense(activation='softsign',units=10,kernel_initializer='uniform',input_dim=17))
    ann.add(Dense(activation='softsign',units=10,kernel_initializer='uniform'))
    ann.add(Dense(activation='sigmoid',units=1,kernel_initializer='uniform'))
       
    ann.compile(optimizer='RMSprop',loss='binary_crossentropy',metrics=['accuracy'])
    return ann

ann = KerasClassifier(build_fn=build_ann, batch_size=10, epochs=50)
from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator=ann, X=x, y=y, cv=10, n_jobs=-1)
#print(accuracies.mean())

#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

lr = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

neurons = [1, 5, 10, 15, 20, 25, 30]

#Tuning batch_size and epochs
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=ann, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x, y)
#Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#Tuning optimizer
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=ann, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x, y)
# # Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#Tuning learning rate and momentum
param_grid = dict(lr=lr, momentum=momentum)
grid = GridSearchCV(estimator=ann, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x, y)
# # Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#Tuning Kernel Initializer
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=ann, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x, y)
# # Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#Tuning Activation Function
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=ann, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x, y)
# # Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#Tuning number of neurons in the hidden layer
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=ann, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x, y)
# # Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
















