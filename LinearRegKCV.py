import numpy
import pandas
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from hydroeval import evaluator, nse
from lib import train_test_err

def linear(xTrain, yTrain, xTest, yTest):
    ## K-fold cross validation
    ##########################
    kFold = KFold(10,shuffle=False) ## Chia thành 10 phần lấy ngẫu nhiên (shuffle=True)
    linearReg = LinearRegression()
    CVw = [] ## List điểm cross-validation
    trainFoldListX = [] ## List data train model
    trainFoldListY = [] ## List data train model
    for  (train, validate) in (kFold.split(xTrain, yTrain)): ## K-fold 
        linearReg.fit(xTrain[train], yTrain[train]) ## Train model with k-fold training set
        yPredTrain = linearReg.predict(xTrain[train]) 
        yPredvalidate = linearReg.predict(xTrain[validate])

        ### Calculate train error and validation error
        trainErr = train_test_err(yTrain[train].tolist(), yPredTrain.tolist())
        testErr = train_test_err(yTrain[validate].tolist(),yPredvalidate.tolist())
        CVw.append(sum([trainErr, testErr])) ## Add to list điểm
        trainFoldListX.append(xTrain[train].tolist()) ## Add data to list data train model
        trainFoldListY.append(yTrain[train].tolist()) ## Add data to list data train model
    
    fitDataIndex = CVw.index(min(CVw)) ## Find the index of data model has lowest CVw point

    linearReg.fit(trainFoldListX[fitDataIndex],trainFoldListY[fitDataIndex]) ## Retrain model with the data
    predict = linearReg.predict(xTest) ## Predict with test data
    train_set = linearReg.predict(xTrain) ## Predict with train data
    return {'Title':'LinearRegression k-fold coss validation',
    'R2 score': r2_score(yTest, predict),
    'Score NSE': nse(yTest, predict),
    'Score NSE by hydroeval': evaluator(nse, predict, yTest),
    'Score MAE': mean_absolute_error(yTest, predict),
    'Score RMSE': mean_squared_error(yTest, predict)**0.5,
    'Test error': train_test_err(yTest.tolist(), predict.tolist()),
    'Train error': train_test_err(yTrain.tolist(), train_set.tolist())}

data = pandas.read_csv('Student_Performance.csv') ## Load data

dTrain, dTest = train_test_split(data, test_size=0.3, shuffle=False) ## Split data
xTrain, yTrain = numpy.array(dTrain.iloc[:,:5]), numpy.array(dTrain.iloc[:,5]) ## Data train model
xTest, yTest = numpy.array(dTest.iloc[:,:5]), numpy.array(dTest.iloc[:,5])  ## Data test model

linearReg = linear(xTrain, yTrain, xTest, yTest)

for i in linearReg:
    print('{0}: {1}'.format(i, linearReg[i]))