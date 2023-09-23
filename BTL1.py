import numpy
import pandas
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from hydroeval import evaluator, nse

### Hàm đánh giá Nash-Sutcliffe efficiency (NSE)
def nse(targets, predictions):
    return (1-(numpy.sum((targets-predictions)**2)/numpy.sum((targets-numpy.mean(targets))**2)))

### Hàm tính train error, validation error
def train_test_err(targets, predictions):
    size = len(targets)
    if size != len(predictions):
        raise ValueError('Different size from targets and predictions')
    return sum(list(map(lambda x, y: (x-y)**2, predictions, targets)))/size

### LinearRegression k-fold cross validation
def linear():
    ## K-fold cross validation
    ##########################
    kFold = KFold(10,shuffle=True) ## Chia thành 10 phần lấy ngẫu nhiên (shuffle=True)
    linearReg = LinearRegression()
    CVw = [] ## List điểm cross-validation
    trainFoldListX = [] ## List data train model
    trainFoldListY = [] ## List data train model
    for  (train, validate) in (kFold.split(xTrain, yTrain)): ## K-fold 
        linearReg.fit(xTrain[train], yTrain[train]) ## Train model with k-fold training set
        yPredTrain = linearReg.predict(xTrain[train]) 
        yPredvalidate = linearReg.predict(xTrain[validate])

        ### Calculate train error and validation error
        trainErr, testErr = train_test_err(yTrain[train].tolist(), yPredTrain.tolist()), train_test_err(yTrain[validate].tolist(),yPredvalidate.tolist())
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
            
### Lasso
def lasso():
    lassoReg = Lasso()
    lassoReg.fit(xTrain, yTrain)
    predict = numpy.array(lassoReg.predict(xTest))
    train_set = lassoReg.predict(xTrain)
    return {'Title': 'Lasso',
    'R2 score': r2_score(yTest, predict),
    'Score NSE': nse(yTest, predict),
    'Score NSE by hydroeval': evaluator(nse, predict, yTest),
    'Score MAE': mean_absolute_error(yTest, predict),
    'Score RMSE': mean_squared_error(yTest, predict)**0.5,
    'Test error': train_test_err(yTest.tolist(), predict.tolist()),
    'Train error': train_test_err(yTrain.tolist(), train_set.tolist())}

### Ridge
def ridge():
    ridgeReg = Ridge()
    ridgeReg.fit(xTrain, yTrain)
    predict = ridgeReg.predict(xTest)
    train_set = ridgeReg.predict(xTrain)
    return {'Title': 'Ridge',
    'R2 score': r2_score(yTest, predict),
    'Score NSE': nse(yTest, predict),
    'Score NSE by hydroeval': evaluator(nse, predict, yTest),
    'Score MAE': mean_absolute_error(yTest, predict),
    'Score RMSE': mean_squared_error(yTest, predict)**0.5,
    'Test error': train_test_err(yTest.tolist(), predict.tolist()),
    'Train error': train_test_err(yTrain.tolist(), train_set.tolist())}

data = pandas.read_csv('Student_Performance.csv') ## Load data

dTrain, dTest = train_test_split(data, test_size=0.3, shuffle=True) ## Split data
xTrain, yTrain = numpy.array(dTrain.iloc[:,:5]), numpy.array(dTrain.iloc[:,5]) ## Data train model
xTest, yTest = numpy.array(dTest.iloc[:,:5]), numpy.array(dTest.iloc[:,5])  ## Data test model

ll = [lasso(), ridge(), linear()] ## List models
ll.sort(key=lambda d: d['R2 score'], reverse=True) ## Sort model by key (R2 score) descending

### Print
for i in ll:
    for j in i:
        print(j, ': ', i[j], sep='')
    print()
