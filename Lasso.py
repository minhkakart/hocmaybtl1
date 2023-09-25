import numpy
import pandas
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from hydroeval import evaluator, nse
from lib import train_test_err

### Lasso
def lasso(xTrain, yTrain, xTest, yTest):
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

data = pandas.read_csv('Student_Performance.csv') ## Load data

dTrain, dTest = train_test_split(data, test_size=0.3, shuffle=False) ## Split data
xTrain, yTrain = numpy.array(dTrain.iloc[:,:5]), numpy.array(dTrain.iloc[:,5]) ## Data train model
xTest, yTest = numpy.array(dTest.iloc[:,:5]), numpy.array(dTest.iloc[:,5])  ## Data test model

lass = lasso(xTrain, yTrain, xTest, yTest)

for i in lass:
    print('{0}: {1}'.format(i, lass[i]))