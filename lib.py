import numpy
import pandas
from sklearn.model_selection import train_test_split

### Hàm đánh giá Nash-Sutcliffe efficiency (NSE)
def nse(targets, predictions):
    return (1-(numpy.sum((targets-predictions)**2)/numpy.sum((targets-numpy.mean(targets))**2)))

### Hàm tính train error, validation error
def train_test_err(targets, predictions):
    size = len(targets)
    if size != len(predictions):
        raise ValueError('Different size from targets and predictions')
    return sum(list(map(lambda x, y: (x-y)**2, predictions, targets)))/size

data = pandas.read_csv('Student_Performance.csv') ## Load data

dTrain, dTest = train_test_split(data, test_size=0.3, shuffle=False) ## Split data
xTrain, yTrain = numpy.array(dTrain.iloc[:,:5]), numpy.array(dTrain.iloc[:,5]) ## Data train model
xTest, yTest = numpy.array(dTest.iloc[:,:5]), numpy.array(dTest.iloc[:,5])  ## Data test model