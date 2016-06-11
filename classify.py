from utils import DataLoader

import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn
import pandas

print("LOAD")
MAX_DOC_LENGTH = 144
loader = DataLoader('data/tweets',5,MAX_DOC_LENGTH)
x,y = loader.get_xy()
split = int(len(x)*0.8)
X_train,X_test = x[:split],x[split:]
y_train,y_test = pandas.Series(y[:split]),pandas.Series(y[split:])
y_train =  y_train.convert_objects(convert_numeric=True)
y_test =  y_test.convert_objects(convert_numeric=True)

char_processor = learn.preprocessing.ByteProcessor(MAX_DOC_LENGTH)
X_test_vec = np.array(list(char_processor.transform(X_test)))

print("RESTORE")
classifier = learn.TensorFlowEstimator.restore('save')

print("VALIDATE")
out = classifier.predict(X_test_vec)
score = metrics.accuracy_score(y_test, out)
print("Accuracy: %f" % score)

print("CLASSIFY")
print(out[0])
