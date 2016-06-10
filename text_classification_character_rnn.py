#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
This is an example of using recurrent neural networks over characters
for DBpedia dataset to predict class from description of an entity.

This model is similar to one described in this paper:
   "Character-level Convolutional Networks for Text Classification"
   http://arxiv.org/abs/1509.01626

and is somewhat alternative to the Lua code from here:
   https://github.com/zhangxiangxiao/Crepe
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import DataLoader

import numpy as np
from sklearn import metrics
import pandas

import tensorflow as tf
from tensorflow.contrib import learn

### Training data

print("LOAD")
loader = DataLoader('data/tweets',5,144)
x,y = loader.get_xy()
split = int(len(x)*0.8)
X_train,X_test = x[:split],x[split:]
y_train,y_test = pandas.Series(y[:split]),pandas.Series(y[split:])
y_train =  y_train.convert_objects(convert_numeric=True)
y_test =  y_test.convert_objects(convert_numeric=True)

### Process vocabulary

MAX_DOCUMENT_LENGTH = 144

print("PREPROCESS")
char_processor = learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(char_processor.fit_transform(X_train)))
X_test = np.array(list(char_processor.transform(X_test)))

### Models

HIDDEN_SIZE = 20

def char_rnn_model(X, y):
    byte_list = learn.ops.one_hot_matrix(X, loader.num_alpha)
    byte_list = learn.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, byte_list)
    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    _, encoding = tf.nn.rnn(cell, byte_list, dtype=tf.float32)
    return learn.models.logistic_regression(encoding, y)

classifier = learn.TensorFlowEstimator(model_fn=char_rnn_model, n_classes=loader.num_hash,
    steps=100, optimizer='Adam', learning_rate=0.01, continue_training=True)

# Continuously train for 100 steps & predict on test set.

print("TRAIN")
while True:
	classifier.fit(X_train, y_train)
	score = metrics.accuracy_score(y_test, classifier.predict(X_test))
	print("Accuracy: %f" % score)
