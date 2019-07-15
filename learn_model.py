#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import json
from argparse import ArgumentParser

import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam

import query_processing

def get_model(width = 64, depth = 2, loss="mean_squared_error", len_input=32):
    one_input = Input(shape=(len_input,), name='one_input')
    x = Dense(width, activation="relu", kernel_initializer='normal')(one_input)
    
    for i in range(1,depth):
        width = max(8, int(width/2))
        x = Dense(width, activation="relu", kernel_initializer='normal')(x)
        x = Dropout(0.2)(x)
        
    x = Dense(1, kernel_initializer='normal', name="main_output", activation="linear")(x)
    
    model = Model(inputs=one_input, outputs=x)
    model.compile(loss=loss, optimizer=Adam(lr=0.0001))
    return model

def denormalize(y, y_min, y_max):
    return K.exp(y * (y_max - y_min) + y_min)

def denormalize_np(y, y_min, y_max):
    return np.exp(y * (y_max - y_min) + y_min)

def normalize(y):
    y = np.log(y)
    return (y - min(y))/(max(y) - min(y))

def q_loss(y_true, y_pred):
    y_true = denormalize(y_true, 0, N)
    y_pred = denormalize(y_pred, 0, N)
    
    return K.maximum(y_true, y_pred)/K.minimum(y_true, y_pred)

def q_loss_np(y_true, y_pred):
    #y_true = denormalize_np(y_true, 0, N)
    #y_pred = denormalize_np(y_pred, 0, N)
    
    return np.maximum(y_true, y_pred)/np.minimum(y_true, y_pred)

if __name__ == '__main__':
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    data = np.load(config["vector_file"])
    test_size = config["test_size"]
    ypred_per_run = []
    times_per_run = []
    qerror_per_run = []
    y = normalize(data[:, -1])
    N = np.max(np.log(data[:, -1]))
    last_qerror = np.inf

    for run in range(config["runs"]):
        K.clear_session()
        sample = np.random.choice(range(len(data)), size=len(data)-test_size, replace=False)
        not_sample = list(set(range(len(data))) - set(sample))

        X_train = data[sample, :-1]
        y_train = y[sample]

        X_test = data[not_sample, :-1]
        y_test = denormalize_np(y[not_sample], 0, N)

        model = get_model(depth=2, width=512, loss=q_loss, len_input=len(X_train[0]))
        start = time.time()
        model.fit(X_train, y_train, epochs=1, verbose=0, shuffle=True, batch_size=32, validation_split=0.1)
        end = time.time() - start
        times_per_run.append(end)
        
        y_pred = model.predict(X_test)[:, 0]
        # for testing per sample forward pass time
        # model.evaluate(X_test, normalize(y_test), batch_size=X_test.shape[0])
        y_pred = denormalize_np(y_pred, 0, N)
        ypred_per_run.append([(x.item(), y.item()) for x,y in zip(y_test, y_pred)])
        qerror = np.mean(q_loss_np(y_test, y_pred))

        if qerror < last_qerror:
            best_model = model
            last_qerror = qerror
        qerror_per_run.append(qerror)


    print("Average q-error: {:.2f}, Best q-error: {:.2f}".format(np.mean(qerror_per_run), last_qerror))

    with open("pred.json", "w") as output_file:
        json.dump(ypred_per_run, output_file)

    best_model.save(config["model_file"])
