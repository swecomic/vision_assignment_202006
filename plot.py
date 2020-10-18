# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:54:58 2020

@author: sweco
"""
import pandas as pd
import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
  
    plt.figure(figsize=(8,12))
  
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],
             label = 'Test Error')
    plt.ylim([0,5])
    plt.legend()
  
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy%')
    plt.plot(hist['epoch'], hist['accuracy'],
             label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'],
             label = 'Test Accuracy')
    plt.ylim([0,1])
    plt.legend()
    plt.show()
