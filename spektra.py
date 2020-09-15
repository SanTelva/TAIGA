# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:18:11 2020

@author: Hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def delta(time1, time2):
    #12:34:56,789.101.112
    t1, t1m = time1.split(",")
    t1 = list(map(int, t1.split(":")))
    t1m = list(map(int, t1m.split(".")))
    t2, t2m = time2.split(",")
    t2 = list(map(int, t2.split(":")))
    t2m = list(map(int, t2m.split(".")))
    
    dt = [t2[i] - t1[i] for i in range(3)]
    dtm = [t2[i] - t1[i] for i in range(3)]
    
    dt = 3600*dt[0]+60*dt[1]+dt[2]
    dtm = (1000000*dtm[0]+1000*dtm[1]+dtm[2])*10**(-9)
    if dtm > 0.001:
        return abs(dt+dtm)
    return abs(dt)

def diffSpectra(array, xlog = False, ylog=False, mult = 1):
    #array.sort()
    left = np.min(array)
    right = np.max(array)
    n = int(1+3.322*np.log10(len(array)))*2
    if not xlog:
        xdi = np.linspace(left, right, n)
        yhist = np.zeros(n)
        diff = (right - left) / (n-1)
        for elem in array:
            i = int(elem // diff)
            if not (elem == right): yhist[:i] += 1
    else:
        left = np.log10(left)
        right = np.log10(right)
        print(left, right)
        xdi = np.logspace(left, right, n)
        yhist = np.zeros(n)
        diff = (right - left) / (n - 1)
        for elem in array:
            i = int(np.log10(elem) // diff)
            if i - 1 >= n:
                #print(elem)
                yhist[-1] += 1
            else:  yhist[:i-1] += 1
        #yhist /= (5*3600)
    fig, ax = plt.subplots(figsize=(7,7))
    #ax.set_xlim(left = 500, right = 10**5)
    
    if xlog: ax.set_xscale("log")
    if ylog: 
        ax.set_yscale("log")
        ax.set_ylim(bottom = 0.1, top = 100000)
    plt.scatter(xdi, yhist/mult)
    plt.savefig("intSpectra1f.png", dpi=400)
    return (xdi, yhist)

df = pd.read_csv("Params.csv", "\t")
array = df["Size"]
#for i in range(len(array)):
    #array[i]=delta(df['Time'][i], df['Time'][i+1])
x, y = diffSpectra(array, ylog = True, xlog = True, mult = delta(df['Time'][0], df['Time'][12222]))
#plt.hist(array, bins = 400)
#plt.scatter(x, y)
