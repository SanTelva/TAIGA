
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:18:11 2020

@author: Hp
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from Event import delta
import matplotlib.pyplot as plt


def func(x, a, b):
    return a * np.exp(-b * x)
def Spectra(array, integral = False, xlog = False, ylog = False, ax = None):
    array = pd.Series(array)
    if ylog: ax.set_yscale("log")
    return plt.hist(array, log=(xlog or ylog), bins = 50, ax = ax)


def logSpectra(a, inte = False, bins = 400):
    fig, ax = plt.subplots(figsize=(7, 7))
    a = np.array(a)
    
    left, right = np.log10(min(a)), np.log10(max(a))
    b = np.logspace(left, right, bins)
    h = np.zeros(bins)
    for e in a:
        i = np.log10(e)
        po = 0
        while b[po] < e: po += 1
        if po < len(b):
            if inte: h[:po] += 1
            else: h[po] += 1
        
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.scatter(b, h)
        

fig, ax = plt.subplots(figsize = (7, 7))   
df = pd.read_csv("Params00.csv", sep="\t")
df1 = pd.read_csv("Params01.csv", sep="\t")
array = [delta(df1["Time"][i], df1["Time"][i-1]) for i in range(1, len(df1["Time"]))]

array = pd.Series(array)
a2 = array[array < 2]
y, t, _ = plt.hist(a2, log=True, bins = 150, label=r"$\Delta$t spectrum")
t = t[:-1]
popt, pcov = curve_fit(func, t, y)
plt.plot(t, func(t, *popt), 'r-',
label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
#A, g = popt
ax.legend()
ax.set_xlabel("Time, s")
ax.text(0.6, 50, r"$A~\exp(-b\Delta t)$")
t = delta(df["Time"][0], df["Time"][len(df["Time"])-1])

