#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 19:46:00 2021

@author: alex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Event
import os

def dist(c1, c2):
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

EXPOS = "160920.00"

def param_hist(events, param, mode="", bins = 40):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(param+mode, fontsize=14)
    ax.grid()
    if param.lowercase() == "size" or param.lowercase()=="con2":
        array = [getattr(e, ) for e in events]
        ax.hist(array, bins=bins, c="g")
    elif (param+mode) in e.Hillas.keys():
        array = [e.Hillas[param+key] for e in events]
        ax.hist(array, bins=bins, c="g")
    else:
        print("Wrong Hillas key")
        raise KeyError
    
        

def cut(events, save = False, mode = "S"):
    marked_events = []
    for e in events:
        if (e.size > 120 
            and e.сon2 > 0.54
            and dist(e.Hillas["coords"+mode], (0, 0)) < 2.1
            and e.Hillas["width"+mode] < 0.076 * np.log10(e.size) - 0.047
            and e.Hillas["length"+mode] < 0.31
            and 0.36 < e.Hillas["dist"+mode] < 1.53):
                marked_events.append(e)
                if save: e.vizualize(pixel_coords=pixel_coords, save=True)
    return marked_events

PATH="../"+EXPOS+"/events/"

events_cutted = []
for name in os.listdir(PATH):
    mode = "S"
    try:
        e = Event.readevent(PATH+name)
        if (e.size > 120 
            and e.Con2 > 0.54
            and dist(e.Hillas["coords"+mode], (0, 0)) < 2.1
            and e.Hillas["width"+mode] < 0.076 * np.log10(e.size) - 0.047
            and e.Hillas["length"+mode] < 0.31
            and 0.36 < e.Hillas["dist"+mode] < 1.53):
                events_cutted.append(e)
    except ValueError:
        os.remove(PATH+name)
        print("File", PATH+name, "was removed")
        
    