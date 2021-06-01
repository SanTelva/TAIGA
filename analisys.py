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

EXPOS = "171020.00"
IACT = "IACT01"
def param_hist(events, param, mode="", bins = 40, save=False):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(param, fontsize=14)
    ax.grid()
    if param.lower() == "size" or param.lower()=="con2":
        array = [getattr(e, param.lower()) for e in events]
        ax.hist(array, bins=bins)
    elif (param+"S") in e.Hillas.keys():
        if mode == "":
            for mod in ("N", "S", "A"):
                array = [e.Hillas[param+mod] for e in events]
                ax.hist(array, bins=bins, label = mod)
        else:
            array = [e.Hillas[param+mode] for e in events]
            ax.hist(array, bins=bins, label = mode)
    else:
        print("Wrong Hillas key")
        raise KeyError
    ax.legend()
    if save: plt.savefig(
            "/".join(["..", IACT, EXPOS, "events_cutted", "results", str(len(events))+"_"+param+mode+".png"]))
    
        

def cut(events, save = False, mode = "S"):
    marked_events = []
    for e in events:
        if (e.size > 120 
            and e.сon2 > 0.54
            and dist(e.Hillas["coords"+mode], (0, 0)) < 2.1
            and e.Hillas["width"+mode] < 0.076 * np.log10(e.size) - 0.047
            and e.Hillas["length"+mode] < 0.31
            and 0.36 < e.Hillas["dist"+mode] < 1.53
            and e.Hillas["alpha"+mode] < 10):
                marked_events.append(e)
                if save: e.vizualize(pixel_coords=pixel_coords, save=True)
    return marked_events

PATH = "/".join(["..", IACT, EXPOS, "events", ""])
# "../" + EXPOS + "/events/"
PATH_AN = "/".join(["..", IACT, EXPOS, "events_cutted", ""])
# "../"+EXPOS + "/events_cutted/"

events_cutted = []
for name in [f for f in os.listdir(PATH_AN) if f.endswith('.event')]:
    mode = "S"
    e = Event.readevent(PATH_AN+name)
    events_cutted.append(e)
        
for param in ("size", "con2", "width", "length", "dist", "azwidth", "alpha"):
        param_hist(events_cutted, param, mode="A", save=False)