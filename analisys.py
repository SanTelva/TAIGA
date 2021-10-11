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
import re

def logSpectra(a, inte = False, bins = 40, title = "Спектр", weight = 1):
    fig, ax1 = plt.subplots(figsize=(7, 7))
    a = np.array(a)
    hist, bins = np.histogram(a, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax1.hist(a, bins=logbins, cumulative = (-1) * inte, weights = weight)
    ax1.set_xscale('log')
    ax1.grid()
    ax1.set_yscale('log')
    ax1.set_title(title)
    
def AnalyzeFol(PATH):
    events = read_events(PATH)
    sizes = [e.size for e in events]
    n = len(events)
    t = Event.delta(events[0].time, events[-1].time)
    
    EXPOS = re.findall("/(\d{6})", PATH)
    if EXPOS =='170920':
        t = 60*154
    if not EXPOS: EXPOS = PATH 
    logSpectra(sizes, inte = False, title = EXPOS[0]+" norm diff", weight = [1/t] * n)
    logSpectra(sizes, inte = True, title = EXPOS[0]+" norm integral", weight = [1/t] * n)
    return n, t

def sigma(S, A):
    return (S - A) / np.sqrt(S + A)

def dist(c1, c2):
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

# EXPOS = "171020.00"
# IACT = "IACT01"

def read_events(PATH='.'):
    events = []
    for name in os.listdir(PATH):
        e = Event.readevent(PATH + "/" + name)
        events.append(e)
    return events


def param_hist(events, param, mode="", bins = 40, save=False, log = False):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(param, fontsize=14)
    ax.grid()
    ON, OFF = 0, 0
    if param.lower() == "size" or param.lower()=="con2":
        array = np.array([getattr(e, param.lower()) for e in events])
    #     if log:
    #         left, right = np.log10(min(array)), np.log10(max(array))
    #         b = np.logspace(left, right, bins)
    #         h = np.zeros(bins)
    #         for ev in array:
    #             i = np.log10(ev)
    #             po = 0
    #             while b[po] < ev: po += 1
    #             if po < len(b):
    #                 if inte: h[:po] += 1
    #                 else: h[po] += 1
    #         ax.plot(b, h)    
    #         ax.set_yscale("log")
    #         ax.set_xscale("log")
        h = ax.hist(array, bins=bins)
    elif (param+"S") in e.Hillas.keys():
        if mode == "":
            for mod in ("S", "A"):
                array = [e.Hillas[param+mod] for e in events]
                h = ax.hist(array, bins=bins, label = mod)
                if mod == "S":
                    ON = sum([1 for z in array if z < 5])
                else:
                    OFF = sum([1 for z in array if z < 5])
                if param == "alpha":
                    ax.set_xlim(left=0, right = 15)
        else:
            array = [e.Hillas[param+mode] for e in events]
            ax.hist(array, bins=bins, label = mode, log = log)
    else:
        print("Wrong Hillas key")
        raise KeyError
    ax.legend()
    if save: plt.savefig(
            "/".join(["..", IACT, EXPOS, "events_cutted", "results", str(len(events))+"_"+param+mode+".png"]))
    return ON, OFF
        

def cut(events, save = False, mode = "S"):
    marked_events = []
    for e in events:
        if (e.size > 120 
            and e.con2 > 0.54
            and dist(e.Hillas["coords"+mode], (0, 0)) < 2.1
            and e.Hillas["width"+mode] < 0.076 * np.log10(e.size) - 0.047
            and e.Hillas["length"+mode] < 0.31
            and 0.36 < e.Hillas["dist"+mode] < 1.53
            and e.Hillas["alpha"+mode] < 10):
                marked_events.append(e)
                if save: e.vizualize(pixel_coords=pixel_coords, save=True)
    return marked_events

# PATH = "/".join(["..", IACT, EXPOS, "events", ""])
# "../" + EXPOS + "/events/"
#PATH_AN = "/".join(["..", IACT, EXPOS, "events_cutted", ""])
# "../"+EXPOS + "/events_cutted/"
PATH_AN = "../DFTA"
PATH_DATA = "../DFdays"

events_TA = []
files = [c for c in os.listdir(PATH_DATA)]
ready =  [c for c in os.listdir(PATH_AN)]
# ready = ["230920", "170920", "211020", "171020"]
# ready = ["171020"]
for day in ready:
    # # events = []
    # if day not in ready:
    #     # каты здесь работают хуже, чем каты налету
    #     print(day)
    #     os.mkdir(PATH_AN+"/"+day)
    #     names = [f for f in os.listdir(PATH_DATA+"/"+day)]
    #     print(day, len(names))
    #     for name in names:
    #         e = Event.readevent(PATH_DATA+"/"+day+"/"+name)
    #         events.append(e)
    #     for e in cut(events):
    #         e.saveevent("", "", "DFTA/"+day+"/")
    # if day in ready:
        names = [f for f in os.listdir(PATH_AN+"/"+day)]
        print(day, len(names))
        for name in names:
            e = Event.readevent(PATH_AN+"/"+day+"/"+name)
            # if (e.Hillas["alphaS"] < 15) and (e.Hillas["alphaA"] < 15):
            events_TA.append(e)
    
    
        
S, A = param_hist(events_TA, "alpha", mode='', bins = 15)
# param_hist(events_TA, "size", log = True)
print(sigma(S, A))
print(len(events_TA), "events")