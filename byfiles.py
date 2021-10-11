#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 23:21:54 2021

@author: alex
"""
import Event
import os

def read_events(PATH='.'):
    events = []
    for name in os.listdir(PATH):
        e = Event.readevent(PATH + "/" + name)
        events.append(e)
    return events