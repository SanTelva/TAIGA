#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:31:47 2020

@author: alex
"""

import datetime
import pandas as pd

timestamp = 1574523966.0
 #1339521878.04 
def time(timestamp):
    value = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    milli, micro = value.strftime('%f')[0:3], value.strftime('%f')[2:-1]
    return value.strftime('%H:%M:%S,')+".".join([milli, micro, "000"])

