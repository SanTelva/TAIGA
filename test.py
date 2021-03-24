# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:39:01 2021

@author: Asus
"""
import pandas as pd
import os
        
files = [f for f in os.listdir('.') if f.startswith("pointing_data_2020")]
file = files[0]
file2 = files[1]
df = pd.read_csv(file)


#def sew_tracking(template="pointing_data_2020"):
#    files = [f for f in os.listdir('.') if f.startswith(template)]
#    df = pd.read_csv(files[0])
#    df = df[["hh", "mm", "ss", "source_x", "source_y", "tracking", "is_good"]][df["is_good"]==1]
#    if len(files) > 1:
#        for i in range(1, len(files)):
#            df1 = pd.read_csv(files[i])
#            df1 = df1[["hh", "mm", "ss", "source_x", "source_y", "tracking", "is_good"]][df1["is_good"]==1]
#            df = pd.concat([df, df1])
#    return df
#
#df = sew_tracking()