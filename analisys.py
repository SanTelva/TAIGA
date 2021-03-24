# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:42:39 2021

@author: Asus
"""

def cut(events):
    marked_events = []
    for e in events:
        if (e.size > 120 
            and e.Con2 > 0.54
            and dist(e.Hillas["coordsS"], (0, 0)) < 2.1
            and e.Hillas["widthS"] < 0.076 * np.log10(e.size) - 0.047
            and 0.36 < e.Hillas["disS"] < 1.53):
                marked_events.append(e)
                e.vizualize(pixel_coords=pixel_coords, save=True)
    return marked_events