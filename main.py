import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
mpl.use('Agg')
#mpl.rc()
from copy import deepcopy
import gc
#from matplotlib.patches import RegularPolygon #drawing hexagons
from Event import Event

Nrun = '{:03}'.format(10)
COLORS = np.array(['r', 'y', 'g', 'c', 'b', 'm', 'k'])

def Peds(Nrun):
    pathped = "./231119.01/peds/231119.ped_"
    peds = [[0 for i in range(64)] for j in range(24)] #здесь хранятся данные по пьедесталам 
                           #каждого канала по данному рану
    pedsfile = open(pathped+Nrun, "r")
    while True:  
        line = pedsfile.readline().split()
        if not line:
            break
        peds[int(line[0]) - 1][int(line[1])] = round(float(line[2])) #кластеры индексируются с 1, каналы с 0, 
                                                              #пьедестал округляю
    pedsfile.close()
    return peds
peds = Peds(Nrun)

c = [(17, [[4, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1], [149, 0], [26, 0], [0, 0], [2, 0], [0, 0], [0, 0], [3, 0], [4, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [182, 1], [20, 0], [79, 0], [17, 0], [82, 1], [6, 0], [3, 0], [1, 0], [11, 0], [0, 0], [19, 0], [0, 0], [0, 0], [0, 0], [25, 0], [4, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [150, 0], [12, 0], [50, 0], [0, 0], [0, 0], [0, 0], [2, 0], [0, 0], [130, 0], [13, 0], [19, 0], [2, 0], [0, 0], [0, 0], [4, 0], [0, 0], [0, 0], [0, 0]])]
a = Event(clusters = c)

factor = open("./231119.01/factors_051019.07fixed.txt", "r")
cluster_factors = [[1 for j in range(64)] for i in range(24)]
line = factor.readline()
for i in range(9):
    factor.readline()
while True:  
    line = factor.readline().split()
    #print(line)
    if line == []:
        break
    if line[4] == "NaN" or line[5] == "NaN": cluster_factors[int(line[0])-1][int(line[1])] = None
    else: cluster_factors[int(line[0])-1][int(line[1])] = float(line[4]) * float(line[5])
    #количество d.c., соответствующих одному фотоэлектрону
factor.close()

#попытаемся пересчитать в систему координат
xycoord = "xy_turn_2019j.txt"
coord = open("./231119.01/"+xycoord, "r")
pixel_coords = dict() #сопоставление ID ФЭУ и его координат
cluster_coords = [[[None, None, None] for i in range(64)] for j in range(24)]
while True:  
    # читаем одну строку
    line = coord.readline().split()
    #print(line)
    if line == []:
        break
    #print(list(map(float, line[3:5])))
    cluster_coords[int(line[0]) - 1][int(line[5])] = [float(line[3]), float(line[4]), int(line[2])]
    pixel_coords[int(line[2])] = (int(line[0]), float(line[3]), float(line[4]))
#print(len(coord.readlines()))
coord.close()

def outs(Nrun, peds = peds):
    pathout = "./231119.01/outs/231119.out_"

    fin=open(pathout+Nrun, "r")
    events = []
    events_cleaned = []
    while True:  
        # читаем одну строку
        line = fin.readline()
        if not line:
            break
        nclusters = int(line)
        clusters = []
        event = Event()
        for icluster in range(nclusters):
            cluster = [[0 for j in (1, 2)] for i in range(64)]
            Ncluster, Nevent, eventTime = fin.readline().split()
            Nchannel = 0
            for i in range(8):
                line = list(map(int, fin.readline().split()))
                for j in range(8):
                    cluster[Nchannel] = [line[2 * j] - peds[int(Ncluster) - 1][Nchannel], line[2 * j + 1]]   # 
                    #cluster[Nchannel][0] /= cluster_factors[int(Ncluster) - 1][Nchannel]
                    if cluster[Nchannel][0] < 0: cluster[Nchannel][0] = 0
                    Nchannel += 1
            clusters.append((int(Ncluster), cluster))
            #print("\n#", Nevent, "#", Ncluster)
            #for Nchannel in range(64):
                #if (not Nchannel%2) and (cluster[Nchannel][1]):
                    #print(Nchannel // 2, cluster[Nchannel][0])
        event = Event(int(Nevent), eventTime, clusters).recount(cluster_factors, cluster_coords)      
        events.append(event)
        z = event.clean()
        if len(z) > 4:
            events_cleaned.append(z)
        #print(clusters)
    fin.close()
    return events_cleaned

events_cleaned = []
fout = open("Params.csv", "w")
foutEvents = open("events.txt", "w")
print("Nrun", "ID", "Time", "Size", "A", "B", "Width", "Length", "Dis", "Miss", sep = "\t", file = fout)
for nrun in range(1, 139):
    #events_cleaned += outs('{:03}'.format(nrun), peds = Peds(nrun))
    Nrun = '{:03}'.format(nrun)
    print(Nrun)
    peds = Peds(Nrun)
    o = outs(Nrun, peds)
    for e in o:
        print(
        nrun, 
        e.Nevent, 
        e.time, 
        e.size,
        '{:.3f}'.format(e.Hillas["a"]),
        '{:.3f}'.format(e.Hillas["b"]), 
        '{:.3f}'.format(e.Hillas["width"]),
        '{:.3f}'.format(e.Hillas["length"]), 
        '{:.3f}'.format(e.Hillas["dis"]), 
        '{:.3f}'.format(e.Hillas["miss"]), 
        sep="\t", file=fout)
        if e.size > 1000:
             print(nrun, e.Nevent, file = foutEvents)
        #    e.vizualize(pixel_coords = pixel_coords, save = True)
    events_cleaned += o
foutEvents.close()
fout.close()    