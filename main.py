import numpy as np
import pandas as pd
from Event import *
import os
import re
def sew_tracking(IACT = "IACT01", template="pointing_data_2020"):
    files = [('../'+IACT+'/trackings/'+f) for f in os.listdir('../'+IACT+'/trackings/') if f.startswith(template)]
    print(files)
    df = pd.read_csv(files[0])
    df = df[["time", "hh", "mm", "ss", "telescope_el", "source_x", "source_y", "tracking", "is_good"]][df["is_good"]==1]
    if len(files) > 1:
        for i in range(1, len(files)):
            df1 = pd.read_csv(files[i])
            df1 = df1[["time", "hh", "mm", "ss", "telescope_el", "source_x", "source_y", "tracking", "is_good"]][df1["is_good"]==1]
            df = pd.concat([df, df1])
    return df

Nportion = '{:03}'.format(10)
COLORS = np.array(['r', 'y', 'g', 'c', 'b', 'm', 'k'])
EXPOS = "121020.00"
IACT = "IACT01"
if IACT == "IACT02":
    xyfile = "xy_iact02_2020jul.txt"
    factor_file = "factors_iact2_310120.led.txt"
elif IACT == "IACT01":
    xyfile = "xy_turn_2019j.txt"
    factor_file = "factors_051019.07normed.txt"
if len(EXPOS.split('.')) > 1:
    OBSERVDATE, NRUN = EXPOS.split(".")
else:
    OBSERVDATE, NRUN = EXPOS, 0
NRUN = int(NRUN)
day, month, year = int(OBSERVDATE[:2]), int(OBSERVDATE[2:4]), 2000+int(OBSERVDATE[4:6]) 
PEDSTYPE = "peds"

wobble=True
elev_thr = 50

portions_amount = len(os.listdir(path="../"+IACT+"/"+EXPOS+"/outs"))
# portions_amount = 2
if not "events" in os.listdir("../"+IACT+"/"+EXPOS):
    os.mkdir("../"+IACT+"/"+EXPOS+"/"+"events")
# portions_amount=2
def dist(c1, c2):
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)


def Peds(Nportion):
    pathped = "/".join(["..", IACT, EXPOS, PEDSTYPE, EXPOS[:-2]+"ped_"])
    #pathped = "../231119.01/peds.mediana/231119.ped_"
    peds = [[0 for i in range(64)] for j in range(24)] #здесь хранятся данные по пьедесталам 
                           #каждого канала по данному рану
    pedsfile = open(pathped+Nportion, "r")
    while True:  
        line = pedsfile.readline().split()
        if not line:
            break
        peds[int(line[0]) - 1][int(line[1])] = round(float(line[2])) #кластеры индексируются с 1, каналы с 0, 
                                                              #пьедестал округляю
    pedsfile.close()
    return peds
peds = Peds(Nportion)

# "../231119.01/factors_051019.07fixed.txt"
factor = open("/".join(["..", IACT, factor_file]), "r")
cluster_factors = [[1 for j in range(64)] for i in range(24)]
# line = factor.readline()
# for i in range(9):
#     factor.readline()
# while True:  
#     line = factor.readline().split()
#     #print(line)
#     if line == []:
#         break
#     if line[4] == "NaN" or line[5] == "NaN": cluster_factors[int(line[0])-1][int(line[1])] = None
#     else: cluster_factors[int(line[0])-1][int(line[1])] = float(line[4]) * float(line[5])
#     #количество d.c., соответствующих одному фотоэлектрону
# factor.close()
line = factor.readline()
while True:  
    line = factor.readline().split()
    #print(line)
    if line == []:
        break
    cl, ch, pe, sens = line
    if pe == "NaN" or sens == "NaN": cluster_factors[int(cl)-1][int(ch)] = None
    else: cluster_factors[int(cl)-1][int(ch)] = float(pe) * float(sens)
    #количество d.c., соответствующих одному фотоэлектрону
factor.close()

#попытаемся пересчитать в систему координат 
coord = open("/".join(["..", IACT, xyfile]), "r")
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
neighbours = [[] for i in range(640)]
defined = [(None, None) for i in range(640)]
for cluster in cluster_coords:
    for x, y, i in cluster:
        if i is not None:
            defined[i] = (x, y)
for i in range(640):
    for j in range(640):
        if not (None in defined[i] or None in defined[j]):
            if dist(defined[i], defined[j]) < 3.1 and i != j:
                neighbours[i].append(j)
del defined
                
def outs(Nportion, peds = peds):
    pathout = "/".join(["..", IACT, EXPOS, "outs", EXPOS[:-2]+"out_"])
    # pathout = "../231119.01/outs/231119.out_"

    fin=open(pathout+Nportion, "r")
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
        event = Event(int(Nevent), eventTime, clusters)
        event.recount(cluster_factors, cluster_coords)      
        events.append(event)

        z = event.cclean(neighbours)
        if len(z) >= 4:
            events_cleaned.append(z)
        #print(clusters)
    fin.close()
    return events_cleaned, events

events_cleaned = []
events = 0
timefile = open("Times.txt", "w")
# fout = open("Params"+IACT+EXPOS+"W"*wobble+".csv", "w")
pointing = sew_tracking(IACT, "pointing_data_"+"-".join([str(year),'{:02d}'.format(month), '{:02d}'.format(day)]))
pointing.index = np.arange(len(pointing))

#foutEvents = open("events.txt", "w")
 # print("nportion", "ID", "Time", 
 #      "Size", "сon2",
 #      "<xN>", "<xS>", "<xA>", 
 #      "<yN>", "<yS>", "<yA>",
 #      "WidthN", "WidthS", "WidthA",
 #      "LengthN", "LengthS", "LengthA",
 #      "distN", "distS", "distA", 
 #      "MissN", "MissS", "MissA", 
 #      "AzwidthN", "AzwidthS", "AzwidthA",  
 #      "AlphaN", "AlphaS", "AlphaA", 
      # sep = "\t", file = fout)
for nportion in range(1, 1+portions_amount):
    if not re.search("\d*", EXPOS).group(0)  in os.listdir("../DFdays"):
        os.mkdir("../DFdays/"+re.search("\d*", EXPOS).group(0))
    #events_cleaned += outs('{:03}'.format(nportion), peds = Peds(nportion))
    Nportion = '{:03}'.format(nportion)
    print(Nportion, "/", '{:03}'.format(portions_amount))
    peds = Peds(Nportion)
    o, events= outs(Nportion, peds)
    timeindex = 0
    for e in o:
        timepoint = time(pointing.time[timeindex])
        while delta(timepoint, e.time) > 1 and timeindex<len(pointing)-1:# or timestamp(e.time, datetime.date(year, month, day)) > timepoint:
            timeindex += 1
            timepoint = time(pointing.time[timeindex])
            # print("point: ", timepoint, file = timefile)
            if timestamp(e.time) < timestamp(timepoint):
                
                break
        # print("event: ", e.time, e.Nevent, file = timefile)
           
        if not (wobble): e.params(angles = True)
        elif pointing.telescope_el[timeindex] > elev_thr: #порог на угол места
            e.source_x = pointing.source_x[timeindex]
            e.source_y = pointing.source_y[timeindex]
            e.params(e.source_x, e.source_y, angles = True)
            e.zenith = 90 - pointing.telescope_el[timeindex]
        if not(None in e.Hillas.values()) and e.zenith < 90 - elev_thr:
            # print(
            #     nportion, 
            #     e.Nevent, 
            #     e.time.replace(",", ";"), 
            #     e.size,
            #     '{:.3f}'.format(e.con2),
            #     '{:.3f}'.format(e.Hillas["coordsN"][0]),
            #     '{:.3f}'.format(e.Hillas["coordsS"][0]),
            #     '{:.3f}'.format(e.Hillas["coordsA"][0]),
            #     '{:.3f}'.format(e.Hillas["coordsN"][1]),
            #     '{:.3f}'.format(e.Hillas["coordsS"][1]),
            #     '{:.3f}'.format(e.Hillas["coordsA"][1]),
            #     '{:.3f}'.format(e.Hillas["widthN"]),
            #     '{:.3f}'.format(e.Hillas["widthS"]),
            #     '{:.3f}'.format(e.Hillas["widthA"]),
            #     '{:.3f}'.format(e.Hillas["lengthN"]), 
            #     '{:.3f}'.format(e.Hillas["lengthS"]),
            #     '{:.3f}'.format(e.Hillas["lengthA"]),
            #     '{:.3f}'.format(e.Hillas["distN"]), #0.1206 -- convert from cm to degrees
            #     '{:.3f}'.format(e.Hillas["distS"]),
            #     '{:.3f}'.format(e.Hillas["distA"]),
            #     '{:.3f}'.format(e.Hillas["missN"]), 
            #     '{:.3f}'.format(e.Hillas["missS"]),
            #     '{:.3f}'.format(e.Hillas["missA"]),
            #     '{:.3f}'.format(e.Hillas["azwidthN"]), 
            #     '{:.3f}'.format(e.Hillas["azwidthS"]), 
            #     '{:.3f}'.format(e.Hillas["azwidthA"]), 
            #     '{:.3f}'.format(e.Hillas["alphaN"]), 
            #     '{:.3f}'.format(e.Hillas["alphaS"]), 
            #     '{:.3f}'.format(e.Hillas["alphaA"]), 
            #     sep="\t", file=fout)
            e.saveevent("", "", "DFdays/"+re.search("\d*", EXPOS).group(0))
            events_cleaned.append(e)
  

def cut(events, save = False, mode = "S"):
    marked_events = []
    for e in events:
        if len(e.Hillas) > 10: #исключаем необсчитанные события
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


marked = cut(events_cleaned)
if not re.search("\d*", EXPOS).group(0)  in os.listdir("../DFTA"):
    os.mkdir("../DFTA/"+re.search("\d*", EXPOS).group(0))
    for e in marked:
        e.saveevent("", "", "DFTA/"+re.search("\d*", EXPOS).group(0))
    
# plt.hist([e.Hillas["alphaS"] for e in events_cleaned])
# fout.close()
