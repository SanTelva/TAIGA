import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
# import gc
import datetime
#mpl.use("Agg")

def delta(time1, time2):
    #12:34:56,789.101.112
    t1, t1m = time1.split(",")
    t1 = list(map(int, t1.split(":")))
    t1m = list(map(int, t1m.split(".")))
    t2, t2m = time2.split(",")
    t2 = list(map(int, t2.split(":")))
    t2m = list(map(int, t2m.split(".")))
    
    dt = [t2[i] - t1[i] for i in range(3)]
    dtm = [t2m[i] - t1m[i] for i in range(3)]
    
    dt = 3600*dt[0]+60*dt[1]+dt[2]
    dtm = (1000000*dtm[0]+1000*dtm[1]+dtm[2])*10**(-9)
    return abs(dt+dtm)

def time(timestamp):
    value = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    milli, micro = value.strftime('%f')[0:3], value.strftime('%f')[2:-1]
    return value.strftime('%H:%M:%S,')+".".join([milli, micro, "000"])

def timestamp(eventtime, date = datetime.date(2019, 11, 23)):
    t, m = eventtime.split(",")
    datestr = datetime.date.strftime(date, "%d.%m.%Y ")
    m = ','+''.join(m.split(".")[:-1])
    result = datetime.datetime.timestamp(datetime.datetime.strptime(datestr + t + m, "%d.%m.%Y %H:%M:%S,%f"))
    return result

class Event():
    def __init__(self, Nevent = 0, eventtime = "12:34:56,789.101.112", clusters = None, pixels = None, source_x = 0, source_y = 0):
        if clusters is None:
            clusters = []
        self.clusters = dict()
        for cluster in clusters:
            self.clusters[cluster[0]] = cluster[1]
        self.Nclusters = len(clusters)
        self.Nevent = '{:06}'.format(Nevent)
        self.time = eventtime
        self.size = 0
        self.vmax1 = 0
        self.vmax2 = 0
        self.pixels = deepcopy(pixels)
        self.source_x = source_x
        self.source_y = source_y
        # self.xm = None
        # self.ym = None
        # self.x2m = None
        # self.y2m = None
        # self.xym = None
        self.Hillas = {"widthN": None, "lengthN": None, "distN": None, "missN": None, "alphaN": None}
        
    def __str__(self):
        return "#"+self.Nevent+'  '+self.time
    def __repr__(self):
        return "#"+self.Nevent+'  '+self.time
    def __len__(self):
        return len(self.pixels)

    
    def recount(self, factors, coords):
        self.pixels = dict()
        
        for cluster in self.clusters:
            #print(self.clusters[cluster])
            for channel in range(64):
                if self.clusters[cluster][channel][0] > 0:
                    x = coords[cluster - 1][channel][0]
                    y = coords[cluster - 1][channel][1]
                    n = coords[cluster - 1][channel][2]
                    if factors[cluster][channel] is not None:
                        v = int(round(self.clusters[cluster][channel][0] / factors[cluster][channel]))
                    else:
                        v = None
                    
                    if x is not None and y is not None and v is not None:
                        self.pixels[n]=(x, y, v)
                        self.size += v
                        if v > self.vmax1:
                            self.vmax1 = v
                        elif v > self.vmax2:
                            self.vmax2 = v
        del self.clusters
        self.clusters = []

    
    
    def params(self, source_x=0, source_y=0, angles = True):
        '''
        N = norm, source_x=0, source_y=0.
        Source is just above IACT, without shift.
        
        S = source, all Hillas parametres have to be recounted
        relatively source_x, source_y
        
        A = anti-source, HP have to be recounted
        relatively -source_x, -source_y
        Parameters
        ----------
        source_x : TYPE, optional
            DESCRIPTION. The default is 0.
        source_y : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns dictionary of Hillas parameters for this event, False if size == 0

        '''
        source_x, source_y = self.source_x, self.source_y
        for pixel in self.pixels:
            x, y, v = self.pixels[pixel]
            if v > self.vmax1:
                self.vmax1 = v
            elif v > self.vmax2:
                self.vmax2 = v
        if None in self.Hillas.values() and self.size > 0:
            self.xNm, self.yNm, self.xN2m, self.yN2m, self.xyNm = 0, 0, 0, 0, 0
            self.con2 = (self.vmax1+self.vmax2)/self.size
            self.xSm, self.ySm, self.xS2m, self.yS2m, self.xySm = 0, 0, 0, 0, 0
            self.xAm, self.yAm, self.xA2m, self.yA2m, self.xyAm = 0, 0, 0, 0, 0
            xSsum, xS2sum, ySsum, yS2sum, xySsum = 0, 0, 0, 0, 0
            xAsum, xA2sum, yAsum, yA2sum, xyAsum = 0, 0, 0, 0, 0
            xNsum, xN2sum, yNsum, yN2sum, xyNsum = 0, 0, 0, 0, 0
            
            for pixel in self.pixels:
                x, y, v = self.pixels[pixel]
                # x -= source_x
                # y -= source_y
                # self.pixels[pixel] = (x, y, v)
                xNsum += x * v
                xN2sum += x * x * v
                yNsum += y * v
                yN2sum += y * y * v
                xyNsum += x * y * v
                if source_x or source_y:
                    xS = x - source_x
                    yS = y - source_y
                    
                    xA = x + source_x
                    yA = y + source_y
                    
                    xSsum += xS * v
                    xS2sum += xS * xS * v
                    ySsum += yS * v
                    yS2sum += yS * yS * v
                    xySsum += xS * yS * v
                    
                    xAsum += xA * v
                    xA2sum += xA * xA * v
                    yAsum += yA * v
                    yA2sum += yA * yA * v
                    xyAsum += xA * yA * v       
            self.xNm = xNsum / self.size
            self.yNm = yNsum / self.size
            self.xN2m = xN2sum / self.size
            self.yN2m = yN2sum / self.size
            self.xyNm = xyNsum / self.size
            
            self.xSm = xSsum / self.size
            self.ySm = ySsum / self.size
            self.xS2m = xS2sum / self.size
            self.yS2m = yS2sum / self.size
            self.xySm = xySsum / self.size
            
            self.xAm = xAsum / self.size
            self.yAm = yAsum / self.size
            self.xA2m = xA2sum / self.size
            self.yA2m = yA2sum / self.size
            self.xyAm = xyAsum / self.size
            
            sigmaxN = self.xN2m - self.xNm**2
            sigmayN = self.yN2m - self.yNm**2
            sigmaxyN = self.xyNm - self.xNm*self.yNm
            dN = sigmayN-sigmaxN
            zN = (dN**2+4*sigmaxyN**2)**0.5
            uN = 1+dN/zN
            vN = 2-uN
            
            if source_x or source_y:
                sigmaxS = self.xS2m - self.xSm**2
                sigmayS = self.yS2m - self.ySm**2
                sigmaxyS = self.xySm - self.xSm*self.ySm
                dS = sigmayS-sigmaxS
                zS = (dS**2+4*sigmaxyS**2)**0.5
                uS = 1+dS/zS
                vS = 2-uS
                
                sigmaxA = self.xA2m - self.xAm**2
                sigmayA = self.yA2m - self.yAm**2
                sigmaxyA = self.xyAm - self.xAm*self.yAm
                dA = sigmayA-sigmaxA
                zA = (dA**2+4*sigmaxyA**2)**0.5
                uA = 1+dA/zA
                vA = 2-uA
            
            #a = self.Hillas["a"] = (d+np.sqrt(d*d+4*sigmaxy**2))/(2*sigmaxy)
            # b = self.Hillas["b"] = self.ym-self.Hillas["a"]*self.xm
            # self.Hillas["width"] = np.sqrt((sigmay-2*a*sigmaxy+a*a*sigmax)/(1+a*a))
            # self.Hillas["length"] = np.sqrt((sigmax+2*a*sigmaxy+a*a*sigmay)/(1+a*a))
            # self.Hillas["dis"] = np.sqrt(self.xm**2+self.ym**2)
            # self.Hillas["miss"] = abs(b/np.sqrt(1+a*a))
            self.Hillas["size"] = self.size
            if sigmaxN+sigmayN < zN:
                return False
            self.Hillas["widthN"] = np.sqrt(((sigmaxN+sigmayN-zN)/2))
            self.Hillas["lengthN"] = ((sigmaxN+sigmayN+zN)/2)**0.5
            self.Hillas["distN"] = np.sqrt(self.xNm**2+self.yNm**2)
            self.Hillas["missN"] = np.sqrt((uN*self.xNm**2+vN*self.yNm**2)/2
                                           -(2*sigmaxyN*self.xNm*self.yNm/zN))
            self.Hillas["azwidthN"] = np.sqrt(self.xNm**2*self.yN2m
                                             -2*self.xNm*self.yNm
                                             +self.xN2m*self.yNm**2)/self.Hillas["distN"]
            self.Hillas["coordsN"] = (self.xNm, self.yNm)
            self.Hillas["alphaN"] = np.degrees(np.arcsin(self.Hillas["missN"]/self.Hillas["distN"]))
            
            if source_x or source_y:
                self.Hillas["widthS"] = ((sigmaxS+sigmayS-zS)/2)**0.5
                self.Hillas["widthA"] = ((sigmaxA+sigmayA-zA)/2)**0.5
                self.Hillas["lengthS"] = ((sigmaxS+sigmayS+zS)/2)**0.5
                self.Hillas["distS"] = np.sqrt(self.xSm**2+self.ySm**2)
                self.Hillas["missS"] = np.sqrt((uS*self.xSm**2+vS*self.ySm**2)/2
                                               -(2*sigmaxyS*self.xSm*self.ySm/zS))
                self.Hillas["azwidthS"] = np.sqrt(self.xSm**2*self.yS2m
                                                 -2*self.xSm*self.ySm
                                                 +self.xS2m*self.ySm**2)/self.Hillas["distS"]
                self.Hillas["coordsS"] = (self.xSm, self.ySm)
                self.Hillas["alphaS"] = np.degrees(np.arcsin(self.Hillas["missS"]/self.Hillas["distS"]))
                
                
                self.Hillas["lengthA"] = ((sigmaxA+sigmayA+zA)/2)**0.5
                self.Hillas["distA"] = np.sqrt(self.xAm**2+self.yAm**2)
                self.Hillas["missA"] = np.sqrt((uA*self.xAm**2+vA*self.yAm**2)/2
                                               -(2*sigmaxyA*self.xAm*self.yAm/zA))
                self.Hillas["azwidthA"] = np.sqrt(self.xAm**2*self.yA2m
                                                 -2*self.xAm*self.yAm
                                                 +self.xA2m*self.yAm**2)/self.Hillas["distA"]
                self.Hillas["coordsA"] = (self.xAm, self.yAm)
                self.Hillas["alphaA"] = np.degrees(np.arcsin(self.Hillas["missA"]/self.Hillas["distA"]))
            else:
                self.xSm = self.xAm = self.xNm
                self.ySm = self.yAm = self.yNm
                self.Hillas["widthS"] = self.Hillas["widthA"] = self.Hillas["widthN"]
                self.Hillas["lengthS"] = self.Hillas["lengthA"] = self.Hillas["lengthN"]
                self.Hillas["distS"] = self.Hillas["distA"] = self.Hillas["distN"]
                self.Hillas["missS"] = self.Hillas["missA"] = self.Hillas["missN"]
                self.Hillas["azwidthS"] = self.Hillas["azwidthA"] = self.Hillas["azwidthN"]
                self.Hillas["coordsS"] = self.Hillas["coordsA"] = self.Hillas["coordsN"]
                self.Hillas["alphaS"] = self.Hillas["alphaA"] = self.Hillas["alphaN"]
                
            if angles:
                self.Hillas["widthN"] *= 0.1206
                self.Hillas["lengthN"] *= 0.1206
                self.Hillas["distN"]*= 0.1206
                self.Hillas["missN"] *= 0.1206
                self.Hillas["azwidthN"] *= 0.1206
                self.Hillas["coordsN"] = (0.1206 * self.xNm, 0.1206*self.yNm)
                
                self.Hillas["widthS"] *= 0.1206
                self.Hillas["lengthS"] *= 0.1206
                self.Hillas["distS"] *= 0.1206
                self.Hillas["missS"] *= 0.1206
                self.Hillas["azwidthS"] *= 0.1206
                self.Hillas["coordsS"] = (0.1206*self.xSm, 0.1206*self.ySm)
                
                self.Hillas["widthA"] *= 0.1206
                self.Hillas["lengthA"] *= 0.1206
                self.Hillas["distA"] *= 0.1206
                self.Hillas["missA"] *= 0.1206
                self.Hillas["azwidthA"] *= 0.1206
                self.Hillas["coordsA"] = (0.1206*self.xAm, 0.1206*self.yAm)

            return self.Hillas
        elif self.size == 0:
            return False
        else:
            return self.Hillas
    def vizualize(self, colors = ['r', 'y', 'g', 'c', 'b', 'm', 'k'], alpha = 0.3, pixel_coords = None, save = False):
        #alpha = 0.3 #параметр для нормирования прозрачности
        fig, ax = plt.subplots(figsize=(7,7))
        print(self.Nevent)
        ax.set_title("#"+self.Nevent+"   "+str(self.time), fontsize=16)
        ax.set_xlim([-40, 40])
        ax.set_ylim([-40, 40])
        
        if pixel_coords is not None:
            for pixel in pixel_coords:
                if pixel not in self.pixels:
                    plt.scatter(pixel_coords[pixel][1], pixel_coords[pixel][2], color = colors[(pixel_coords[pixel][0]-1)%7], alpha = 0.1)
                else:
                    plt.scatter(pixel_coords[pixel][1], pixel_coords[pixel][2], color = "orange", alpha = alpha + (1-alpha)*self.pixels[pixel][2]/self.vmax1)
                    ax.text(self.pixels[pixel][0], self.pixels[pixel][1], str(self.pixels[pixel][2]), fontsize = 10) 
        else:    
            for pixel in self.pixels:
                plt.scatter(self.pixels[pixel][0], self.pixels[pixel][1], color = "orange", alpha = alpha + (1-alpha)*self.pixels[pixel][2]/self.vmax1)
                ax.text(self.pixels[pixel][0], self.pixels[pixel][1], str(self.pixels[pixel][2]), fontsize = 10) 
        if save:
            plt.savefig("results/pics/"+self.Nevent+".png", dpi = 200)
            plt.close(fig)
    
    # def clean(self, A = 14, B = 7):
    #     b = Event(int(self.Nevent), self.time, clusters=[], pixels=self.pixels)
    #     for pixel in self.pixels:
    #         if self.pixels[pixel][2] < B:
    #             b.pixels.pop(pixel)
    #         else:
    #             n = False
    #             if self.pixels[pixel][2] < A:
    #                  for pixel1 in self.pixels:
    #                      if pixel1 not in b.pixels or pixel == pixel1: pass
    #                      elif dist(self.pixels[pixel], self.pixels[pixel1]) < 3.1 and self.pixels[pixel1][2] >= A:
    #                          n = True
    #                          break
    #                  if not n: b.pixels.pop(pixel)
    #             else:
    #                 for pixel1 in self.pixels:
    #                     if pixel1 not in b.pixels or pixel == pixel1: pass #если пиксель уже прогнали
    #                     elif dist(self.pixels[pixel], self.pixels[pixel1]) < 3.1 and self.pixels[pixel1][2] >= B:
    #                         n = True
    #                         break
    #                 if not n: b.pixels.pop(pixel)
    #     b.size = 0
    #     b.vmax1 = 0
    #     for pixel in b.pixels:
    #         b.size += b.pixels[pixel][2]
    #         if b.pixels[pixel][2] > b.vmax1:
    #             b.vmax1 = b.pixels[pixel][2]
    #     if len(b.pixels) >= 4:
    #         try:
    #              b.params()
    #         except RuntimeWarning:
    #             pass
    #     return b
    def saveevent(self, EXPOS, path="/events/"):
        fout = open("../"+EXPOS+ path + self.Nevent+".event", "w")
        print(self.Nevent, self.time, sep="\t", file=fout)
        print(len(self.pixels), file=fout)
        print("Source_x\t", self.source_x, file = fout)
        print("Source_y\t", self.source_y, file = fout)
        for pixel in self.pixels:
            print(pixel, self.pixels[pixel][0], self.pixels[pixel][1], self.pixels[pixel][2], sep='\t', file=fout)
        if None not in self.Hillas.values():
                print( 
                'Size:\t', self.size, "\n",
                'con2:\t{:.3f}\n'.format(self.con2),
                '<xN>\t{:.3f}\n'.format(self.Hillas["coordsN"][0]),
                '<xS>\t{:.3f}\n'.format(self.Hillas["coordsS"][0]),
                '<xA>\t{:.3f}\n'.format(self.Hillas["coordsA"][0]),
                '<yN>\t{:.3f}\n'.format(self.Hillas["coordsN"][1]),
                '<yS>\t{:.3f}\n'.format(self.Hillas["coordsS"][1]),
                '<yA>\t{:.3f}\n'.format(self.Hillas["coordsA"][1]),
                'widthN\t{:.3f}\n'.format(self.Hillas["widthN"]),
                'widthS\t{:.3f}\n'.format(self.Hillas["widthS"]),
                'widthA\t{:.3f}\n'.format(self.Hillas["widthA"]),
                'lengthN\t{:.3f}\n'.format(self.Hillas["lengthN"]), 
                'lengthS\t{:.3f}\n'.format(self.Hillas["lengthS"]),
                'lengthA\t{:.3f}\n'.format(self.Hillas["lengthA"]),
                'distN\t{:.3f}\n'.format(self.Hillas["distN"]), #0.1206 -- convert from cm to degrees
                'distS\t{:.3f}\n'.format(self.Hillas["distS"]),
                'distA\t{:.3f}\n'.format(self.Hillas["distA"]),
                'missN\t{:.3f}\n'.format(self.Hillas["missN"]), 
                'missS\t{:.3f}\n'.format(self.Hillas["missS"]),
                'missA\t{:.3f}\n'.format(self.Hillas["missA"]),
                'azwidthN\t{:.3f}\n'.format(self.Hillas["azwidthN"]), 
                'azwidthS\t{:.3f}\n'.format(self.Hillas["azwidthS"]), 
                'azwidthA\t{:.3f}\n'.format(self.Hillas["azwidthA"]), 
                'alphaN\t{:.3f}\n'.format(self.Hillas["alphaN"]), 
                'alphaS\t{:.3f}\n'.format(self.Hillas["alphaS"]), 
                'alphaA\t{:.3f}\n'.format(self.Hillas["alphaA"]), 
                file=fout)
        fout.close()
    def cclean(self, neighbours, A = 14, B = 7):
        b = Event(int(self.Nevent), self.time, clusters=[], pixels=self.pixels)
        for pixel in self.pixels:
            if self.pixels[pixel][2] < B:
                b.pixels.pop(pixel)
            else:
                n = False
                if self.pixels[pixel][2] < A:
                     for pixel1 in neighbours[pixel]:
                         if pixel1 in self.pixels and self.pixels[pixel1][2] >= A:
                             n = True
                             break
                     if not n: b.pixels.pop(pixel)
                else:
                    for pixel1 in neighbours[pixel]:
                        if pixel1 not in b.pixels: pass #если пиксель уже прогнали
                        elif self.pixels[pixel1][2] >= B:
                            n = True
                            break
                    if not n: b.pixels.pop(pixel)
        b.size = 0
        b.vmax1 = 0
        b.vmax2 = 0
        for pixel in b.pixels:
            b.size += b.pixels[pixel][2]
            if b.pixels[pixel][2] > b.vmax1:
                b.vmax1 = b.pixels[pixel][2]
            elif b.pixels[pixel][2] > b.vmax2:
                b.vmax2 = b.pixels[pixel][2]
        return b


def readevent(filename):
    fin = open(filename, "r")
    Nevent, time = fin.readline().split("\t")
    n = int(fin.readline())
    source_x = float(fin.readline().split("\t")[1])
    source_y = float(fin.readline().split("\t")[1])
    pixels = dict()
    for i in range(n):
        pixel, x, y, v = map(float, fin.readline().split())
        pixel = int(pixel)
        v = int(v)
        pixels[pixel] = (x, y, v)
        
    e = Event(int(Nevent), time, clusters=None, pixels = pixels, source_x = source_x, source_y = source_y)
    e.size = int(fin.readline().split("\t")[1])
    e.con2 = float(fin.readline().split("\t")[1])
    try:
        e.Hillas["coordsN"] = [0] * 2
        e.Hillas["coordsS"] = [0] * 2
        e.Hillas["coordsA"] = [0] * 2
        e.Hillas["coordsN"][0] = float(fin.readline().split("\t")[1])
        e.Hillas["coordsS"][0] = float(fin.readline().split("\t")[1])
        e.Hillas["coordsA"][0] = float(fin.readline().split("\t")[1])
        e.Hillas["coordsN"][1] = float(fin.readline().split("\t")[1])
        e.Hillas["coordsS"][1] = float(fin.readline().split("\t")[1])
        e.Hillas["coordsA"][1] = float(fin.readline().split("\t")[1])
    
        for i in range(18):
            key, param = fin.readline().split("\t")
            e.Hillas[key.strip()]=float(param)
            e.Hillas["widthN"] = None
            e.params()
    except ValueError:
        print("Problems with", filename)
        fin.close()
        raise ValueError
        
    fin.close()
    return e