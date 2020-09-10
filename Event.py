import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
mpl.use('Agg')
#mpl.rc()
from copy import deepcopy
import gc

class Event():
    def __init__(self, Nevent = 0, eventtime = "12:34:56,789.101.112", clusters = None):
        if clusters is None:
            clusters = []
        self.clusters = dict()
        for cluster in clusters:
            self.clusters[cluster[0]] = cluster[1]
        self.Nclusters = len(clusters)
        self.Nevent = '{:06}'.format(Nevent)
        self.time = eventtime
        self.size = 0
        self.vmax = 0
        self.pixels = None
        
        self.xm = None
        self.ym = None
        self.x2m = None
        self.y2m = None
        self.xym = None
        self.Hillas = {"a": None, "b": None, "width": None, "length": None, "dis": None, "miss": None}
        
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
                        
                        if v > self.vmax: self.vmax = v

        return self
    
    def params(self):
        if None in self.Hillas.values() and self.size > 0:
            self.xm, self.ym, self.x2m, self.y2m, self.xym = 0, 0, 0, 0, 0
            xsum, x2sum, ysum, y2sum, xysum = 0, 0, 0, 0, 0
            for pixel in self.pixels:
                x, y, v = self.pixels[pixel]
                xsum += x * v
                x2sum += x * x * v
                ysum += y * v
                y2sum += y * y * v
                xysum += x * y * v
            self.xm = xsum / self.size
            self.ym = ysum / self.size
            self.x2m = x2sum / self.size
            self.y2m = y2sum / self.size
            self.xym = xysum / self.size
            sigmax = self.x2m - self.xm**2
            sigmay = self.y2m - self.ym**2
            sigmaxy = self.xym - self.xm*self.ym
            d = sigmay-sigmax

            a = self.Hillas["a"] = (d+np.sqrt(d*d+4*sigmaxy**2))/(2*sigmaxy)
            b = self.Hillas["b"] = self.ym-self.Hillas["a"]*self.xm
            self.Hillas["width"] = np.sqrt((sigmay-2*a*sigmaxy+a*a*sigmax)/(1+a*a))
            self.Hillas["length"] = np.sqrt((sigmax+2*a*sigmaxy+a*a*sigmay)/(1+a*a))
            self.Hillas["dis"] = np.sqrt(self.xm**2+self.ym**2)
            self.Hillas["miss"] = abs(b/np.sqrt(1+a*a))
            self.Hillas["size"] = self.size
            self.Hillas["coords"] = (self.xm, self.ym)
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
                    plt.scatter(pixel_coords[pixel][1], pixel_coords[pixel][2], color = "orange", alpha = alpha + (1-alpha)*self.pixels[pixel][2]/self.vmax)
                    ax.text(self.pixels[pixel][0], self.pixels[pixel][1], str(self.pixels[pixel][2]), fontsize = 10) 
        else:    
            for pixel in self.pixels:
                plt.scatter(self.pixels[pixel][0], self.pixels[pixel][1], color = "orange", alpha = alpha + (1-alpha)*self.pixels[pixel][2]/self.vmax)
                ax.text(self.pixels[pixel][0], self.pixels[pixel][1], str(self.pixels[pixel][2]), fontsize = 10) 
        if save:
            plt.savefig("results/"+self.Nevent+".png", dpi = 200)
            plt.close(fig)
    
    def clean(self, A = 14, B = 7):
        def dist(c1, c2):
            return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        b = deepcopy(self)
        b.size = 0
        for pixel in self.pixels:
            if self.pixels[pixel][2] < A:
                b.pixels.pop(pixel)
            else:
                n = False
                for pixel1 in self.pixels:
                    if dist(self.pixels[pixel], self.pixels[pixel1]) < 3.1 and self.pixels[pixel][2] > B:
                        n = True
                        b.size += b.pixels[pixel][2]
                        break
                if not n: b.pixels.pop(pixel)
        del self.clusters
        self.clusters = []    #убираем мусор
        if len(b.pixels): 
            try:
                 b.params()
            except RuntimeWarning:
                pass
        return b
