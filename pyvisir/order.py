import warnings
import json
import os
import configparser as cp
import pdb as pdb

import numpy as np
import numpy.ma as ma
import astropy.io.fits as pf
import scipy.fftpack as fp
from scipy.stats import tmean, tvar
from scipy.ndimage.filters import median_filter
from scipy import constants
from scipy import interpolate as ip
import matplotlib.pylab as plt
import pyvisir.inpaint as inpaint
import utils.helpers as helpers

class Order():
    def __init__(self,Nod,onum=1,write_path=None,doTracePlot=False):
        self.type = 'order'
        self.flist = Nod.flist
        self.airmass = Nod.airmass
        self.target = Nod.target
        self.obsid = Nod.obsid
        self.date = Nod.date

        self.Envi    = Nod.Envi
        self.onum    = onum
        self.setting = Nod.setting

        self.doTracePlot = doTracePlot

        self.image = Nod.image
        self.uimage = Nod.uimage
        self.sh = self.image.shape

        self.yrange = self.Envi.getYRange(self.setting,onum)
        self.image = Nod.image[self.yrange[0]:self.yrange[1],:]
        self.uimage = Nod.uimage[self.yrange[0]:self.yrange[1],:]
        self.sky = Nod.sky[self.yrange[0]:self.yrange[1],:]
        self.usky = Nod.usky[self.yrange[0]:self.yrange[1],:]
        self.sh = self.image.shape

        self._subMedian()
        yrs,traces = self.fitTrace(cwidth=30.,porder=2,pad=False,doTracePlot=self.doTracePlot)
        # Traces is the polynomial fit to the order
        self.image_rect,self.uimage_rect = self.yRectify(self.image,self.uimage,yrs,traces)
                
        self.sky_rect,self.usky_rect = self.yRectify(self.sky,self.usky,yrs,traces)
        # Now rectangularly rectified images of each order  

        if write_path:
            self.file = self.writeImage(path=write_path)

    def _cullEdges(self):
        orderw = self.Envi.getOrderWidth(self.setting)
        fullw = self.sh[1]
        self.image_rect[:,:(fullw-orderw)/2] = 0.
        self.image_rect[:,-(fullw-orderw)/2:] = 0.        
        self.sky_rect[:,:(fullw-orderw)/2] = 0.
        self.sky_rect[:,-(fullw-orderw)/2:] = 0.        
            
    def fitTrace(self,kwidth=10,porder=3,cwidth=30,pad=False,doTracePlot=False):
        sh = self.sh     # Dimensions of image (ny,nx)
        yr1 = (0,sh[0])  # (0, ny)
        yrs = [yr1]

        polys = []
        for yr in yrs:
            yindex = np.arange(yr[0],yr[1])      # Array from 0 to ny-1
            kernel = np.median(self.image[yindex,int(sh[1]/2-kwidth):int(sh[1]/2+kwidth)],1)
            centroids = []

            for i in np.arange(sh[1]):
                col = self.image[yindex,i]   # Extract a single column
                col_med = np.median(col)
                    
                cc = fp.ifft(fp.fft(kernel)*np.conj(fp.fft(col-col_med)))
                cc_sh = fp.fftshift(cc)
                centroid = helpers.calc_centroid(cc_sh,cwidth=cwidth).real - yindex.shape[0]/2.

                centroids.append(centroid)

            centroids = np.array(centroids)
        
            xindex = np.arange(sh[1])
            gsubs = np.where((np.isnan(centroids)==False) & (xindex>50) & (xindex<sh[1]-50) &
                             (centroids<15) & (centroids>-15))
            
            centroids[gsubs] = median_filter(centroids[gsubs],size=5)
            coeffs = np.polyfit(xindex[gsubs],centroids[gsubs],porder)

            poly = np.poly1d(coeffs)
            polys.append(poly)
            
            if(doTracePlot):
                trace_y=poly(xindex)+np.argmax(kernel)
                fig=plt.figure()
                ax1=fig.add_subplot(111)
                ax1.imshow(self.image)
                ax1.plot(xindex, trace_y, linestyle='--')
                ax1.set_xlim(0,np.shape(self.image)[1]) 
                ax1.set_ylim(np.shape(self.image)[0],0) 
                plt.show()

        return yrs,polys

    def yRectify(self,image,uimage,yrs,traces):
        
        sh = self.sh
        image_rect = np.zeros(sh)
        uimage_rect = np.zeros(sh)
        
        for yr,trace in zip(yrs,traces):
            index = np.arange(yr[0],yr[1])
            for i in np.arange(sh[1]):
                col = ip.interp1d(index,image[index,i],bounds_error=False,fill_value=0)
                image_rect[index,i] = col(index-trace(i))
                col = ip.interp1d(index,uimage[index,i],bounds_error=False,fill_value=1e10)
                uimage_rect[index,i] = col(index-trace(i))

        return image_rect,uimage_rect

    def xRectify(self,image,uimage,xrs,traces):
        
        sh = self.sh
        image_rect = np.zeros(sh)
        uimage_rect = np.zeros(sh)
        
        for xr,trace in zip(xrs,traces):
            index = np.arange(xr[0],xr[1])
            for i in np.arange(sh[0]):
                row = ip.interp1d(index,image[i,index],bounds_error=False,fill_value=0)
                image_rect[i,index] = row(index-trace(i))
                row = ip.interp1d(index,uimage[i,index],bounds_error=False,fill_value=1e10)
                uimage_rect[i,index] = row(index-trace(i))

        return image_rect,uimage_rect
 
                
    def _subMedian(self):
        self.image = self.image-np.median(self.image,axis=0)
            
    def writeImage(self,filename=None,path='.'):
        date   = self.date.replace('-','')
        filename = path+'/'+self.target+'_'+str(self.obsid)+'_'+str(date)+'_order'+str(self.onum)+'.fits'

        hdu  = pf.PrimaryHDU(self.image_rect)
        uhdu = pf.ImageHDU(self.uimage_rect)
        sky_hdu = pf.ImageHDU(self.sky_rect)
        usky_hdu = pf.ImageHDU(self.usky_rect)

        hdu.header['SETNAME'] = (self.setting, 'Setting name')
        hdu.header['ORDER'] = (str(self.onum),'Order number')

        hdulist = pf.HDUList([hdu,uhdu,sky_hdu,usky_hdu])

        hdulist.writeto(filename,overwrite=True)

        return filename
