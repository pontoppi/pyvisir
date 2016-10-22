import os
import pdb
import numpy as np
import astropy.io.fits as fits
import pandas as pd

class visir_run():
    def __init__(self,path):
        self.df = self.read_run(path)
        
    def read_run(self,path):
        all_files = os.listdir(path)
        fits_files = [file for file in all_files if file[-5:]=='.fits']

        cats = [] 
        targets = []
        airmasses = []
        exptimes = []
        nodpositions = []
        xoffs = []
        yoffs = []
        airmasses = []
        pwvs = []
        obsids = []
        wavesets = []
        fullpaths = []
        ncycles = []

        for fits_file in fits_files:
            fullpath = os.path.join(path,fits_file)
            hdr = fits.getheader(fullpath)
            cat = hdr['HIERARCH ESO DPR CATG']
            target = hdr['HIERARCH ESO OBS TARG NAME']
            exptime = hdr['HIERARCH ESO DET SEQ1 EXPTIME']
    
            if 'HIERARCH ESO SEQ NODPOS' in hdr:
                nodpos = hdr['HIERARCH ESO SEQ NODPOS']
            else:
                nodpos = 'N/A'
    
            if 'HIERARCH ESO SEQ CUMOFFSETX' in hdr:
                xoff = hdr['HIERARCH ESO SEQ CUMOFFSETX']
                yoff = hdr['HIERARCH ESO SEQ CUMOFFSETY']
            else:
                xoff = 0
                yoff = 0
    
            airmass = hdr['HIERARCH ESO TEL AIRM START']
            pwv = hdr['HIERARCH ESO TEL AIRM START']
                        
            obsid = hdr['HIERARCH ESO OBS ID']
            
            if 'HIERARCH ESO INS GRAT1 WLEN' in hdr:
               waveset = hdr['HIERARCH ESO INS GRAT1 WLEN']
            else:
               waveset = -1
               
            ncycle =  hdr['HIERARCH ESO DET CHOP NCYCLES']

            cats.append(cat)
            targets.append(target)
            nodpositions.append(nodpos)
            exptimes.append(exptime)
            xoffs.append(xoff)
            yoffs.append(yoff)
            airmasses.append(airmass)
            pwvs.append(pwv)
            obsids.append(obsid)
            wavesets.append(waveset)
            fullpaths.append(fullpath)
            ncycles.append(ncycle)

            table = pd.DataFrame({'obsid':pd.Series(obsids),
                                  'category':pd.Series(cats),
                                  'target':pd.Series(targets),
                                  'nodpos':pd.Series(nodpositions),
                                  'exptime':pd.Series(exptimes),
                                  'xoff':pd.Series(xoffs),
                                  'yoff':pd.Series(yoffs),
                                  'airmass':pd.Series(airmasses),
                                  'pwv':pd.Series(pwvs),
                                  'waveset':pd.Series(wavesets),
                                  'file':pd.Series(fullpaths),
                                  'ncycle':pd.Series(ncycles)})

        return table
    
    def get_obsids(self):
        obsids = self.df['obsid']
        return obsids.unique()
    
    def get_sciencelist(self,obsid):
        scilist = self.df.loc[(self.df['obsid'] == obsid) & (self.df['category'] == 'SCIENCE')]
        return scilist[1:] #skip the first frame since these seem to have an electronic problem
    
    def get_caliblist(self,obsid):
        scilist = self.df.loc[(self.df['obsid'] == obsid) & (self.df['category'] == 'CALIB')]
        return scilist[1:] #skip the first frame since these seem to have an electronic problem
    
