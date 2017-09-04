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
        fits_files = [file for file in all_files if file[-5:]=='.fits'] #find fits files only

        date_obss = []
        dates = [] 
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

        # Loop through fits files and extract header info
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
    
            date_obs = hdr['DATE-OBS']
            date = date_obs[0:10]

            airmass = hdr['HIERARCH ESO TEL AIRM START']
            pwv = hdr['HIERARCH ESO TEL AMBI IWV START']
                        
            obsid = hdr['HIERARCH ESO OBS ID']
            
            if 'HIERARCH ESO INS GRAT1 WLEN' in hdr:
               waveset = hdr['HIERARCH ESO INS GRAT1 WLEN']
            else:
               waveset = -1
               
            ncycle =  hdr['HIERARCH ESO DET CHOP NCYCLES']

            date_obss.append(date_obs)
            dates.append(date)
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
                                  'date_obs':pd.Series(date_obss),
                                  'date':pd.Series(dates),
                                  'ncycle':pd.Series(ncycles)})

        return table
    
    # Convenience function to get obsids from data frame, no duplicates
    def get_obsids(self):
        obsids = self.df['obsid']
        return obsids.unique()
    
    # Convenience function to extract parts of data frame where category is "science" and
    # target is that specified by user
    def get_sciencelist(self,obsid, skip_first=True):
        scilist = self.df.loc[(self.df['obsid'] == obsid) & (self.df['category'] == 'SCIENCE')]
        if skip_first:
            return scilist[1:] #skip the first frame since these seem to have an electronic problem
        else:
            return scilist  

    # Convenience function to extract parts of data frame where category is "calib" and
    # target is that specified by user
    def get_caliblist(self,obsid,skip_first=True):
        scilist = self.df.loc[(self.df['obsid'] == obsid) & (self.df['category'] == 'CALIB')]
        if skip_first:
            return scilist[1:] #skip the first frame since these seem to have an electronic problem
        else:
            return scilist  
