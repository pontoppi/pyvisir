import warnings
import json
import os
import configparser as cp

import warnings

import numpy as np
import numpy.ma as ma
import astropy.io.fits as pf
from astropy.utils.exceptions import AstropyUserWarning
import scipy.fftpack as fp
from scipy.stats import tmean, tvar
from scipy.ndimage.filters import median_filter
from scipy import constants
import matplotlib.pylab as plt
import pyvisir.inpaint as inpaint
import utils.helpers as helpers

warnings.filterwarnings('ignore', category=AstropyUserWarning)

class Environment():
    '''
    Class to encapsulate global environment parameters.
    
    Parameters
    ----------
    config_file: string
        ConfigParser compliant file containing global parameters.
    
    Methods
    -------
    
    Attributes
    ----------
    pars: SafeConfigParser object
        Contains all global parameters
    
    '''
    def __init__(self,settings_file='visir.ini',detpars_file='detector.ini'):
        sys_dir = self._getSysPath()
        self.settings = cp.SafeConfigParser()
        self.settings.read(sys_dir+'/'+settings_file)
        self.detpars = cp.SafeConfigParser()
        self.detpars.read(sys_dir+'/'+detpars_file)
    
    def _getSysPath(self):
        sys_dir, this_filename = os.path.split(__file__)
        return sys_dir
    
    def getItems(self,option):
        return [self.settings.get(section,option) for section in self.settings.sections()]
    
    def getSections(self):
        return self.settings.sections()
    
    def getWaveRange(self,setting,onum):
        range_str = self.settings.get(setting,'wrange'+str(int(onum)))
        range = ranges[onum]
        return range
    
    def getOrderWidth(self,setting):
        orderw = self.settings.get(setting,'orderw')
        return json.loads(orderw)
    
    def getSpatialCenter(self,setting,onum):
        center_str = self.settings.get(setting,'scenters')
        centers = json.loads(center_str)
        return centers[onum]
    
    def getYRange(self,setting,onum):
        center = self.getSpatialCenter(setting,onum)
        orderw = self.getOrderWidth(setting)
        det = self.getDetPars()
        down = np.max([center-orderw,0])
        up = np.min([center+orderw,det['ny']-1])
        
        return (down,up)
    
    def getDispersion(self,setting,onum):
        w0_str = self.settings.get(setting,'wcenters')
        w0s = json.loads(w0_str)
        dw_str = self.settings.get(setting,'deltaws')
        dws = json.loads(dw_str)
        return {'w0':w0s[onum],'dw':dws[onum]}
    
    def getDetPars(self):
        gain = self.detpars.getfloat('Detector','gain')
        rn   = self.detpars.getfloat('Detector','rn')
        dc   = self.detpars.getfloat('Detector','dc')
        nx   = self.detpars.getint('Detector','nx')
        ny   = self.detpars.getint('Detector','ny')
        return {'gain':gain,'rn':rn,'dc':dc,'nx':nx,'ny':ny}
    
    def getNOrders(self,setting):
        return self.settings.getint(setting,'norders')

class Observation():
    '''
    Private object containing a TEXES observation - that is, all exposures related to a single type of activity.
    
    Any specific activity (Darks, Flats, Science, etc.) are modeled as classes derived off the Observation class.
    
    Parameters
    ----------
    filelist: List of strings
        List of data (.fits) files associated with the observation.
    type: string
        type of observation (e.g., Dark).
    
    Attributes
    ----------
    type
    Envi
    flist
    planes
    header
    
    Methods
    -------
    getSetting
    getNOrders
    getTargetName
    subtractFromStack
    divideInStack
    writeImage
    
    '''
    def __init__(self,filelist,type='image'):
        self.type = type
        self.Envi = Environment()
        self.flist = filelist
        
        self._openList()
        self._makeHeader()
        self.stack,self.ustack = self._getStack()
        
        self.nplanes = self.stack.shape[2]
        self.height = self.stack[:,:,0].shape[0]
    
    def _openList(self):
       warnings.resetwarnings()
       warnings.filterwarnings('ignore', category=UserWarning, append=True)
 
       self.cubes = []
       self.headers = []
        
       for index,row in self.flist.iterrows():
          cube = {'c11':pf.getdata(row['file'],1),
                  'c21':pf.getdata(row['file'],2), #currently, we just store the first two half cycles. 
                  'int':pf.getdata(row['file'],2*row['ncycle']+1)} #the last frame is the chop stack.
          self.cubes.append(cube)
        
       self.exp_time = self.getExpTime()
       self.det_pars = self.Envi.getDetPars()
    
    def getExpTime(self):
       exptime = self.flist['exptime'].iloc[0]
       return exptime
        
    def getSetting(self):
       waveset   = self.flist['waveset'].iloc[0]        
       wavesets   = self.Envi.getItems('waveset')
       gsub = [i for i,v in enumerate(wavesets) if float(v)==waveset]
       return self.Envi.getSections()[gsub[0]]
       
    def getAirmass(self):
       airmasses = self.flist['airmass']
       return airmasses
       
    def getObsID(self):
       obsid = self.flist['obsid'].iloc[0]
       return obsid
       
    def getNOrders(self):
       setting = self.getSetting()
       return self.Envi.getNOrders(setting)
    
    def getTargetName(self):
       target_name = self.flist['target'].iloc[0]
       return target_name.replace(" ", "")
    
    def _getStack(self):
       nexp = len(self.cubes)
       nx = self.det_pars['nx']
       ny = self.det_pars['ny']
        
       stack = np.zeros((ny,nx,nexp))
       ustack = np.zeros((ny,nx,nexp))
       for i,cube in enumerate(self.cubes):
          if self.flist['nodpos'].iloc[i]=='B':
             sign = -1
          else:
             sign = 1
             
          stack[:,:,i]  = cube['int']*self.det_pars['gain']*sign #convert everything to e-
          ustack[:,:,i] = 1.#self._error(cube[j,:,:])

       return stack,ustack
    
    def _error(self,data):
        var_data = np.abs(data*self.exp_pars['itime']*self.exp_pars['nreads']+ #assuming detector units is in e-/s
                          self.exp_pars['itime']*self.exp_pars['nreads']*self.det_pars['dc']+
                          self.det_pars['rn']**2/self.exp_pars['nreads'])
        return np.sqrt(var_data)
    
    def subtractFromStack(self,Obs):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
        try:
            self.uimage = np.sqrt(self.uimage**2+Obs.uimage**2)
            self.image -= Obs.image
        except:
            print('Subtraction failed - no image calculated')
    
    def divideInStack(self,Obs):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
        
        for i in np.arange(self.stack.shape[2]):
            plane = self.stack[:,:,i]
            uplane = self.ustack[:,:,i]
            uplane = np.sqrt((uplane/plane)**2+(Obs.uimage/Obs.image)**2)
            plane /= Obs.image
            uplane *= np.abs(plane)
            
            self.stack[:,:,i]  = plane
            self.ustack[:,:,i] = uplane
    
    def _collapseStack(self,stack=None,ustack=None,method='SigClip',sig=50.):
        '''
        If called without the stack keyword set, this will collapse the entire stack.
        However, the internal stack is overridden if a different stack is passed.
        For instance, this could be a stack of nod pairs.
        '''
        if stack is None:
            stack,ustack = self.stack,self.ustack
        
        #stack_median = np.median(stack,2)
        #stack_stddev = np.std(stack,2)
        #shape = stack.shape
        #masked_stack = ma.zeros(shape)
        masked_stack = ma.masked_invalid(stack)
        masked_ustack = ma.masked_invalid(ustack)
        image = ma.average(masked_stack,2,weights=1./masked_ustack**2)
        uimage = np.sqrt(ma.mean(masked_ustack**2,2)/ma.count(masked_ustack,2))
        
        return image, uimage
    
    def writeImage(self,filename=None):
        if filename is None:
            filename = self.type+'.fits'
        
        hdu = pf.PrimaryHDU(self.image.data)
        uhdu = pf.ImageHDU(self.uimage.data)
        hdulist = pf.HDUList([hdu,uhdu])
        hdulist.writeto(filename,clobber=True)
    
    def getKeyword(self,keyword):
        try:
            klist = [header[keyword] for header in self.headers]
            return klist
        except ValueError:
            print("Invalid header keyword")

class Flat(Observation):
    def __init__(self,filelist,dark=None,norm_thres=5000.,save=False):
        self.type = 'flat'
        self.Envi = Environment()
        self.flist = filelist
        self._openList()
        self.stack,self.ustack = self._getStack()
        self.nplanes = self.stack.shape[2]
        self.height = self.stack[:,:,0].shape[0]
        self.setting,self.echelle,self.crossdisp = self.getSetting()
        
        self.image,self.uimage = self._collapseStack()
        
        if dark:
            self.subtractFromStack(dark)
        
        self._normalize(norm_thres)
        
        #Where the flat field is undefined, it's set to 1 to avoid divide by zeros.
        self.image[np.where(self.image<0.1)] = 1
        
        if save:
            self.writeImage()
        
    def _normalize(self,norm_thres):
        flux = np.median(self.image[np.where(self.image>norm_thres)])
        self.image = self.image/flux
        self.uimage = self.uimage/flux
    
    def makeMask(self):
        return np.where(self.image<0.1,0,1)

class Dark(Observation):
    def __init__(self,filelist,save=False):
        self.type = 'dark'
        self.Envi = Environment()
        self.flist = filelist
        self._openList()
        self.stack,self.ustack = self._getStack()
        self.nplanes = self.stack.shape[2]
        self.height = self.stack[:,:,0].shape[0]
        
        self.image,self.uimage = self._collapseStack()
        self._badPixMap()
        if save:
            self.writeImage()
    
    def _badPixMap(self,clip=30,filename='badpix.dmp'):
        median = np.median(self.image)
        var  = tvar(self.image,(-100,100))
        self.badpix = ma.masked_greater(self.image-median,clip*np.sqrt(var))
        if filename is not None:
            self.badpix.dump(filename)

class Nod(Observation):
    def __init__(self,filelist,dark=None,flat=None,badpix=None):
        self.type = 'nod'
        self.Envi = Environment()
        self.flist = filelist
        
        self._openList()
        
        self.target = self.getTargetName()
        self.setting = self.getSetting()
        self.airmasses = self.getAirmass()
        self.obsid = self.getObsID()
        
        self.airmass = np.mean(self.airmasses)
        
        self.stack,self.ustack = self._getStack()
        self.height = self.stack[:,:,0].shape[0]
        if flat:
            self.divideInStack(flat)
        if badpix:
            badmask = np.load(badpix)
            self._correctBadPix(badmask)
        offsets = self._findYOffsets()
        
        self.stack   = self._yShift(offsets,self.stack)
        self.ustack  = self._yShift(offsets,self.ustack)
        
        self.image,self.uimage = self._collapseStack()
        self.writeImage()
        #An averaged sky frame is constructed using the straight stack collapse.
        self.stack,self.ustack = self._getStack()
        if flat:
            self.divideInStack(flat)
        if badpix:
            badmask = np.load(badpix)
            self._correctBadPix(badmask)
        
        offsets = [offset for offset in offsets for i in range(2)]
        self.stack   = self._yShift(offsets,self.stack)
        self.ustack  = self._yShift(offsets,self.ustack)
        
        self.sky,self.usky = self._collapseStack()
        self.sky -= np.absolute(self.image)/2.
        self.usky = np.sqrt(self.uimage**2/2.+self.usky**2)
    
    def _correctBadPix(self,badmask):
        for i in np.arange(self.stack.shape[2]):
            plane = self.stack[:,:,i]
            maskedImage = np.ma.array(plane, mask=badmask.mask)
            NANMask = maskedImage.filled(np.NaN)
            self.stack[:,:,i] = inpaint.replace_nans(NANMask, 5, 0.5, 1, 'idw')
            
            plane = self.ustack[:,:,i]
            maskedImage = np.ma.array(plane, mask=badmask.mask)
            NANMask = maskedImage.filled(np.NaN)
            self.ustack[:,:,i] = inpaint.replace_nans(NANMask, 5, 0.5, 1, 'idw')
    
    def _findYOffsets(self):
        
        kernel = np.median(self.stack[:,:,1],1)
        offsets = np.empty(0)
        nplanes = self.stack.shape[2]
        for i in np.arange(nplanes):
            profile = np.median(self.stack[:,:,i],1)
            cc = fp.ifft(fp.fft(kernel)*np.conj(fp.fft(profile)))
            cc_sh = fp.fftshift(cc)
            cen = helpers.calc_centroid(cc_sh).real - self.height/2.
            offsets = np.append(offsets,cen)
        return offsets
    
    def _findXOffsets(self):
        
        kernel = np.median(self.stack[:,:,0],0)
        offsets = np.empty(0)
        nplanes = self.stack.shape[2]
        for i in np.arange(nplanes):
            profile = np.median(self.stack[:,:,i],0)
            cc = fp.ifft(fp.fft(kernel)*np.conj(fp.fft(profile)))
            cc_sh = fp.fftshift(cc)
            cen = helpers.calc_centroid(cc_sh,cwidth=3.).real - self.height/2.
            offsets = np.append(offsets,cen)
        
        return offsets
    
    def _yShift(self,offsets,stack):
        
        sh = stack.shape
        internal_stack = np.zeros(sh)
        
        index = np.arange(sh[0])
        for plane in np.arange(sh[2]):
            for i in np.arange(sh[1]):
                col = np.interp(index-offsets[plane],index,stack[:,i,plane])
                internal_stack[:,i,plane] = col
        
        return internal_stack
    
    def _xShift(self,offsets,stack):
        
        sh = stack.shape
        internal_stack = np.zeros(sh)
        index = np.arange(sh[1])
        for plane in np.arange(sh[2]):
            for i in np.arange(sh[0]):
                row = np.interp(index-offsets[plane],index,stack[i,:,plane])
                internal_stack[i,:,plane] = row
        
        return internal_stack
    
    def _getPairs(self):
        As = np.arange(0,self.nnod*2-1,2,dtype=np.int16)
        Bs = np.arange(0,self.nnod*2-1,2,dtype=np.int16)+1
        pairs = [(A,B) for A,B in zip(As,Bs)]
        return pairs

