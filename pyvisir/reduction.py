import os
import pandas as pd
from pyvisir.observation import *
from pyvisir.order import *
from pyvisir.spec1d import *
from pyvisir.calspec import *
from pyvisir.combine_orders import *

import utils.helpers as helpers

class Reduction():
    '''
    Top level basic script for reducing an observation, consisting of a science target and a telluric standard, as well
    as associated calibration files. 
    '''
    def __init__(self, scilist, stdlist, level1=True,level2=True,level3=True,
                 level1_path='L1FILES',shift=0.0, dtau=0.0, flat=None,
                 doWaveFitPlot=False, doWaveFit=True, doTracePlot=False, 
                 doOffsets=True, **kwargs):

        self.scilist = scilist
        self.stdlist = stdlist

        self.flat = flat
        
        self.doWaveFit     = doWaveFit
        self.doWaveFitPlot = doWaveFitPlot
        self.doTracePlot    = doTracePlot
        self.doOffsets      = doOffsets

        self.shift       = shift
        self.dtau        = dtau
        self.level1_path = level1_path

        self.mode  = 'SciStd'
        self.tdict = {'science':self.scilist,'standard':self.stdlist}

        if level1:
            self._level1()

        if level2:
            self._level2()
            
        if level3:
            self._level3()
        
    def _level1(self):
        level1_files = {}

        # keys are: standard, science
        for key in sorted(self.tdict.keys()):
            '''
            Nod is in observation.py. ONod is a complex object, including ONod.image, ONod.sky, ONOd.target, ONod.flist
            ONod.image is main data product - summed A-B stack
            '''    
            ONod    = Nod(self.tdict[key],flat=self.flat,doOffsets=self.doOffsets)        
            # Returns number of orders (read from setting file)                                
            norders = ONod.getNOrders()   

            target_files = []
            for i in np.arange(norders):
                OOrder   = Order(ONod,onum=i,write_path=self.level1_path+'/SPEC2D',doTracePlot=self.doTracePlot)   #in order.py
                #Fits trace to each order; makes rectified image for each order; outputs to SPEC2D
                OSpec1D  = Spec1D(OOrder,sa=False,write_path=self.level1_path+'/SPEC1D',doWaveFitPlot=self.doWaveFitPlot,
                                  doFit=self.doWaveFit)  #in spec1d.py
                                  
                # Now we have: OSpec1D.flux, .wave, .sky, .uflux, .usky (and others)
                # Filenames look like: SPEC2D/object_ID_order0.fits, SPEC1D/object_ID_spec1d0.fits
                OOrder_files = {'2d':OOrder.file, '1d':OSpec1D.file}
                target_files.append(OOrder_files)

            level1_files[key] = target_files

        filename = self._getLevel1File() 
        f = open(self.level1_path+'/'+filename, 'w')
        json.dump(level1_files,f)
        f.close()

    def _level2(self):

        filename = self._getLevel1File()
        f = open(self.level1_path+'/'+filename, 'r')
        level1_files = json.load(f)
        f.close()

        norders = len(level1_files['science'])

        for i in np.arange(norders):
            sci_file = level1_files['science'][i]['1d']
            std_file = level1_files['standard'][i]['1d']
            OCalSpec = CalSpec(sci_file,std_file,shift=self.shift,dtau=self.dtau,write_path='CAL1D',order=i+1)

    def _level3(self):
        filename = self._getLevel1File()
        basename = filename[0:-11]
        all_files = os.listdir('CAL1D')
        level2_files = [os.path.join('CAL1D',file) for file in all_files if basename in file]

        OCombSpec = CombSpec(level2_files,write_path='COMBSPEC',micron=True)
        
    def _getLevel1File(self):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        hdulist=pf.open(self.tdict['science'].file.iloc[0],ignore_missing_end=True)
        header = hdulist[0].header
        hdulist.close()
        # Don't see sci_file attribute defined, so replaced line below with lines above
        # header = pf.open(self.sci_file[0],ignore_missing_end=True)[0].header
        basename = helpers.getBaseName(header)
        filename = basename+'_files.json'
        return filename
