import numpy as np
import numpy.ma as ma
import astropy.io.fits as pf
from scipy import constants
import pdb as pdb
import pickle as pickle
import matplotlib as matplotlib
import matplotlib.pylab as plt
#from cvs.cvsstats import lm  # not sure what this dependency is


class Spec1D():
    def __init__(self,Order,sa=False,write_path=None,doFit=True,doWaveFitPlot=False):
        self.Order   = Order
        self.setting = Order.setting
        self.target  = Order.target
        self.obsid   = Order.obsid
        self.date    = Order.date
        self.airmass = Order.airmass
        self.onum    = Order.onum
        self.Envi    = Order.Envi
        self.sh      = Order.sh[1]
        self.sa      = sa

        # Returns PSF
        PSF = self.getPSF()  

        # getDispersion in observation.py
        # Returns {w0, dw} for requested setting, order
        self.disp = self.Envi.getDispersion(self.setting,self.onum)

        self.wave = self.waveGuess(self.disp)  #Returns wavelength for each pix

        # Extract spectra
        self.flux,self.uflux,self.sky,self.usky = self.extract(PSF)

        #Fit wavelength solution, if desired        
        if(doFit==True):
            self.wave = self.waveFit(self.disp, doPlot=doWaveFitPlot)
        
        # Do SA if requested
        if sa:
            self.sa_pos,self.usa_pos,self.sa_neg,self.usa_neg = self.SpecAst(PSF)

        # Write out data
        if write_path:
            self.file = self.writeSpec(path=write_path)

    def getPSF(self,range=None):
        if range is None:
            range = (0,self.Order.sh[0])
        PSF = np.median(self.Order.image_rect[range[0]:range[1],:],1)
        npsf = PSF.size
        PSF_norm = PSF/np.abs(PSF).sum()
        return PSF_norm
        
    def extract(self,PSF):
        # Placeholder
        pixsig = 1.
        sh   = self.sh
        npsf = PSF.size

        # initialize arrays
        flux  = np.zeros(sh)
        uflux = np.zeros(sh)
        sky1d  = np.zeros(sh)
        usky1d  = np.zeros(sh)
        
        # Get rectified images
        im = self.Order.image_rect
        uim = self.Order.uimage_rect
        sky = self.Order.sky_rect
        usky = self.Order.usky_rect
        
        # Extract spectra
        for i in np.arange(sh):
            flux[i] = (PSF*im[:,i]/uim[:,i]**2).sum() / (PSF**2/uim[:,i]**2).sum()
            uflux[i] = np.sqrt(1.0/(PSF**2/uim[:,i]**2.).sum())
            
            sky1d[i] = (np.abs(PSF)*sky[:,i]).sum() / (PSF**2).sum()
            usky1d[i] = np.sqrt(1.0/(PSF**2/usky[:,i]**2.).sum())

        # apply masks
        flux = ma.masked_invalid(flux)
        flux = ma.filled(flux,1.)
        uflux = ma.masked_invalid(uflux)
        uflux = ma.filled(uflux,1000.)
        sky1d = ma.masked_invalid(sky1d)
        sky1d = ma.filled(sky1d,1.)
        usky1d = ma.masked_invalid(usky1d)
        usky1d = ma.filled(usky1d,1000.)
        sky_cont = self._fitCont(self.wave,sky1d)
        return flux,uflux,sky1d-sky_cont,usky1d
        
    def _fitCont(self,wave,spec):
        bg_temp = 210. #K
        
        niter = 2
        
        cont = self.bb(wave*1e-6,bg_temp)
        gsubs = np.where(np.isfinite(spec))
        for i in range(niter):
            norm = np.median(spec[gsubs])
            norm_cont = np.median(cont[gsubs])
            cont *= norm/norm_cont 
            gsubs = np.where(spec<cont)

        return cont
        
    def bb(self,wave,T):
        cc = constants.c
        hh = constants.h
        kk = constants.k
        
        blambda = 2.*hh*cc**2/(wave**5*(np.exp(hh*cc/(wave*kk*T))-1.))
        
        return blambda
        

    def SpecAst(self,PSF,method='centroid',width=5):
        '''
        The uncertainty on the centroid is:

                 SUM_j([j*SUM_i(F_i)-SUM_i(i*F_i)]^2 * s(F_j)^2)
        s(C)^2 = ------------------------------------------------
                                [SUM_i(F_i)]^4

        
        '''
        
        # Guesstimated placeholder
        aper_corr = 1.4
        posloc = np.argmax(PSF)
        negloc = np.argmin(PSF)

        sa_pos = np.zeros(self.sh)
        sa_neg = np.zeros(self.sh)

        usa_pos = np.zeros(self.sh)
        usa_neg = np.zeros(self.sh)

        im = self.Order.image_rect
        uim = self.Order.uimage_rect

        for i in np.arange(self.sh):
            index = np.arange(width*2+1)-width

            # First calculate SUM_i(F_i)
            F_pos = (im[posloc-width:posloc+width+1,i]).sum() 
            F_neg = (im[negloc-width:negloc+width+1,i]).sum()

            # then SUM_i(i*F_i)
            iF_pos = (index*im[posloc-width:posloc+width+1,i]).sum()
            iF_neg = (index*im[negloc-width:negloc+width+1,i]).sum()

            sa_pos[i] = iF_pos/F_pos
            sa_neg[i] = iF_neg/F_neg
       
            # Now propagate the error
            uF_pos = uim[posloc-width:posloc+width+1,i]
            uF_neg = uim[negloc-width:negloc+width+1,i]
            usa_pos[i]  = np.sqrt(((index*F_pos - iF_pos)**2 * uF_pos**2).sum())/F_pos**2
            usa_neg[i]  = np.sqrt(((index*F_neg - iF_neg)**2 * uF_neg**2).sum())/F_neg**2

        # VISIR flips the spectrum on the detector (as all echelles do).
        sa_pos[i] = -sa_pos[i]
        sa_neg[i] = -sa_neg[i]

        return sa_pos*aper_corr,usa_pos*aper_corr,sa_neg*aper_corr,usa_neg*aper_corr

    def plot(self):        
        plt.plot(self.wave,self.flux_pos,drawstyle='steps-mid')
        plt.plot(self.wave,self.flux_neg,drawstyle='steps-mid')
        plt.show()

    def plotSA(self):
        plt.plot(self.wave,self.sa_pos,drawstyle='steps-mid')
        plt.plot(self.wave,self.sa_neg,drawstyle='steps-mid')
        plt.show()

    def waveGuess(self,disp):
        index = np.arange(self.sh)-self.sh/2.+0.5   #Set central pixel to index of 0
        wave = disp['w0']+index*disp['dw']   #Note: w0 is wave *center*
        return wave

    def waveFit(self,disp,doPlot=False):
        '''
        Colette's wavelength solution. Only works for the 12.414 setting, so needs to be 
        refactored and generalized. 
        '''
        
        # Apply wavelength solution guess to data
        index = np.arange(self.sh)-self.sh/2.+0.5   #Set central pixel to index of 0
        data_wave = disp['w0']+index*disp['dw']   #Note: w0 is wave *center*
        data_sky = self.sky
        
        # Read in linelist from sky emission spectra
        # Note: This will need to be added to - right now only covers 12.36-12.46 microns
        out = np.recfromtxt('/Users/csalyk/mypy/pyvisir_master/pyvisir/emission_spectra/linelist.txt', names=True)
        w0 = out['w0']
        w0 = w0[np.where((w0 > np.min(data_wave)) & (w0 < np.max(data_wave)))]
        
        # Read in sky emission spectrum for plotting 

        emiss_file = '/Users/csalyk/mypy/pyvisir_master/pyvisir/emission_spectra/convol_emiss.p'
        with open(emiss_file, "rb") as f:
            convol_emiss = pickle.load(f)
        model_wave = convol_emiss[0] 
        model_sky = convol_emiss[1]

        # Loop through set of sky emission lines 
        # Fit data at each point.  
        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        wmax_list = []
        for i,my_w0 in enumerate(w0):

            wdata = np.where((data_wave > (my_w0-0.01)) & (data_wave < (my_w0+0.01)))
            wfit = np.where((data_wave > (my_w0-0.0015)) & (data_wave < (my_w0+0.0015)))
            wmax = np.where((data_wave > (my_w0-0.0015)) & (data_wave < (my_w0+0.0015)) & (data_sky == np.max(data_sky[wfit])))[0][0]
            wmax_list.append(wmax)    
            wmodel = np.where((model_wave > (my_w0-0.01)) & (model_wave < (my_w0+0.01)))

            if(doPlot == True):
                fig = plt.figure(figsize=(16,14))
                ax1 = fig.add_subplot(5,4,i+1)        
                ax1.plot([data_wave[wmax]],[data_sky[wmax]], 'ro')
                ax1.plot(data_wave[wdata], data_sky[wdata], linestyle='steps-mid')
                ax1.plot(data_wave[wfit], data_sky[wfit], linestyle='steps-mid', color='red')
                ax1.plot(model_wave[wmodel], model_sky[wmodel]*np.max(data_sky[wdata])/np.max(model_sky[wmodel]))
                ax1.plot([my_w0,my_w0],[-1e6,np.max(data_sky)*1.1])
                ax1.xaxis.set_major_formatter(formatter)
                ax1.set_xlim(my_w0-0.01,my_w0+0.01)
                ax1.set_ylim([-1e6,np.max(data_sky)*1.1])
                plt.show()
        wmax_list = np.array(wmax_list)    

        # out = lm(wmax_list,w0,doprint=False,doplot=False) 
        wstart = out['ahat']
        dwave = out['bhat']
        data_wave = wstart+np.arange(1024)*dwave

        if(doPlot == True):
            formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
            fig = plt.figure(figsize=(10,4))
            ax1 = fig.add_subplot(121)
            ax1.plot(wmax_list, w0, 'ro')
            ax1.xaxis.set_major_formatter(formatter)
            ax1.yaxis.set_major_formatter(formatter)
            ax1.plot(wmax_list,out['yfit'])
            ax1.set_xlabel('Pixel Number')
            ax1.set_ylabel(r'Wavelength [$\mu$m]')

            ax2 = fig.add_subplot(122)
            ax2.plot(model_wave, model_sky)
            ax2.plot(data_wave,data_sky*np.max(model_sky)/np.max(data_sky))
            ax2.set_xlim(np.min(data_wave),np.max(data_wave))
            ax2.xaxis.set_major_formatter(formatter)
            plt.show()

        return data_wave

    def writeSpec(self,filename=None,path='.'):
        
        c1  = pf.Column(name='wave', format='D', array=self.wave)
        c2  = pf.Column(name='flux', format='D', array=self.flux)
        c3  = pf.Column(name='uflux', format='D', array=self.uflux)
        c4  = pf.Column(name='sky', format='D', array=self.sky)
        c5  = pf.Column(name='usky', format='D', array=self.usky)        
        if self.sa:
            c6  = pf.Column(name='sa', format='D', array=self.sa)
            c7  = pf.Column(name='usa', format='D', array=self.usa)

        if self.sa:
            coldefs = pf.ColDefs([c1,c2,c3,c4,c5,c6,c7])
        else:
            coldefs = pf.ColDefs([c1,c2,c3,c4,c5])

        tbhdu = pf.BinTableHDU.from_columns(coldefs)

        # header['SETNAME'] = (self.setting, 'Setting name')
        # header['ORDER'] = (str(self.onum),'Order number')        

        #hdu = pf.PrimaryHDU(header=header)
        hdu = pf.PrimaryHDU()
        thdulist = pf.HDUList([hdu,tbhdu])

        if filename is None:
            date   = self.date.replace('-','')
            filename = path+'/'+self.target+'_'+str(self.obsid)+'_'+str(date)+'_order'+str(self.onum)+'.fits'
        
        thdulist.writeto(filename,overwrite=True)

        return filename
