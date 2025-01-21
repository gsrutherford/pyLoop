import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from omfit_classes import utils_fusion
from omfit_classes import omfit_eqdsk
from omfit_classes import omfit_efit
import omfit_classes
from scipy.interpolate import interp1d
import gadata
import MDSplus as MDS
import os
import netCDF4
from scipy.optimize import curve_fit
import time
import omfit_eqdsk_fast 
class pyLoop:

    def __init__(self,shot, time, efitID = 'EFIT02er', nrho=101, 
            useKinetic = False,doPlot = False, dt_avg = 50,
            outlierTimes = []):

        self.shot = shot
        self.efitID = efitID
        self.nrho = nrho
        self.rho_n = np.linspace(0,1,nrho)
        self.useKinetic = useKinetic

        self.Te_fit_rho_n = None
        self.Ti_fit_rho_n = None
        self.nC_fit_rho_n = None
        self.ne_fit_rho_n = None

        self.Ti_fit_times = []
        self.Te_fit_times = []
        self.nC_fit_times = []
        self.ne_fit_times = []

        self.Te_fits = []
        self.Ti_fits = []
        self.ne_fits = []
        self.nC_fits = []

        self.Te_fit_err = []
        self.ne_fit_err = []
        self.Ti_fit_err = []
        self.nC_fit_err = []

        if not useKinetic:
            if self.shot == 179173:
                targetDir = '/fusion/projects/xpsi/petty/analysis/20190717/shot179173/cos2_er/460_390/'
                self.collectCraigProfs(targetDir)
            elif self.shot == 179186:
                targetDir = '/fusion/projects/xpsi/petty/analysis/20190717/shot179186/cos2_er/460_390/'
                self.collectCraigProfs(targetDir)
            elif self.shot == 179587:
                #targetDir = '/fusion/projects/xpsi/petty/analysis/20190807/shot179587/cos2_er/eccd/'
                targetDir = '/fusion/projects/xpsi/petty/analysis/20190807/shot179587/EFIT02ER/'
                self.collectCraigProfs(targetDir)
            elif self.shot == 179592:
                #targetDir = f'/fusion/projects/xpsi/petty/analysis/20190807/shot179592/cos2_er/440_280/'
                targetDir = f'/fusion/projects/xpsi/petty/analysis/20190807/shot179592/EFIT02ER/'
                self.collectCraigProfs(targetDir)
            else:
                self.collectZipfits()

        self.dt_avg = dt_avg
        self.time =time
    
        self.f = np.zeros(nrho)
        self.eps = np.zeros(nrho)
        self.avgr = np.zeros(nrho)
        self.avgR  = np.zeros(nrho)
        self.avgB2 = np.zeros(nrho)
        self.avgBphi2 = np.zeros(nrho)
        self.avgJpara = np.zeros(nrho)
        self.q = np.zeros(nrho)
        self.ft = np.zeros(nrho)
        #self.grad_term = np.zeros(nrho)
        #self.fcap = np.zeros(nrho)
        #self.gcap = np.zeros(nrho)
        #self.hcap = np.zeros(nrho)
        self.psi_n = np.zeros(nrho)
        self.psi = np.zeros(nrho)
        self.rho_bndry = 0
        self.B_0 = 0
        
        self.dpsi_dt = np.zeros(nrho)
        self.drho_bndry_dt  = 0 
        self.dB0_dt  = 0      

        self.voltage = np.zeros(nrho)
        self.E_para = np.zeros(nrho)
        self.J_ohm = np.zeros(nrho)
        self.J_BS = np.zeros(nrho)
        self.currentSign = 0  

        self.outlierTimes = outlierTimes
        self.doPlot = doPlot

    #Temperature in units of keV
    #density in units of 1/m^3
    def collectKineticProfs(self):
        profile_nc = netCDF4.Dataset(f'/home/rutherfordg/nvloop/{self.shot}.0{self.time}_kinetic_profs.nc')
        self.Te_fit_rho_n = profile_nc.variables['rho'][:]
        self.Ti_fit_rho_n = profile_nc.variables['rho'][:]
        self.nC_fit_rho_n = profile_nc.variables['rho'][:]
        self.ne_fit_rho_n = profile_nc.variables['rho'][:]

        self.Ti_fit_times = profile_nc.variables['time'][:]
        self.Te_fit_times = profile_nc.variables['time'][:]
        self.nC_fit_times = profile_nc.variables['time'][:]
        self.ne_fit_times = profile_nc.variables['time'][:]

        #zipfits come out as (rho, time)
        #kinetic from quickfit come as (time, rho)
        #this is to match zipfits
        self.Te_fits = (profile_nc.variables['T_e'][:]/1e3).T
        self.Ti_fits = (profile_nc.variables['T_12C6'][:]/1e3).T
        self.ne_fits = profile_nc.variables['n_e'][:].T
        self.nC_fits = profile_nc.variables['n_12C6'][:].T

        self.Te_fit_err = np.zeros(self.Te_fits.shape)
        self.ne_fit_err = np.zeros(self.ne_fits.shape)
        self.Ti_fit_err = np.zeros(self.Ti_fits.shape)
        self.nC_fit_err = np.zeros(self.nC_fits.shape)

    
    def collectCraigProfs(self, targetDir):
        from scipy.io import readsav
        import os
        directory = os.listdir(targetDir)
        searchString = f'dne{self.shot}.0'
        for fname in directory:
            if not (searchString in fname):
                continue
            denfit = readsav(f'{targetDir}{fname}',
                    python_dict = True)
            rhos = denfit['ne_str'].RHO_DENS[0]
            den = denfit['ne_str'].DENS[0]*1e19
            den [den < 10] = 10
            if self.ne_fit_rho_n is None:
                self.ne_fit_rho_n = self.rho_n

            if len(rhos) == len(den):
                denTime = float(fname.split('.')[-1][1:])
                interped_den = interp1d(rhos, den)(self.rho_n)
                self.ne_fit_times.append(denTime)
                self.ne_fits.append(interped_den)
                self.ne_fit_err.append(np.ones(len(interped_den)))

        searchString = f'dti{self.shot}.0'
        for fname in directory:
            if not (searchString in fname):
                continue
            tifit = readsav(f'{targetDir}{fname}',
                    python_dict = True)
            rhos = tifit['ti_str'].RHO_TI[0]
            ti = tifit['ti_str'].TI[0]*1
            ti[ti < 1/1000] = 1/1000
            if self.Ti_fit_rho_n is None:
                self.Ti_fit_rho_n = self.rho_n

            tiTime = float(fname.split('.')[-1][1:])
            self.Ti_fit_times.append(tiTime)
            interped_ti = interp1d(rhos, ti)(self.rho_n)
            self.Ti_fits.append(interped_ti)
            self.Ti_fit_err.append(np.ones(len(interped_ti)))

        searchString = f'dte{self.shot}.0'
        for fname in directory:
            if not (searchString in fname):
                continue
    
            tefit = readsav(f'{targetDir}{fname}',
                    python_dict = True)
            rhos = tefit['te_str'].RHO_TE[0]
            te = tefit['te_str'].TE[0]*1
            te[te < 1/1000] = 1/1000
            if self.Te_fit_rho_n is None:
                self.Te_fit_rho_n = self.rho_n
            interp1d_te = interp1d(rhos, te)(self.rho_n)
            teTime = float(fname.split('.')[-1][1:])
            self.Te_fit_times.append(teTime)
            self.Te_fits.append(interp1d_te)
            self.Te_fit_err.append(np.ones(len(interp1d_te)))
            #fig,ax = plt.subplots()
            #ax.plot(rhos, te)
            #ax.set_title(str(teTime))
            #plt.show()

        searchString = f'dimp{self.shot}.0'
        for fname in directory:
            if not (searchString in fname):
                continue
            impfit = readsav(f'{targetDir}{fname}',
                    python_dict = True)
            rhos = impfit['impdens_str'].RHO_IMP[0]
            imp = impfit['impdens_str'].ZDENS[0]*1e19
            imp [imp < 10] = 10
            if self.nC_fit_rho_n is None:
                self.nC_fit_rho_n = self.rho_n

            impTime = float(fname.split('.')[-1][1:].split('_')[0])
            interped_imp = interp1d(rhos, imp)(self.rho_n)
            self.nC_fit_times.append(impTime)
            self.nC_fits.append(interped_imp)
            self.nC_fit_err.append(np.ones(len(interped_imp)))
        #these should be (rho, time)
        self.nC_fits = np.array(self.nC_fits).T
        self.Te_fits = np.array(self.Te_fits).T
        self.ne_fits = np.array(self.ne_fits).T
        self.Ti_fits = np.array(self.Ti_fits).T

        self.nC_fit_err = np.array(self.nC_fit_err).T
        self.Te_fit_err = np.array(self.Te_fit_err).T
        self.ne_fit_err = np.array(self.ne_fit_err).T
        self.Ti_fit_err = np.array(self.Ti_fit_err).T

        self.nC_fit_times = np.array(self.nC_fit_times)
        self.ne_fit_times = np.array(self.ne_fit_times)
        self.Te_fit_times = np.array(self.Te_fit_times)
        self.Ti_fit_times = np.array(self.Ti_fit_times)


    def collectZipfits(self):
        Te_fitData = gadata.gadata('PROFILES.ETEMPFIT', self.shot, tree = 'zipfit01')
        self.Te_fit_times = Te_fitData.ydata
        self.Te_fit_rho_n = Te_fitData.xdata
        #for some reason zdata and zerr have their dimensions swapped
        #so we transpose to simplify future operations
        self.Te_fit_err = Te_fitData.zerr.T
        self.Te_fits = Te_fitData.zdata
        #if the value is less than 1 eV, set it to 1 eV
        self.Te_fits[self.Te_fits < 1/1000] = 1/1000 

        Ti_fitData = gadata.gadata('PROFILES.ITEMPFIT', self.shot, tree = 'zipfit01')
        self.Ti_fit_times = Ti_fitData.ydata
        self.Ti_fit_rho_n = Ti_fitData.xdata
        self.Ti_fit_err = Ti_fitData.zerr.T
        self.Ti_fits = Ti_fitData.zdata
        #if the value is less than 1 eV, set it to 1 eV
        self.Ti_fits[self.Ti_fits < 1/1000] = 1/1000 

        ne_fitData = gadata.gadata('PROFILES.EDENSFIT', self.shot, tree = 'zipfit01')
        self.ne_fit_times = ne_fitData.ydata
        self.ne_fit_rho_n = ne_fitData.xdata
        self.ne_fit_err = ne_fitData.zerr.T*1e19
        self.ne_fits = ne_fitData.zdata*1e19
        #if the value is less than 10 m^-3, set it to 10 m^-3
        self.ne_fits[self.ne_fits < 10] = 10
        
        try:
            nC_fitData = gadata.gadata('PROFILES.ZDENSFIT', self.shot, tree = 'zipfit01')
            self.nC_fit_times = nC_fitData.ydata
            self.nC_fit_rho_n = nC_fitData.xdata
            self.nC_fit_err = nC_fitData.zerr.T*1e19
            self.nC_fits = nC_fitData.zdata*1e19
            self.nC_fits[self.nC_fits < 10] = 10
        except:
            print(f'Something wrong with carbon data, taking Zeff = 1')
            nC_fitData = gadata.gadata('PROFILES.EDENSFIT', self.shot, tree = 'zipfit01')
            self.nC_fit_times = ne_fitData.ydata
            self.nC_fit_rho_n = ne_fitData.xdata
            self.nC_fit_err = ne_fitData.zerr.T*1e19
            self.nC_fits = ne_fitData.zdata*1e19
            #if the value is less than 10 m^-3, set it to 10 m^-3
            self.nC_fits[self.nC_fits < 10] = 10
        assert (self.nC_fit_rho_n == self.ne_fit_rho_n).all()

    #generates a linear fit of the input x and y data
    #evaluates said fit at x0 and returns the result and fit parameter uncertainties
    #aa is the fitted y0, dydx is the slope
    def getLinearFit(self,x, y, x0, y_err=None, returnCoefs= False):

        # Data dimension
        ndata = len(x)

        # If there's only one point, return default values
        if ndata == 1:
            return {'y0': y[0], 'sigy0': 0.0, 'dydx': 0.0, 'sigdydx': 0.0}

        # Set up weights
        do_err = False
        if y_err is not None and np.max(np.abs(y_err)) > 0:
            do_err = True
        else:
            y_err = np.ones(ndata)  # Default to equal weights if no errors provided

        # Weighted sums
        wgt = 1 / y_err**2
        sumwt = np.sum(wgt)
        sumx = np.sum(x * wgt)
        sumy = np.sum(y * wgt)
        xctr = sumx / sumwt
        tt = x - xctr
        sumtt = np.sum(tt**2 * wgt)

        # Coefficients for the linear fit (slope and intercept)
        dydx = np.sum(tt * y * wgt) / sumtt
        aa = (sumy - dydx * sumx) / sumwt
        y0 = aa + x0*dydx
        
        if returnCoefs:
            return [y0, aa, dydx]
        else:
            return [y0]

    #psi is fit differently than the rest as it's more sensitive
    def getdpsiFit(self, x, y, x0):
        from scipy.interpolate import UnivariateSpline

        spline = UnivariateSpline(x,y,k=1,s=0.1)
        deriv = spline.derivative()(x0)
        """
        if self.shot == 202158:
            fig,ax = plt.subplots()
            ax.scatter(x,y)
            ax.plot(x, spline(x))


            spline2 = UnivariateSpline(x[3:-3],y[3:-3],k=1,s=.1)
            deriv2 = spline2.derivative()(x0)
            ax.plot(x[3:-3], spline2(x[3:-3]))
            ax.set_ylabel(r'$\psi (\rho_j)$')
            ax.set_xlabel(r'time (ms)')

            print(f'full, cut: {deriv, deriv2}')
            print(f'full, cut: {spline.derivative()(1300), spline2.derivative()(1300)} @ 1300')
            fig.tight_layout()
            plt.show()
        """
        
        return deriv

    # returns a Te profile at the desired normalized rho points averaged over
    # the time range t-dt/2,t+dt/2
    # input:
    #   shot     = shot of interest
    #   t        = time of interest
    #   dt       = time window for averaging
    #
    # return:
    #   avgTe = Te(keV) at the rho points, averaged over time
    #   errOfAvg = error associated with avgTe
    #   dTe_drhon = derivative w.r.t rho
    #   dTe_drhon_err = error associated with dTe_drhon
    def get_Te(self, t, dt):        
        mask = np.where((self.Te_fit_times >= t-dt/2)*(self.Te_fit_times <= t+dt/2))[0]
        Tes_ofInterest = self.Te_fits[:,mask]
        errsOfInterest = self.Te_fit_err[:,mask]
        avgTe, errOfAvg = self.getAvgAndInterpWithError(self.Te_fit_rho_n, Tes_ofInterest, self.rho_n, errsOfInterest)

        dTe_drhon, dTe_drhon_err = self.getGradientAndError(self.rho_n, avgTe, errOfAvg)
        return avgTe, errOfAvg, dTe_drhon, dTe_drhon_err

    # returns a Ti profile at the desired normalized rho points averaged over
    # the time range t-dt/2,t+dt/2
    # input:
    #   shot     = shot of interest
    #   t        = time of interest
    #   dt       = time window for averaging
    #
    # return:
    #   avgTi = Ti(keV) at the rho points, averaged over time
    #   errOfAvg = error associated with avgTi
    #   dTi_drhon = derivative w.r.t rho
    #   dTi_drhon_err = error associated with dTi_drhon
    def get_Ti(self, t, dt):       
        mask = np.where((self.Ti_fit_times >= t-dt/2)*(self.Ti_fit_times <= t+dt/2))[0]
        Tis_ofInterest = self.Ti_fits[:,mask]
        errsOfInterest = self.Ti_fit_err[:,mask]

        avgTi, errOfAvg = self.getAvgAndInterpWithError(self.Ti_fit_rho_n, 
            Tis_ofInterest, self.rho_n, errsOfInterest)

        dTi_drhon, dTi_drhon_err = self.getGradientAndError(self.rho_n, avgTi, errOfAvg)

        return avgTi, errOfAvg, dTi_drhon, dTi_drhon_err

    # returns a ne profile at the desired normalized rho points averaged over
    # the time range t-dt/2,t+dt/2
    # input:
    #   shot     = shot of interest
    #   t        = time of interest
    #   dt       = time window for averaging
    #
    # return:
    #   avgne = ne(m^-3) at the rho points, averaged over time
    #   errOfAvg = error associated with avgne
    #   dne_drhon = derivative w.r.t rho
    #   dne_drhon_err = error associated with dne_drhon
    def get_ne(self, t, dt):    
        mask = np.where((self.ne_fit_times >= t-dt/2)*(self.ne_fit_times <= t+dt/2))[0]
        nes_ofInterest = self.ne_fits[:,mask]
        errsOfInterest = self.ne_fit_err[:,mask]
        avgne, errOfAvg = self.getAvgAndInterpWithError(self.ne_fit_rho_n, nes_ofInterest, self.rho_n, errsOfInterest)

        dne_drhon, dne_drhon_err = self.getGradientAndError(self.rho_n, avgne, errOfAvg)
        return avgne, errOfAvg, dne_drhon, dne_drhon_err

    # returns a nC profile at the desired normalized rho points averaged over
    # the time range t-dt/2,t+dt/2
    # input:
    #   shot     = shot of interest
    #   t        = time of interest
    #   dt       = time window for averaging
    #
    # return:
    #   avgnC = nC(m^-3) at the rho points, averaged over time
    #   errOfAvg = error associated with avgnC
    #   dnC_drhon = derivative w.r.t rho
    #   dnC_drhon_err = error associated with dnC_drhon
    def get_nC(self, t, dt):        
        mask = np.where((self.nC_fit_times >= t-dt/2)*(self.nC_fit_times <= t+dt/2))[0]
        nCs_ofInterest = self.nC_fits[:,mask]
        errsOfInterest = self.nC_fit_err[:,mask]
        avgnC, errOfAvg = self.getAvgAndInterpWithError(self.nC_fit_rho_n, nCs_ofInterest, self.rho_n, errsOfInterest)

        dnC_drhon, dnC_drhon_err = self.getGradientAndError(self.rho_n, avgnC, errOfAvg)
        return avgnC, errOfAvg, dnC_drhon, dnC_drhon_err

    #returns the <y>(x) profile
    #assumes the second index of y_matrix and error_matrix corresponds to the x coordinate
    def getAvgAndInterpWithError(self, x, y_matrix, new_x,error_matrix):
        avg_y = np.average(y_matrix, axis = 1)
        avg_y_ongrid = interp1d(x, avg_y)(new_x)

        errOfAvg = (1/np.sqrt(error_matrix.shape[1]))*np.sqrt(np.sum(error_matrix**2,axis = 1))
        errOfAvg_ongrid = interp1d(x, errOfAvg)(new_x)

        return avg_y_ongrid, errOfAvg_ongrid

    #returns the gradient and the error in that gradient
    #gradient is on the same x as the input y
    #uses second order central and forward/backward differences
    def getGradientAndError(self, x,y,error):
        #edge order defines edge derivative order
        dydx = np.gradient(y, x, edge_order = 2)

        dydx_err = np.zeros(len(dydx))
        #propagate error through derivative assuming central difference 
        dydx_err[1:-1] = np.sqrt(error[2:]**2 + error[:-2]**2)/(x[2:] - x[:-2])
        #assume 2nd order forward/backward difference at the edges
        dydx_err[0] = np.sqrt((9/4)*error[0]**2 + 4*error[1]**2 + (1/4)*error[2]**2)/(x[2] - x[0])
        dydx_err[-1] = np.sqrt((9/4)*error[-1]**2 + 4*error[-2]**2 + (1/4)*error[-3]**2)/(x[-1] - x[-3])

        return dydx, dydx_err

    # returns the Zeff profile assuming a carbon impurity
    # x axis is assumed to be self.rho_n
    # 
    # input:
    #   ne      = electron density profile
    #   ne_err  = error of ne
    #   nC      = carbon density profile
    #   nC_err  = error of nC
    #
    # return:
    #   Zeff
    #   Zeff_err
    #   dZeff/drho_n
    #   dZeff/drho_n_err
    def get_Zeff(self,ne, ne_err, nC, nC_err):
        Zeff = (30*nC+ne)/ne
        Zeff[Zeff < 1] = 1
        Zeff_err = np.sqrt(30**2*nC_err**2+ne_err**2)/ne
        dZeff_drhon, dZeff_drhon_err = self.getGradientAndError(self.rho_n,Zeff,Zeff_err)
        return Zeff, Zeff_err, dZeff_drhon, dZeff_drhon_err

    #returns the bootstrap current, A/m^2
    def getBootstrapAndConductivity(self, t, dt, gfile, Z_impurity = 6):
        if self.useKinetic:
            self.collectKineticProfs()

        ne, ne_err, _, _ = self.get_ne(t, dt)
        Te, Te_err, _, _ = self.get_Te(t, dt)
        nC, nC_err, _, _ = self.get_nC(t, dt)
        Ti, Ti_err, _, _ = self.get_Ti(t, dt)
        Zeff, Zeff_err, dZeff_drhon, dZeff_drhon_err = self.get_Zeff(ne, ne_err, nC, nC_err)

        #convert temperature to units of eV:
        Te*=1e3
        Te_err*=1e3
        Ti*=1e3
        Ti_err*=1e3

        nD = ne*Zeff-Z_impurity**2*nC
        nD[nD <= 10] = 10
        pressure = (ne*Te + nD*Ti + nC*Ti)*1.602e-19
        """
        fig,ax = plt.subplots()
        ax.plot(self.rho_n, Te)
        ax.plot(self.rho_n, Ti)
        ax2 =ax.twinx()
        ax2.plot(self.rho_n, ne, linestyle = 'dashed')
        ax2.plot(self.rho_n, nC, linestyle = 'dashed')
        ax2.plot(self.rho_n, nD, linestyle = 'dashed')
        plt.show()
        #car = los
        """

        J_BS_prof_1 = self.currentSign*utils_fusion.sauter_bootstrap(
                psi_N = np.array([self.psi_n]), q = np.array([self.q]), 
                fT = np.array([self.ft]), eps = np.array([self.eps]),
                psiraw = np.array([self.psi]), R = np.array([self.avgR]),
                I_psi = np.array([self.f]),
                Ti = np.array([Ti]), ne = np.array([ne]), Te = np.array([Te]),
                charge_number_to_use_in_ion_collisionality = 'Zavg', 
                charge_number_to_use_in_ion_lnLambda = 'Zavg',
                Zis=[1,6], nis = np.array([[nD], [nC]]),
                R0 = 1.6955, p = np.array([pressure]), version = 'osborne')[0]

        sigma_neo = utils_fusion.nclass_conductivity(Zeff = np.array([Zeff]), 
                psi_N = self.psi_n, Ti = np.array([Ti]),
                ne = np.array([ne]), Te = np.array([Te]), 
                q = self.q, eps = self.eps,
                fT = self.ft, R = self.avgR,
                Zdom = 1,
                charge_number_to_use_in_ion_collisionality = 'Zavg',
                charge_number_to_use_in_ion_lnLambda = 'Zavg', 
                Zis=[1,6], nis = np.array([[nD], [nC]]))[0]
        

        return J_BS_prof_1, sigma_neo


    #inputs:
    #   gfile = OMFITgeqdsk object
    #
    # outputs:
    #   ff      = f
    #   eps = inverse aspect ratio
    #   avgR  = major radius of surface
    #   avgB2   = B^2
    #   avgBphi2   = B_phi^2
    #   avgJpara     = j_parallel
    #   q      = safety factor
    #   ft      = trapped particle fraction
    #   grad_term = (b.grad(B))^2
    #   fcap    = B_ref*R_zero/F
    #   gcap    = <B_pol^2>/B_pol0^2
    #   hcap    = dV/(2*pi*R_zero)/(2*pi*rho*drho)
    #   bphirb  = <|B_phi/RB|>
    def get_gfileQuantities(self,gfileAtTime, gfilesInTimeWindow):
        startTime = time.time()
        ###First get the quantities that don't need multiple gfiles
        #We're not going to average them since there is a gfile at our chosen time
        self.currentSign = np.sign(gfileAtTime['CURRENT'])

        fs = np.zeros((len(gfilesInTimeWindow), len(self.rho_n)))
        epss = np.zeros((len(gfilesInTimeWindow), len(self.rho_n)))
        avgrs = np.zeros((len(gfilesInTimeWindow), len(self.rho_n)))
        avgRs = np.zeros((len(gfilesInTimeWindow), len(self.rho_n)))
        avgB2s = np.zeros((len(gfilesInTimeWindow), len(self.rho_n)))
        avgBphi2s = np.zeros((len(gfilesInTimeWindow), len(self.rho_n)))
        avgJparas = np.zeros((len(gfilesInTimeWindow), len(self.rho_n)))
        qs = np.zeros((len(gfilesInTimeWindow), len(self.rho_n)))
        fts = np.zeros((len(gfilesInTimeWindow), len(self.rho_n)))
        psi_ns = np.zeros((len(gfilesInTimeWindow), len(self.rho_n)))
        psis = np.zeros((len(gfilesInTimeWindow), len(self.rho_n)))
        rho_bndrys = np.zeros(len(gfilesInTimeWindow))
        B_0s = np.zeros(len(gfilesInTimeWindow))

        gfileTimes = np.sort(np.array(list(gfilesInTimeWindow.keys())))
        num_gfiles = len(gfilesInTimeWindow)
    
        madeVarsTime = time.time()
        print(f'{madeVarsTime-startTime} to make variables')

        for j in range(num_gfiles):
            gfile = gfilesInTimeWindow[gfileTimes[j]]
            fluxSurfaces = gfile['fluxSurfaces']
            geo = fluxSurfaces['geo']
            avg = fluxSurfaces['avg']
            
            setupTime = time.time()

            print(f'{setupTime-madeVarsTime} to get gfile, avg, geo')

            avgr = avg['a']
            avgR = avg['R']
            avgB2 = avg['Btot**2']
            avgBphi2 = avg['Bt**2']
            q = avg['q']

            avgTime = time.time()
            print(f'{avgTime-setupTime} to read from avgs')

            gfile_psi = geo['psi']
            eps = geo['eps']
            rho_bndrys[j] = geo['rho'][-1]

            geoTime = time.time()
            print(f'{geoTime-avgTime} to read from geo')

            gfile_rho_n = gfile['RHOVN']
            gfile_psi_n = fluxSurfaces['levels']
            f = gfile['FPOL']
            avgJpara = gfile.surfAvg('Jpar', interp = 'cubic')*self.currentSign
            ft = utils_fusion.f_t(r_minor = avgr, R_major = avgR)
            B_0s[j] = gfile['BCENTR']

            gfileQuantsTime = time.time()
            print(f'{gfileQuantsTime-geoTime} to finish')

            fs[j,:] = interp1d(gfile_rho_n, f, kind = 'cubic')(self.rho_n)
            epss[j,:] = interp1d(gfile_rho_n, eps, kind = 'cubic')(self.rho_n)
            avgrs[j,:] = interp1d(gfile_rho_n, avgr, kind = 'cubic')(self.rho_n)
            avgRs[j,:] = interp1d(gfile_rho_n, avgR, kind = 'cubic')(self.rho_n)
            avgB2s[j,:] = interp1d(gfile_rho_n, avgB2, kind = 'cubic')(self.rho_n)
            avgBphi2s[j,:] = interp1d(gfile_rho_n, avgBphi2, kind = 'cubic')(self.rho_n)
            avgJparas[j,:] = interp1d(gfile_rho_n, avgJpara, kind = 'cubic')(self.rho_n)
            qs[j,:] = interp1d(gfile_rho_n, q, kind = 'cubic')(self.rho_n)
            fts[j,:] = interp1d(gfile_rho_n, ft, kind = 'cubic')(self.rho_n)
            psi_ns[j,:] = interp1d(gfile_rho_n, gfile_psi_n, kind = 'cubic')(self.rho_n)
            psis[j,:] = interp1d(gfile_rho_n, gfile_psi, kind = 'cubic')(self.rho_n)   

            print(f'{time.time()-gfileQuantsTime} to interp')

            car = los


        getQuantsTime = time.time()
        print(f'{getQuantsTime - madeVarsTime} to get quants from omfit')
        for k in range(len(self.rho_n)):
            self.f[k] = self.getLinearFit(gfileTimes, fs[:,k],self.time)[0]
            self.eps[k] = self.getLinearFit(gfileTimes, epss[:,k],self.time)[0]
            self.avgr[k] = self.getLinearFit(gfileTimes, avgrs[:,k],self.time)[0]
            self.avgR[k] = self.getLinearFit(gfileTimes, avgRs[:,k],self.time)[0]
            self.avgB2[k] = self.getLinearFit(gfileTimes, avgB2s[:,k],self.time)[0]
            self.avgBphi2[k] = self.getLinearFit(gfileTimes, avgBphi2s[:,k],self.time)[0] 
            self.avgJpara[k] = self.getLinearFit(gfileTimes, avgJparas[:,k],self.time)[0]
            self.q[k] = self.getLinearFit(gfileTimes, qs[:,k],self.time)[0]
            self.ft[k] = self.getLinearFit(gfileTimes, fts[:,k],self.time)[0]
            self.psi_n[k] = self.getLinearFit(gfileTimes, psi_ns[:,k],self.time)[0]
            self.psi[k] = self.getLinearFit(gfileTimes, psis[:,k],self.time)[0]
            self.dpsi_dt[k] = self.getLinearFit(gfileTimes, psis[:,k],self.time, returnCoefs = True)[2]*1000
            
            """
            if k == 0:
                #mask = np.where((gfileTimes >= 1340-150)*
                #            (gfileTimes <= 1340+150))

                fig,ax = plt.subplots()
                ax.scatter(gfileTimes, psis[:,k])
                coef1 = self.getLinearFit(gfileTimes, psis[:,k],self.time, returnCoefs = True)
                #coef2 = self.getLinearFit(gfileTimes[mask], psis[mask,k],self.time, returnCoefs = True)
                ax.plot(gfileTimes,coef1[1] + coef1[2]*gfileTimes)
                #ax.plot(gfileTimes[3:-3], coef2[1] + coef2[2]*gfileTimes[3:-3])
                #print(f'slop1, slope2: {coef1[2], coef2[2]}')
                plt.show()
            """
        self.rho_bndry = self.getLinearFit(gfileTimes, rho_bndrys,self.time)[0]
        self.B_0 = self.getLinearFit(gfileTimes, B_0s,self.time)[0]
            
        self.drho_bndry_dt = self.getLinearFit(gfileTimes, rho_bndrys,self.time, returnCoefs = True)[2]*1000  
        self.dB0_dt = self.getLinearFit(gfileTimes, B_0s,self.time, returnCoefs = True)[2]*1000 
        fitTime = time.time()
        print(f'{fitTime - getQuantsTime} to do fits')



    #shot   = shot number
    # t1     = time for first vloop analysis (msec)
    # dtstep = interval between vloop analyses (msec) 
    #             (t1, t1+dtstep, t1+2*dtstep, ...)
    # nvlt   = number of vloop analyses requested
    # dtavg  = averaging time window for each analysis (msec)
    #             (t-dtavg/2 to t+dtavg/2)
    def nvloop(self):
        currentTime = time.time()
        previousTime = time.time()
        #efitTimeNode = tree.getNode('RESULTS.GEQDSK.GTIME')
        gtimes = gadata.gadata('RESULTS.GEQDSK.GTIME', self.shot, tree = self.efitID).zdata


        if self.useKinetic:
            dir_list = os.listdir(f'/home/rutherfordg/nvloop/{self.shot}.0{time}_kinetic_efits')
            gfilesInTimeWindow = {}
            relevantGfileTimes = []
            for filename in dir_list:
                print(filename)
                if filename[0] != 'g':
                    continue
                gtime = float(filename.split('.')[-1][1:])
                if self.time - self.dt_avg/2 <= gtime <= self.time + self.dt_avg/2:
                    relevantGfileTimes.append(gtime)
                    gfilesInTimeWindow[gtime]=omfit_eqdsk_fast.OMFITgeqdsk_fast(f'/home/rutherfordg/nvloop/{self.shot}.0{time}_kinetic_efits/{filename}')
            relevantGfileTimes = np.array(relevantGfileTimes)
            relevantGfileTimes.sort()
        elif self.shot in [179173, 179186, 179592, 179587]:
            targetDir = ''
            if self.shot == 179173:
                targetDir = '/fusion/projects/xpsi/petty/analysis/20190717/shot179173/cos2_er/460_390'
            elif self.shot == 179186:
                targetDir = '/fusion/projects/xpsi/petty/analysis/20190717/shot179186/cos2_er/460_390'
            elif self.shot == 179177:
                targetDir = '/fusion/projects/xpsi/petty/analysis/20190717/shot179177/cos2_er/460_390'
            elif self.shot == 179592:
                targetDir = '/fusion/projects/xpsi/petty/analysis/20190807/shot179592/cos2_er/440_280'
            elif self.shot == 179587:
                targetDir = '/fusion/projects/xpsi/petty/analysis/20190807/shot179587/cos2_er/119'
            dir_list = os.listdir(targetDir)
            gfilesInTimeWindow = {}
            relevantGfileTimes = []
            for filename in dir_list:
                if filename[0] != 'g':
                    continue
                gtime = float(filename.split('.')[-1][1:])
                if self.time - self.dt_avg/2 <= gtime <= self.time + self.dt_avg/2:
                    relevantGfileTimes.append(gtime)
                    gfilesInTimeWindow[gtime]=omfit_eqdsk_fast.OMFITgeqdsk_fast(f'{targetDir}/{filename}')
            relevantGfileTimes = np.array(relevantGfileTimes)
            relevantGfileTimes.sort()
        else:
            mask = np.where((gtimes >= self.time - self.dt_avg/2)*
                            (gtimes <= self.time + self.dt_avg/2))
            relevantGfileTimes = gtimes[mask]

            gfilesInTimeWindow = omfit_eqdsk_fast.from_mds_plus(device = 'd3d',shot = self.shot, 
                    times = relevantGfileTimes, snap_file = self.efitID, 
                    get_afile = False, get_mfile = False, quiet = True,
                    time_diff_warning_threshold = 1)['gEQDSK']

        currentTime = time.time()
        print(f'getting profiles and gfiles took {currentTime-previousTime}')
        previousTime = currentTime

        gfileTimes = gfilesInTimeWindow.keys()
        for outlierTime in self.outlierTimes:
            gfilesInTimeWindow.pop(outlierTime,None)
        """
        if self.time not in relevantGfileTimes:
            print('***')
            print('Requested gfile time is not in tree, skipping it')
            print('***')
            return
        """
        print(f'num gfiles in window: {len(relevantGfileTimes)}')
        if len(relevantGfileTimes) < 3:
            print('***')
            print('Need at least 3 eqdsks in timewindow')
            print('***')
            return


        gfileAtTime = gfilesInTimeWindow[float(self.time)]
        self.get_gfileQuantities(gfileAtTime, gfilesInTimeWindow)

        currentTime = time.time()
        print(f'getting gfile quantities took {currentTime-previousTime}')
        previousTime = currentTime

        J_BS, sigma_neo = self.getBootstrapAndConductivity(self.time, self.dt_avg, 
                            gfileAtTime)
        self.J_BS = J_BS
        
        currentTime = time.time()
        print(f'getting BS took {currentTime-previousTime}')
        previousTime = currentTime

        vv1 = 2*np.pi*self.rho_n**2*(self.B_0/self.q[:])*self.rho_bndry*self.drho_bndry_dt
        vv2 = np.pi*self.rho_bndry**2*self.rho_n**2*self.dB0_dt/(self.q)
        self.voltage = 2*np.pi*self.dpsi_dt - vv1

        #"""
        self.E_para = ((self.voltage-vv2)*
                    self.avgBphi2/(2*np.pi*self.B_0*self.f))

        fig,ax = plt.subplots()
        ax.plot(self.rho_n, self.voltage, lw = 2, label = 'with correction')
        ax.plot(self.rho_n, 2*np.pi*self.dpsi_dt, lw = 2, label = 'no correction')
        ax.set_xlabel(f'rho_n')
        ax.set_ylabel(f'Voltage')
        ax.legend()
        ax.set_box_aspect(1)
        fig.tight_layout()
        plt.show()

        fig,ax = plt.subplots()
        ax.plot(self.rho_n, self.E_para, lw = 2, label = 'with correction')
        ax.plot(self.rho_n, ((self.voltage)*self.avgBphi2/(2*np.pi*self.B_0*self.f)), lw = 2, label = 'no E correction')
        ax.plot(self.rho_n, ((2*np.pi*self.dpsi_dt)*self.avgBphi2/(2*np.pi*self.B_0*self.f)), lw = 2, label = 'no E or V correction')
        ax.set_xlabel(f'rho_n')
        ax.set_ylabel(f'E||')
        ax.legend()
        ax.set_box_aspect(1)
        fig.tight_layout()
        plt.show()

            

        self.J_ohm = self.E_para*sigma_neo
        self.J_NI = self.avgJpara-self.J_BS-self.J_ohm 

        if self.doPlot:
            self.plot()


    def plot(self):
            fig,axes = plt.subplots(nrows = 2, ncols = 3)
            axes[0,0].plot(self.rho_n, np.abs(self.q), lw = 2)
            axes[0,0].set_title('q')
            axes[0,0].set_xlabel(r'$\hat{\rho}$')
            #axes[0,0].set_ylim([1,7])

            axes[0,1].plot(self.rho_n, self.avgJpara/1e6, lw = 2)
            axes[0,1].set_title(r'<$J_{||}$> (MA/m^2)')
            axes[0,1].set_xlabel(r'$\hat{\rho}$')
            #axes[0,1].set_ylim([.2,1.2])

            axes[0,2].plot(self.rho_n, self.dpsi_dt, lw = 2)
            axes[0,2].set_title(r'$\dot{\psi}$')
            axes[0,2].set_xlabel(r'$\hat{\rho}$')


            axes[1,0].plot(self.rho_n, self.voltage, lw = 2)
            axes[1,0].set_title(r'$V_\phi$ (V)')
            axes[1,0].set_xlabel(r'$\hat{\rho}$')
            #axes[1,0].set_ylim([-.15,.1])

            axes[1,1].plot(self.rho_n, self.E_para, lw = 2)
            axes[1,1].set_title(r'$E_{||}$ (V/m)')
            axes[1,1].set_xlabel(r'$\hat{\rho}$')
            #axes[1,1].set_ylim([-.015,.01])

            fig.suptitle(f'Shot {self.shot}, {self.time} ms')
            axes[1,2].plot(self.rho_n, self.psi,lw = 2)
            axes[1,2].set_title(r'$\psi$')
            axes[1,2].set_xlabel(r'$\hat{\rho}$')

            fig.tight_layout()
            #"""
            #"""
            figJ, axesJ = plt.subplots(nrows = 2)
            axesJ[0].plot(self.rho_n, self.J_ohm/1e6, lw = 2, label = r'$J_{ohm}$')
            axesJ[0].plot(self.rho_n, self.J_BS/1e6, lw = 2, label = r'$J_{BS}$')
            axesJ[0].plot(self.rho_n, self.avgJpara/1e6, lw = 2, label = r'$J_{||}$')
                    
            axesJ[1].plot(self.rho_n, self.J_NI/1e6, lw = 2, label = r'$J_{NI}$')
            axesJ[1].axhline(0,lw = 2, color = 'k', linestyle = 'dashed')
            axesJ[0].set_xlabel(r'$\hat{\rho}$')
            axesJ[1].set_xlabel(r'$\hat{\rho}$')
            axesJ[1].set_ylabel(r'$J_{NI}$ (MA/m^2)')
            axesJ[0].set_ylabel(r'$J$ (MA/m^2)')

            #if np.sign(J_NI[0]) > 0:
            #axesJ[1].set_ylim([-1,1.5])
            #else:
            #    axesJ[1].set_ylim([-1.5,.5])

            #axesJ[0].set_ylim([-.5,1.5])
            #
            axesJ[0].legend()
            figJ.suptitle(f'Shot {self.shot}, {self.time} ms')
            figJ.tight_layout()
            """
            
            figVER,axesVER  = plt.subplots(nrows=3, figsize = (8,8))
            axesVER[0].plot(self.rho_n, self.voltage, lw = 2)
            axesVER[0].set_ylabel('Voltage (V)')
            axesVER[0].set_xlabel(r'$\hat{\rho}$')
            #axesVER[0].set_ylim([-.7,-.05])
            #axesVER[0].set_yticks([-.65,-.45,-.25,-.05])
            axesVER[0].grid()

            axesVER[1].plot(self.rho_n, self.E_para, lw = 2)
            axesVER[1].set_ylabel('E|| (V/m)')
            axesVER[1].set_xlabel(r'$\hat{\rho}$')
            #axesVER[1].set_ylim([-.1,0])
            axesVER[1].grid()

            axesVER[2].plot(self.rho_n, sigma_neo, lw = 2)
            axesVER[2].set_ylabel(r'$\sigma_{neo}$ ($1/ \Omega m$)')
            axesVER[2].set_xlabel(r'$\hat{\rho}$')
            #axesVER[2].set_ylim([0,1.2e8])
            axesVER[2].grid()

            figVER.suptitle(f'Shot {self.shot}, {self.time} ms')
            figVER.tight_layout()
            
            """
            plt.show() 
                   
loop = pyLoop(179173, 2900.0, dt_avg = 600, efitID = 'EFIT02', useKinetic = False)
#loop = pyLoop(147634, 4505.0, dt_avg = 1000, efitID = 'EFIT02', useKinetic = False, doPlot = True, outlierTimes = [4055,4405,4605])
#loop = pyLoop(179173, 1500, dt_avg = 200, efitID = 'EFIT02er', useKinetic = False, doPlot = True, outlierTimes = [])
#loop = pyLoop(202158, 1300.0, dt_avg = 200, efitID = 'EFIT02er', doPlot = False, outlierTimes = [1100])
loop.nvloop()














