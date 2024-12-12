import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
from Spline_IDL import spline
import scipy.interpolate as interpolate
from getGfileDict import getGfileDict
from omfit_classes import utils_fusion
from omfit_classes import omfit_eqdsk
from omfit_classes import omfit_efit
import omfit_classes
from scipy.interpolate import interp1d
import gadata
import MDSplus as MDS


class pyLoop:

    def __init__(self,shot, efitID = 'EFIT02', nrho=101, useZipfits = True,
            t1 = None, dt_step = 50, numTimeSlices = 1, dt_avg = 50):

        self.shot = shot
        self.efitID = efitID
        self.nrho = nrho
        self.rho_n = np.linspace(0,1,nrho)
        self.useZipfits = useZipfits

        if useZipfits:
            self.collectZipfits()

        self.t1 = t1
        self.dt_step = dt_step
        self.numTimeSlices = numTimeSlices
        self.dt_avg = dt_avg
        self.timeSlices = self.t1 + self.dt_step*np.arange(0,self.numTimeSlices,1)
    
        self.f = np.zeros((numTimeSlices,nrho))
        self.eps = np.zeros((numTimeSlices,nrho))
        self.avgr = np.zeros((numTimeSlices,nrho))
        self.avgR  = np.zeros((numTimeSlices,nrho))
        self.avgB2 = np.zeros((numTimeSlices,nrho))
        self.avgBphi2 = np.zeros((numTimeSlices,nrho))
        self.avgJpara = np.zeros((numTimeSlices,nrho))
        self.q = np.zeros((numTimeSlices,nrho))
        self.ft = np.zeros((numTimeSlices,nrho))
        self.grad_term = np.zeros((numTimeSlices,nrho))
        self.fcap = np.zeros((numTimeSlices,nrho))
        self.gcap = np.zeros((numTimeSlices,nrho))
        self.hcap = np.zeros((numTimeSlices,nrho))
        self.gfileTime = np.zeros(numTimeSlices)
        self.psi_n = np.zeros((numTimeSlices,nrho))
        self.rho_bndry = np.zeros(numTimeSlices)
        self.B_0 = np.zeros(numTimeSlices)
        
        self.dpsi_dt = np.zeros((numTimeSlices,nrho))
        self.drho_bndry_dt  = np.zeros(numTimeSlices) 
        self.dB0_dt  = np.zeros(numTimeSlices)      

        self.voltage = np.zeros((numTimeSlices,nrho))
        self.E_para = np.zeros((numTimeSlices,nrho))
        self.J_ohm = np.zeros((numTimeSlices,nrho))
        self.J_BS = np.zeros((numTimeSlices,nrho))

        
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

        nC_fitData = gadata.gadata('PROFILES.ZDENSFIT', self.shot, tree = 'zipfit01')
        self.nC_fit_times = nC_fitData.ydata
        self.nC_fit_rho_n = nC_fitData.xdata
        self.nC_fit_err = nC_fitData.zerr.T*1e19
        self.nC_fits = nC_fitData.zdata*1e19
        self.nC_fits[self.nC_fits < 10] = 10
        assert (self.nC_fit_rho_n == self.ne_fit_rho_n).all()

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
    def get_gfileQuantities(self,gfileAtTime, gfilesInTimeWindow, timeIndex):

        ###First get the quantities that don't need multiple gfiles
        #We're not going to average them since there is a gfile at our chosen time
        gfile_rho_n = gfileAtTime['RHOVN']
        gfile_psi_n = gfileAtTime['fluxSurfaces']['levels']
        f = gfileAtTime['FPOL']
        eps = gfileAtTime['fluxSurfaces']['geo']['eps']
        avgr = gfileAtTime['fluxSurfaces']['avg']['a']
        avgR = gfileAtTime['fluxSurfaces']['avg']['R']
        avgB2 = gfileAtTime['fluxSurfaces']['avg']['Btot**2']
        avgBphi2 = gfileAtTime['fluxSurfaces']['avg']['Bt**2']
        avgJpara = gfileAtTime.surfAvg('Jpar')
        q = gfileAtTime['fluxSurfaces']['avg']['q']
        ft = utils_fusion.f_t(r_minor = avgr, R_major = avgR)
        grad_term = gfileAtTime['fluxSurfaces']['avg']['grad_term']
        fcap    = gfileAtTime['fluxSurfaces']['avg']['fcap']
        gcap    = gfileAtTime['fluxSurfaces']['avg']['gcap']
        hcap    = gfileAtTime['fluxSurfaces']['avg']['hcap']

        self.f[timeIndex,:] = interp1d(gfile_rho_n, f)(self.rho_n)
        self.eps[timeIndex,:] = interp1d(gfile_rho_n, eps)(self.rho_n)
        self.avgr[timeIndex,:] = interp1d(gfile_rho_n, avgr)(self.rho_n)
        self.avgR[timeIndex,:]  = interp1d(gfile_rho_n, avgR)(self.rho_n)
        self.avgB2[timeIndex,:] = interp1d(gfile_rho_n, avgB2)(self.rho_n)
        self.avgJpara[timeIndex,:] = interp1d(gfile_rho_n, avgJpara)(self.rho_n)
        self.avgBphi2[timeIndex,:] = interp1d(gfile_rho_n, avgBphi2)(self.rho_n)
        self.q[timeIndex,:] = interp1d(gfile_rho_n, q)(self.rho_n)
        self.ft[timeIndex,:] = interp1d(gfile_rho_n, ft)(self.rho_n)
        #self.grad_term[timeIndex,:] = interp1d(gfile_rho_n, grad_term)(self.rho_n)
        #self.fcap[timeIndex,:] = interp1d(gfile_rho_n, fcap)(self.rho_n)
        #self.gcap[timeIndex,:] = interp1d(gfile_rho_n, gcap)(self.rho_n)
        #self.hcap[timeIndex,:] = interp1d(gfile_rho_n, hcap)(self.rho_n)
        self.gfileTime[timeIndex] = gfileAtTime.case_info()['time']
        self.psi_n[timeIndex,:] = interp1d(gfile_rho_n, gfile_psi_n)(self.rho_n)
        self.rho_bndry[timeIndex] = gfileAtTime['fluxSurfaces']['geo']['rho'][-1]
        self.B_0[timeIndex] = gfileAtTime['BCENTR']

        #now onto the quantities that require all of the gfiles in the time window
        psis_at_psiN = np.zeros((len(gfilesInTimeWindow), len(gfile_psi_n)))
        rho_bndrys = np.zeros(len(gfilesInTimeWindow))
        B_0s = np.zeros(len(gfilesInTimeWindow))
        gfileTimes = list(gfilesInTimeWindow.keys())
        for j in range(len(gfilesInTimeWindow)):
            
            gfile = gfilesInTimeWindow[gfileTimes[j]]
            #gfile_rho_n = gfile['RHOVN']
            psis_at_psiN[j,:] = gfile['fluxSurfaces']['geo']['psi']
            #psis[j,:] = interp1d(gfile_rho_n, psi)(self.rho_n)
            rho_bndrys[j] = gfile['fluxSurfaces']['geo']['rho'][-1]
            B_0s[j] = gfile['BCENTR']

        gfileIndex = np.argmin(np.abs(gfileTimes - self.timeSlices[timeIndex]))
        drho_bndry_dt = np.gradient(rho_bndrys, gfileTimes)*1e3 #convert from 1/ms to 1/s
        self.drho_bndry_dt[timeIndex] = drho_bndry_dt[gfileIndex]   

        local_psidot = np.zeros(len(gfile_psi_n))
        for k in range(len(gfile_psi_n)):
            local_psidot[k] = np.gradient(psis_at_psiN[:,k], gfileTimes)[gfileIndex]*1e3 #convert from 1/ms to 1/s
        self.dpsi_dt[timeIndex, :] = interp1d(gfile_rho_n,local_psidot)(self.rho_n)
        self.dB0_dt[timeIndex] = np.gradient(B_0s, gfileTimes)[gfileIndex]*1e3


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
        mask = np.where((self.Te_fit_times > t-dt/2)*(self.Te_fit_times < t+dt/2))[0]
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
        mask = np.where((self.Ti_fit_times > t-dt/2)*(self.Ti_fit_times < t+dt/2))[0]
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
        mask = np.where((self.ne_fit_times > t-dt/2)*(self.ne_fit_times < t+dt/2))[0]
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
        mask = np.where((self.nC_fit_times > t-dt/2)*(self.nC_fit_times < t+dt/2))[0]
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
    def getBootstrapAndConductivity(self, t, dt, gfile, timeIndex, Z_impurity = 6):
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
        pressure = (ne*Te + nD*Ti + nC*Ti)*1.602e-19
        """
        fig,ax = plt.subplots()
        ax.plot(self.rho_n, Te)
        ax.plot(self.rho_n, Ti)
        ax2 =ax.twinx()
        ax2.plot(self.rho_n, ne, linestyle = 'dashed')
        ax2.plot(self.rho_n, nC, linestyle = 'dashed')
        plt.show()
        """
        J_BS_prof = utils_fusion.sauter_bootstrap(gEQDSKs = gfile, psi_N = self.psi_n[timeIndex], 
                Ti = np.array([Ti]), ne = np.array([ne]), Te = np.array([Te]),
                charge_number_to_use_in_ion_collisionality = 'Koh', 
                charge_number_to_use_in_ion_lnLambda = 'Koh',
                Zis=[1,6], nis = np.array([[nD], [nC]]),
                R0 = 1.6955, p = np.array([pressure]), version = 'osborne')[0]

        sigma_neo = utils_fusion.nclass_conductivity(Zeff = np.array([Zeff]), 
                psi_N = self.psi_n[timeIndex], Ti = np.array([Ti]),
                ne = np.array([ne]), Te = np.array([Te]), 
                q = self.q[timeIndex], eps = self.eps[timeIndex],
                fT = self.ft[timeIndex], R = self.avgR[timeIndex],
                Zdom = 1,
                charge_number_to_use_in_ion_collisionality = 'Koh',
                charge_number_to_use_in_ion_lnLambda = 'Koh', 
                Zis=[1,6], nis = np.array([[nD], [nC]]))[0]
        """
        sigma_neo = utils_fusion.nclass_conductivity_from_gfile(gEQDSK = gfile, 
                Zeff = np.array([Zeff]), psi_N = self.psi_n[timeIndex], Ti = np.array([Ti]),
                ne = np.array([ne]), Te = np.array([Te]), 
                charge_number_to_use_in_ion_collisionality = 'Koh',
                charge_number_to_use_in_ion_lnLambda = 'Koh', 
                Zis=[1,6], nis = np.array([[nD], [nC]]))[0]
        """
        return J_BS_prof, sigma_neo


    #shot   = shot number
    # t1     = time for first vloop analysis (msec)
    # dtstep = interval between vloop analyses (msec) 
    #             (t1, t1+dtstep, t1+2*dtstep, ...)
    # nvlt   = number of vloop analyses requested
    # dtavg  = averaging time window for each analysis (msec)
    #             (t-dtavg/2 to t+dtavg/2)
    def nvloop(self):

        #efitTimeNode = tree.getNode('RESULTS.GEQDSK.GTIME')
        gtimes = gadata.gadata('RESULTS.GEQDSK.GTIME', self.shot, tree = self.efitID).zdata

        for timeIndex in range(len(self.timeSlices)):
            time = self.timeSlices[timeIndex]
            mask = np.where((gtimes >= time - self.dt_avg/2)*(gtimes <= time + self.dt_avg/2))
            relevantGfileTimes = gtimes[mask]

            if time not in relevantGfileTimes:
                print('***')
                print('Requested gfile time is not in tree, skipping it')
                print('***')
                continue

            gfilesInTimeWindow = omfit_eqdsk.from_mds_plus(device = 'd3d',shot = self.shot, 
                        times = relevantGfileTimes, snap_file = self.efitID, 
                        get_afile = False, get_mfile = False, quiet = True,
                        time_diff_warning_threshold = 1)['gEQDSK']

            gfileAtTime = gfilesInTimeWindow[float(time)]
            self.get_gfileQuantities(gfileAtTime, gfilesInTimeWindow, timeIndex)

            J_BS, sigma_neo = self.getBootstrapAndConductivity(time, self.dt_avg, 
                                gfileAtTime,timeIndex)
            self.J_BS[timeIndex, :] = J_BS
            
            vv1 = 2*np.pi*self.rho_n**2*(self.B_0[timeIndex]/self.q[timeIndex,:])*self.rho_bndry[timeIndex]*self.drho_bndry_dt[timeIndex]
            vv2 = np.pi*self.rho_bndry[timeIndex]**2*self.rho_n**2*self.dB0_dt[timeIndex]/(self.q[timeIndex])
            self.voltage[timeIndex,:] = 2*np.pi*self.dpsi_dt[timeIndex,:] - vv1

            #"""
            self.E_para[timeIndex,:] = ((self.voltage[timeIndex,:]-vv2)*
                        self.avgBphi2[timeIndex]/(2*np.pi*self.B_0[timeIndex]*self.f[timeIndex]))

            self.J_ohm[timeIndex,:] = self.E_para[timeIndex,:]*sigma_neo

            fig,axes = plt.subplots(nrows = 2, ncols = 2)
            axes[0,0].plot(self.rho_n, np.abs(self.q[timeIndex,:]), lw = 2, label = r'$q$')
            axes[0,0].set_ylabel('q')
            axes[0,0].set_xlabel(r'$\hat{\rho}$')
            axes[0,0].set_ylim([1,7])

            axes[0,1].plot(self.rho_n, np.abs(self.avgJpara[timeIndex,:])/1e6, lw = 2, label = r'$J_{||}$')
            axes[0,1].set_ylabel(r'<$J_{||}$> (MA/m^2)')
            axes[0,1].set_xlabel(r'$\hat{\rho}$')
            axes[0,1].set_ylim([.2,1.2])

            #axes[0,2].plot(self.rho_n, self.dpsi_dt[timeIndex,:], lw = 2, label = r'$dpsi_dt$')
            #axes[0,2].set_ylabel(r'$\dot{\psi}$')
            #axes[0,2].set_xlabel(r'$\hat{\rho}$')


            axes[1,0].plot(self.rho_n, self.voltage[timeIndex,:], lw = 2, label = r'$V_\phi$')
            axes[1,0].set_ylabel(r'$V_\phi$ (V)')
            axes[1,0].set_xlabel(r'$\hat{\rho}$')
            #axes[1,0].set_ylim([-.15,.1])

            axes[1,1].plot(self.rho_n, self.E_para[timeIndex,:], lw = 2)
            axes[1,1].set_ylabel(r'$E_{||}$ (V/m)')
            axes[1,1].set_xlabel(r'$\hat{\rho}$')
            #axes[1,1].set_ylim([-.015,.01])

            #axes[1,2].plot(self.rho_n, self.ft[timeIndex,:], lw = 2, label = r'$f_t$')
            #axes[1,2].set_ylabel('trapped fraction')
            #axes[1,2].set_xlabel(r'$\hat{\rho}$')
            #axes[1,2].set_ylim([0,.8])

            fig.tight_layout()
            #"""
            figJ, axesJ = plt.subplots(nrows = 2)
            axesJ[0].plot(self.rho_n, self.J_ohm[timeIndex,:]/1e6, lw = 2, label = r'$J_{ohm}$')
            axesJ[0].plot(self.rho_n, self.J_BS[timeIndex,:]/1e6, lw = 2, label = r'$J_{BS}$')
            axesJ[0].plot(self.rho_n, np.abs(self.avgJpara[timeIndex,:])/1e6, lw = 2, label = r'$J_{||}$')
            J_NI =  np.abs(self.avgJpara[timeIndex,:])-self.J_BS[timeIndex,:]-self.J_ohm[timeIndex,:]           
            axesJ[1].plot(self.rho_n, J_NI/1e6, lw = 2, label = r'$J_{NI}$')
            axesJ[1].axhline(0,lw = 2, color = 'k', linestyle = 'dashed')
            axesJ[0].set_xlabel(r'$\hat{\rho}$')
            axesJ[1].set_xlabel(r'$\hat{\rho}$')
            axesJ[1].set_ylabel(r'$J_{NI}$ (MA/m^2)')
            axesJ[0].set_ylabel(r'$J$ (MA/m^2)')
            axesJ[0].set_ylim([-.5,1.5])
            axesJ[1].set_ylim([-.5,1.5])
            axesJ[0].legend()
            #"""
            fig.tight_layout()
            plt.show()
            #"""
 #use 03 for compare since it's broken           
loop = pyLoop(shot = 147634, t1 = 4555, dt_step = 100, numTimeSlices = 1, 
    dt_avg = 100, efitID = 'EFIT02')
loop.nvloop()

