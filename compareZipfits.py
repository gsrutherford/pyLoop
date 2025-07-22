import gadata
import numpy as np
import matplotlib.pyplot as plt

def makePlot():

    fig,axes = plt.subplots(nrows = 2, ncols = 2)
    axes = axes.flatten()
    print(axes)
    axes[0].set_xlabel(r'$\rho_n$')
    axes[1].set_xlabel(r'$\rho_n$')
    axes[2].set_xlabel(r'$\rho_n$')
    axes[3].set_xlabel(r'$\rho_n$')
    
    axes[0].set_ylabel(r'$T_e$ (keV)')
    axes[1].set_ylabel(r'$T_i$ (keV)')
    axes[2].set_ylabel(r'$n_e$ ($m^{-3}$)')
    axes[3].set_ylabel(r'$n_C$ ($m^{-3}$)')

    avgAndPlotZipfits(199749, 2500, 3500, axes)
    avgAndPlotZipfits(147634, 2500, 3500, axes)

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[3].legend()

    fig.tight_layout()
    plt.show()


def avgAndPlotZipfits(shot, startTime, endTime, axes):
    Te_fitData = gadata.gadata('PROFILES.ETEMPFIT', shot, tree = 'zipfit01')
    Te_fit_times = Te_fitData.ydata
    Te_fit_rho_n = Te_fitData.xdata
    Te_fits = Te_fitData.zdata
    #if the value is less than 1 eV, set it to 1 eV
    Te_fits[Te_fits < 1/1000] = 1/1000 

    Temask = np.where((Te_fit_times >= startTime)*(Te_fit_times <= endTime))[0]
    print(f'rho_n, times, fit: {Te_fit_rho_n.shape, Te_fit_times.shape, Te_fits.shape}')
    axes[0].plot(Te_fit_rho_n, np.average(Te_fits[:,Temask], axis = 1), label = f'{shot}', lw = 2)
    
    Ti_fitData = gadata.gadata('PROFILES.ITEMPFIT', shot, tree = 'zipfit01')
    Ti_fit_times = Ti_fitData.ydata
    Ti_fit_rho_n = Ti_fitData.xdata
    Ti_fits = Ti_fitData.zdata
    #if the value is less than 1 eV, set it to 1 eV
    Ti_fits[Ti_fits < 1/1000] = 1/1000 

    Timask = np.where((Ti_fit_times >= startTime)*(Ti_fit_times <= endTime))[0]
    axes[2].plot(Ti_fit_rho_n, np.average(Ti_fits[:,Timask], axis = 1), label = f'{shot}', lw = 2)

    ne_fitData = gadata.gadata('PROFILES.EDENSFIT', shot, tree = 'zipfit01')
    ne_fit_times = ne_fitData.ydata
    ne_fit_rho_n = ne_fitData.xdata
    ne_fits = ne_fitData.zdata*1e19
    #if the value is less than 10 m^-3, set it to 10 m^-3
    ne_fits[ne_fits < 10] = 10
    
    nemask = np.where((ne_fit_times >= startTime)*(ne_fit_times <= endTime))[0]
    axes[1].plot(ne_fit_rho_n, np.average(ne_fits[:,nemask], axis = 1), label = f'{shot}', lw = 2)

    try:
        nC_fitData = gadata.gadata('PROFILES.ZDENSFIT', shot, tree = 'zipfit01')
        nC_fit_times = nC_fitData.ydata
        nC_fit_rho_n = nC_fitData.xdata
        nC_fits = nC_fitData.zdata*1e19
        nC_fits[nC_fits < 10] = 10
    except:
        print(f'Something wrong with carbon data, taking Zeff = 1')
        nC_fitData = gadata.gadata('PROFILES.EDENSFIT', shot, tree = 'zipfit01')
        nC_fit_times = ne_fitData.ydata
        nC_fit_rho_n = ne_fitData.xdata
        nC_fits = ne_fitData.zdata*1e19
        #if the value is less than 10 m^-3, set it to 10 m^-3
        nC_fits[nC_fits < 10] = 10
    assert (nC_fit_rho_n == ne_fit_rho_n).all()

    nCmask = np.where((nC_fit_times >= startTime)*(nC_fit_times <= endTime))[0]
    axes[3].plot(nC_fit_rho_n, np.average(nC_fits[:,nCmask], axis = 1), label = f'{shot}', lw = 2)


if __name__ == '__main__':
    makePlot()








