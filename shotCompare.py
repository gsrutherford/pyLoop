import pyLoop
import matplotlib.pyplot as plt
import numpy as np
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 16)
plt.rc('legend', fontsize = 14)

def compareMDSefits():
    start = 3900
    stop = 5700
    time = (start + stop)/2
    dt = stop - start

    shot = 172538
    loop_01 = pyLoop.pyLoop(shot, time, dt_avg = dt,efitID = 'EFIT01', doPlot = False)
    loop_02 = pyLoop.pyLoop(shot, time, dt_avg = dt,efitID = 'EFIT02', doPlot = False)
    loop_02er = pyLoop.pyLoop(shot, time, dt_avg = dt,efitID = 'EFIT02er', doPlot = False)
    loop_kin = pyLoop.pyLoop(shot, time, dt_avg = dt,efitID = 'EFIT02er', doPlot = False, useKinetic = True)

    loop_01.nvloop()
    loop_02.nvloop()
    loop_02er.nvloop()
    loop_kin.nvloop()

    fig,ax = plt.subplots()
    ax.set_ylabel(r'E (V/m)')
    ax.set_xlabel(r'$\hat{\rho}$')
    ax.plot(loop_01.rho_n, loop_01.E_para, label = 'EFIT01')
    ax.plot(loop_02.rho_n, loop_02.E_para, label = 'EFIT02')
    ax.plot(loop_02er.rho_n, loop_02er.E_para, label = 'EFIT02er')
    ax.plot(loop_kin.rho_n, loop_kin.E_para, label = 'kinetic')
    ax.legend()
    ax.set_title(f'{shot} {start}-{stop} ms')
    fig.tight_layout()
    plt.show()

def compareShotsToplaunch():
    start = 1500
    stop = 2300
    time = (start + stop)/2
    dt = stop - start

    toplaunch_shot = 179587
    ech_shot = 179592

    loop_topLaunch = pyLoop.pyLoop(toplaunch_shot, time, dt_avg = dt,
        efitID = 'EFIT02er', outlierTimes = [],doPlot = False)
    loop_ECH = pyLoop.pyLoop(ech_shot, time, dt_avg = dt, 
        efitID = 'EFIT02er', doPlot = False, outlierTimes = [])

    loop_topLaunch.nvloop()
    loop_ECH.nvloop()

    diff = loop_topLaunch.J_NI - loop_ECH.J_NI

    fig,ax = plt.subplots(figsize=(10,6))
    #"""
    ax.plot(loop_topLaunch.rho_n, loop_topLaunch.J_NI/1e4, lw = 2, color = 'tab:blue', label = rf'{loop_topLaunch.shot} $J_{{NI}}$')
    ax.plot(loop_ECH.rho_n, loop_ECH.J_NI/1e4, lw = 2, color = 'tab:blue', linestyle = 'dashed',label = rf'{loop_ECH.shot} $J_{{NI}}$')
    ax.plot(loop_topLaunch.rho_n, loop_topLaunch.J_ohm/1e4, lw = 2, color = 'tab:orange', label = rf'{loop_topLaunch.shot} $J_{{ohm}}$')
    ax.plot(loop_ECH.rho_n, loop_ECH.J_ohm/1e4, lw = 2, color = 'tab:orange', linestyle = 'dashed',label = rf'{loop_ECH.shot} $J_{{ohm}}$')
    ax.plot(loop_topLaunch.rho_n, loop_topLaunch.J_BS/1e4, lw = 2, color = 'tab:red', label = rf'{loop_topLaunch.shot} $J_{{BS}}$')
    ax.plot(loop_ECH.rho_n, loop_ECH.J_BS/1e4, lw = 2, color = 'tab:red', linestyle = 'dashed',label = rf'{loop_ECH.shot} $J_{{BS}}$')
    ax.plot(loop_topLaunch.rho_n, loop_topLaunch.avgJpara/1e4, lw = 2, color = 'tab:green', label = rf'{loop_topLaunch.shot} $J_{{||}}$')
    ax.plot(loop_ECH.rho_n, loop_ECH.avgJpara/1e4, lw = 2, color = 'tab:green', linestyle = 'dashed',label = rf'{loop_ECH.shot} $J_{{||}}$')
    ax.legend(ncol = 2)    
    ax.set_xlabel(r'$\rho_n$')
    ax.set_ylabel(r'J components $(A/cm^2)$')
    fig.tight_layout()
    #"""

    fig,ax = plt.subplots()
    ax.plot(loop_ECH.rho_n, diff/1e4, lw = 2)
    ax.set_xlabel(r'$\rho_n$')
    ax.set_ylabel(r'$\Delta J_{NI}$ $(A/cm^2)$')
    ax.axhline(0, color = 'k', lw = 2, linestyle = 'dashed')
    fig.tight_layout()
    plt.show()
    """
    fig,ax = plt.subplots()
    ax.plot(loop_ECH.rho_n, loop_ECH.voltage, color = 'b')
    ax.plot(loop_topLaunch.rho_n, loop_topLaunch.voltage, color = 'r')
    ax.set_xlabel(r'$\rho_n$')
    ax.set_ylabel(r'$V_{loop}$ (V)')
    ax.set_ylim([0,.12])
    ax.set_xlim([0,1])
    ax.set_box_aspect(1)
    fig.tight_layout()
    plt.show()

    fig,ax = plt.subplots()
    ax.plot(loop_ECH.rho_n, loop_ECH.J_ohm/1e3, color = 'b')
    ax.plot(loop_topLaunch.rho_n, loop_topLaunch.J_ohm/1e3, color = 'r')
    ax.set_xlabel(r'$\rho_n$')
    ax.set_ylabel(r'$J_{ohm}$ (kA/m^2)')
    ax.set_ylim([0,250])
    ax.set_xlim([0,1])
    ax.set_box_aspect(1)
    fig.tight_layout()
    plt.show()
    """

def compareTwoShots(refShot, CDShot, startTime, endTime):
    dt = endTime - startTime
    time = startTime + dt/2

    loop_ref = pyLoop.pyLoop(refShot, time, dt_avg = dt,
        efitID = 'EFIT02',doPlot = False)

    loop_CD = pyLoop.pyLoop(CDShot, time, dt_avg = dt,
        efitID = 'EFIT02',doPlot = False)

    loop_ref.nvloop()
    loop_CD.nvloop()

    fig,axes = plt.subplots(figsize = (6,9), nrows = 2)

    deltaNI = (loop_CD.J_NI - loop_ref.J_NI)
    deltaBS = (loop_CD.J_BS - loop_ref.J_BS) 
    deltaOhm = (loop_CD.J_ohm - loop_ref.J_ohm)

    axes[0].plot(loop_ref.rho_n, deltaNI/1e6, label = r'$\Delta$ $J_{NI}$', lw = 2)
    axes[0].plot(loop_ref.rho_n, deltaBS/1e6, label = r'$\Delta$ $J_{BS}$', lw = 2)
    axes[0].plot(loop_ref.rho_n, deltaOhm/1e6, label = r'$\Delta$ $J_{ohm}$', lw = 2)
    
    rho_n_midpoints = (loop_ref.rho_n[:-1] + loop_ref.rho_n[1:])/2
    dArea = loop_ref.avgCXarea[1:] - loop_ref.avgCXarea[:-1]
    dDeltaNI = (deltaNI[1:] + deltaNI[:-1])/2
    dDeltaBS = (deltaBS[1:] + deltaBS[:-1])/2
    dDeltaOhm = (deltaOhm[1:] + deltaOhm[:-1])/2

    axes[1].plot(rho_n_midpoints, np.cumsum(dArea*dDeltaNI)/1e3, label = r'$\Delta$ $J_{NI}$')
    axes[1].plot(rho_n_midpoints, np.cumsum(dArea*dDeltaBS)/1e3, label = r'$\Delta$ $J_{BS}$')
    axes[1].plot(rho_n_midpoints, np.cumsum(dArea*dDeltaOhm)/1e3, label = r'$\Delta$ $J_{ohm}$')


    axes[0].set_ylabel(rf'$J_{{{CDShot}, {{X}}}} - J_{{{refShot}, {{X}}}}$ (MA/m^2)')
    #axes[0].set_ylim([-100,100])
    axes[0].legend(ncol = 1)
    axes[0].set_xlabel(r'$\rho_n$')

    fig.suptitle(f'All shots averaged from {startTime} - {endTime} ms')
    axes[1].set_ylabel(r'$\int \Delta$ $J_{X}$ (kA)')
    #axes[1].set_ylim([-100,100])
    axes[1].legend(ncol = 1)
    axes[1].set_xlabel(r'$\rho_n$')

    fig.tight_layout()
    plt.show()


def compareShotsHelicon():
    startTime = 1110
    endTime = 1510

    compareTwoShots(202156, 202159, startTime, endTime)

compareShotsHelicon()




