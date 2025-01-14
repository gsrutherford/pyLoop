import pyLoop
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 16)
plt.rc('legend', fontsize = 14)

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

def compareShotsHelicon():
    """
    loop_202156_1340 = pyLoop.pyLoop(202156, 1340.0, dt_avg = 300, 
        efitID = 'EFIT02er', outlierTimes = [1440,1500],doPlot = False)
    loop_202156_1300 = pyLoop.pyLoop(202156, 1300.0, dt_avg = 400,
        efitID = 'EFIT02er', outlierTimes = [1440,1500],doPlot = False)

    loop_202156_1340.nvloop()
    loop_202156_1300.nvloop()
    """
    fig,ax = plt.subplots(figsize=(10,6))

    """
    ax.plot(loop_202156_1340.rho_n, loop_202156_1340.J_NI/1e4, lw = 2, color = 'tab:blue', label = r'1200-1500 $J_{NI}$')
    ax.plot(loop_202156_1300.rho_n, loop_202156_1300.J_NI/1e4, lw = 2, color = 'tab:blue', linestyle = 'dashed',label = r'1100-1500 $J_{NI}$')
    ax.plot(loop_202156_1340.rho_n, loop_202156_1340.J_ohm/1e4, lw = 2, color = 'tab:orange', label = r'1200-1500 $J_{ohm}$')
    ax.plot(loop_202156_1300.rho_n, loop_202156_1300.J_ohm/1e4, lw = 2, color = 'tab:orange', linestyle = 'dashed',label = r'1100-1500 ${ohm}$')
    ax.plot(loop_202156_1340.rho_n, loop_202156_1340.J_BS/1e4, lw = 2, color = 'tab:red', label = r'1200-1500 $J_{BS}$')
    ax.plot(loop_202156_1300.rho_n, loop_202156_1300.J_BS/1e4, lw = 2, color = 'tab:red', linestyle = 'dashed',label = r'1100-1500 $J_{BS}$')
    ax.plot(loop_202156_1340.rho_n, loop_202156_1340.avgJpara/1e4, lw = 2, color = 'tab:green', label = r'1200-1500 $J_{||}$')
    ax.plot(loop_202156_1300.rho_n, loop_202156_1300.avgJpara/1e4, lw = 2, color = 'tab:green', linestyle = 'dashed',label = r'1100-1500 $J_{||}$')
    """
    #"""
    time = 1340.0
    dt = 300

    loop_202156 = pyLoop.pyLoop(202156, time, dt_avg = dt,
        efitID = 'EFIT02er', outlierTimes = [1440,1500],doPlot = False)
    loop_202159 = pyLoop.pyLoop(202159, time, dt_avg = dt, 
        efitID = 'EFIT02er', doPlot = False, outlierTimes = [1400])
    loop_202158 = pyLoop.pyLoop(202158, time, dt_avg = dt, 
        efitID = 'EFIT02er', doPlot = False, outlierTimes = [1100])
    loop_202155 = pyLoop.pyLoop(202155, time, dt_avg = dt, efitID = 'EFIT02er', doPlot = False)

    loop_202156.nvloop()
    loop_202159.nvloop()
    loop_202158.nvloop()
    loop_202155.nvloop()

    NI_156 = loop_202156.J_NI
    NI_158 = loop_202158.J_NI
    NI_159 = loop_202159.J_NI
    NI_155 = loop_202155.J_NI

    ax.plot(loop_202156.rho_n, (NI_156 -NI_156)*1e-4, label = "no helicon or ECH", lw = 2)
    ax.plot(loop_202156.rho_n, (NI_156 -NI_159)*1e-4, label = "Helicon 710-1710", lw = 2)
    ax.plot(loop_202156.rho_n, (NI_156 -NI_158)*1e-4, label = "Helicon 1110-2110", lw = 2)
    ax.plot(loop_202156.rho_n, (NI_156 -NI_155)*1e-4, label = "ECH 710-1710", lw = 2)
    ax.set_title(f'All shots averaged from {time-dt/2} - {time+dt/2} ms')

    ax.set_ylabel(r'$J_{NI, 202156} - J_{NI,X}$ (A/cm^2)')
    ax.set_ylim([-20,100])
    #"""
    
    ax.legend(ncol = 1)
    ax.set_xlabel(r'$\rho_n$')
    #ax.set_title('Shot 202156')
    fig.tight_layout()
    plt.show()

compareShotsToplaunch()




