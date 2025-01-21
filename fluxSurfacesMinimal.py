from omfit_classes import fluxSurface
import time
from omfit_classes.sortedDict import SortedDict
import numpy as np
from scipy import integrate, interpolate, constants
from omfit_classes.utils_math import (
    parabola,
    paraboloid,
    parabolaMaxCycle,
    contourPaths,
    reverse_enumerate,
    RectBivariateSplineNaN,
    deriv,
    line_intersect,
    interp1e,
    centroid,
    pack_points,
)

class fluxSurfaces_minimal(fluxSurface.fluxSurfaces):

    def surfAvg(self, function=None):
        startTime = time.time()
        """
        Flux surface averaged quantity for each flux surface

        :param function: function which returns the value of the quantity to be flux surface averaged at coordinates r,z

        :return: array of the quantity fluxs surface averaged for each flux surface

        :Example:

        >> def test_avg_function(r, z):
        >>     return RectBivariateSplineNaN(Z, R, PSI, k=1).ev(z,r)

        """
        #t0 = datetime.datetime.now()
        if not self.calculateAvgGeo:
            return

        self._BrBzAndF()

        # define handy function for flux-surface averaging
        def flxAvg(k, input):
            return np.sum(self.fluxexpansion_dl[k] * input) / self.int_fluxexpansion_dl[k]
        print(f'self.nc: {self.nc}')
        # if user wants flux-surface averaging of a specific function, then calculate and return it
        if function is not None:
            if 'avg' not in self:
                self.surfAvg()
            avg = np.zeros((self.nc))
            for k in range(self.nc):
                avg[k] = flxAvg(k, function(self['flux'][k]['R'], self['flux'][k]['Z']))
            return avg

        self['avg'] = SortedDict()
        self['geo'] = SortedDict()
        self['geo']['psi'] = self['levels'] * (self.flx - self.PSIaxis) + self.PSIaxis

        # calculate flux surface average of typical quantities
        if not self.quiet:
            printi('Flux surface averaging ...')
        for item in [
            'R',
            'a',
            #'R**2',
            #'1/R',
            '1/R**2',
            'Bp',
            #'Bp**2',
            #'Bp*R',
            #'Bp**2*R**2',
            #'Btot',
            'Btot**2',
            #'Bt',
            'Bt**2',
            #'ip',
            'vp',
            'q',
            #'hf',
            #'Jt',
            #'Jt/R',
            #'fc',
            #'grad_term',
            #'P',
            #'F',
            #'PPRIME',
            #'FFPRIM',
        ]:
            self['avg'][item] = np.zeros((self.nc))
        
        beforeLoop = time.time()

        for k in range(self.nc):
            Bp2 = self['flux'][k]['Br'] ** 2 + self['flux'][k]['Bz'] ** 2
            signBp = (
                self._cocos['sigma_rhotp']
                * self._cocos['sigma_RpZ']
                * np.sign(
                    (self['flux'][k]['Z'] - self['Z0']) * self['flux'][k]['Br']
                    - (self['flux'][k]['R'] - self['R0']) * self['flux'][k]['Bz']
                )
            )
            Bp = signBp * np.sqrt(Bp2)
            Bt = self['flux'][k]['F'] / self['flux'][k]['R']
            B2 = Bp2 + Bt**2
            B = np.sqrt(B2)
            bratio = B / np.max(B)
            #self['flux'][k]['Bmax'] = np.max(B)

            # self['avg']['psi'][k]       = flxAvg(k, function(self['flux'][k]['R'],self['flux'][k]['Z']) )
            self['avg']['R'][k] = flxAvg(k, self['flux'][k]['R'])
            self['avg']['a'][k] = flxAvg(k, np.sqrt((self['flux'][k]['R'] - self['R0']) ** 2 + (self['flux'][k]['Z'] - self['Z0']) ** 2))
            #self['avg']['R**2'][k] = flxAvg(k, self['flux'][k]['R'] ** 2)
            #self['avg']['1/R'][k] = flxAvg(k, 1.0 / self['flux'][k]['R'])
            self['avg']['1/R**2'][k] = flxAvg(k, 1.0 / self['flux'][k]['R'] ** 2)
            self['avg']['Bp'][k] = flxAvg(k, Bp)
            #self['avg']['Bp**2'][k] = flxAvg(k, Bp2)
            #self['avg']['Bp*R'][k] = flxAvg(k, Bp * self['flux'][k]['R'])
            #self['avg']['Bp**2*R**2'][k] = flxAvg(k, Bp2 * self['flux'][k]['R'] ** 2)
            #self['avg']['Btot'][k] = flxAvg(k, B)
            self['avg']['Btot**2'][k] = flxAvg(k, B2)
            #self['avg']['Bt'][k] = flxAvg(k, Bt)
            self['avg']['Bt**2'][k] = flxAvg(k, Bt**2)

            self['avg']['vp'][k] = (
                self._cocos['sigma_rhotp']
                * self._cocos['sigma_Bp']
                * np.sign(self['avg']['Bp'][k])
                * self.int_fluxexpansion_dl[k]
                * (2.0 * np.pi) ** (1.0 - self._cocos['exp_Bp'])
            )
            self['avg']['q'][k] = (
                self._cocos['sigma_rhotp']
                * self._cocos['sigma_Bp']
                * self['avg']['vp'][k]
                * self['flux'][k]['F']
                * self['avg']['1/R**2'][k]
                / ((2 * np.pi) ** (2.0 - self._cocos['exp_Bp']))
            )
            #grad_parallel = np.diff(B) / self.fluxexpansion_dl[k][1:] / B[1:]
            #self['avg']['grad_term'][k] = np.sum(self.fluxexpansion_dl[k][1:] * grad_parallel**2) / self.int_fluxexpansion_dl[k]
        print(f'{time.time() - beforeLoop} to get through first for loop')
        # q on axis by extrapolation
        if self['levels'][0] == 0:
            x = self['levels'][1:]
            y = self['avg']['q'][1:]
            self['avg']['q'][0] = y[1] - ((y[1] - y[0]) / (x[1] - x[0])) * x[1]

        for k in range(self.nc):
            self['flux'][k]['q'] = self['avg']['q'][k]

        for k in range(self.nc):
            geo = fluxGeo(self['flux'][k]['R'], self['flux'][k]['Z'], lcfs=(k == (self.nc - 1)))
            for item in sorted(geo):
                if item not in self['geo']:
                    self['geo'][item] = np.zeros((self.nc))
                self['geo'][item][k] = geo[item]
        self['geo']['phi'] = (
            self._cocos['sigma_Bp']
            * self._cocos['sigma_rhotp']
            * integrate.cumtrapz(self['avg']['q'], self['geo']['psi'], initial=0)
            * (2.0 * np.pi) ** (1.0 - self._cocos['exp_Bp'])
        )

        # fix geometric quantities on axis
        if self['levels'][0] == 0:

            #  calculate rho only if levels start from 0
            self['geo']['rho'] = np.sqrt(np.abs(self['geo']['phi'] / (np.pi * self['BCENTR'])))
        else:
            # the values of phi, rho have meaning only if I can integrate from the first flux surface on...
            if 'phi' in self['geo']:
                del self['geo']['phi']
            if 'rho' in self['geo']:
                del self['geo']['rho']

        # the values of rhon has a meaning only if I have the value at the lcfs
        if 'rho' in self['geo'] and self['levels'][self.nc - 1] == 1.0:
            self['geo']['rhon'] = self['geo']['rho'] / max(self['geo']['rho'])

        else:
            if 'rhon' in self['geo']:
                del self['geo']['rhon']
        print(f'{time.time() - startTime} to get through fluxSurfaces')

def fluxGeo(inputR, inputZ, lcfs=False, doPlot=False):
    '''
    Calculate geometric properties of a single flux surface

    :param inputR: R points

    :param inputZ: Z points

    :param lcfs: whether this is the last closed flux surface (for sharp feature of x-points)

    :param doPlot: plot geometric measurements

    :return: dictionary with geometric quantities
    '''
    # Cast as arrays
    inputR = np.array(inputR)
    inputZ = np.array(inputZ)

    # Make sure the flux surfaces are closed
    if inputR[0] != inputR[1]:
        inputRclose = np.hstack((inputR, inputR[0]))
        inputZclose = np.hstack((inputZ, inputZ[0]))
    else:
        inputRclose = inputR
        inputZclose = inputZ
        inputR = inputR[:-1]
        inputR = inputZ[:-1]

    # This is the result
    geo = SortedDict()

    # These are the extrema indices
    imaxr = np.argmax(inputR)
    iminr = np.argmin(inputR)
    imaxz = np.argmax(inputZ)
    iminz = np.argmin(inputZ)

    # Find the extrema points
    if lcfs:
        r_at_max_z, max_z = inputR[imaxz], inputZ[imaxz]
        r_at_min_z, min_z = inputR[iminz], inputZ[iminz]
        z_at_max_r, max_r = inputZ[imaxr], inputR[imaxr]
        z_at_min_r, min_r = inputZ[iminr], inputR[iminr]
    else:
        r_at_max_z, max_z = parabolaMaxCycle(inputR, inputZ, imaxz, bounded='max')
        r_at_min_z, min_z = parabolaMaxCycle(inputR, inputZ, iminz, bounded='min')
        z_at_max_r, max_r = parabolaMaxCycle(inputZ, inputR, imaxr, bounded='max')
        z_at_min_r, min_r = parabolaMaxCycle(inputZ, inputR, iminr, bounded='min')

    #dl = np.sqrt(np.ediff1d(inputR, to_begin=0) ** 2 + np.ediff1d(inputZ, to_begin=0) ** 2)
    geo['R'] = 0.5 * (max_r + min_r)
    #geo['Z'] = 0.5 * (max_z + min_z)
    geo['a'] = 0.5 * (max_r - min_r)
    geo['eps'] = geo['a'] / geo['R']
    return geo



