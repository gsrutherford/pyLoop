from omfit_classes import fluxSurface
import time
from omfit_classes.sortedDict import SortedDict
import numpy as np
from scipy import integrate, interpolate, constants

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
        #self['midplane'] = SortedDict()
        self['geo']['psi'] = self['levels'] * (self.flx - self.PSIaxis) + self.PSIaxis
        #self['geo']['psin'] = self['levels']

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
            'Btot',
            'Btot**2',
            'Bt',
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
            self['avg']['Btot'][k] = flxAvg(k, B)
            self['avg']['Btot**2'][k] = flxAvg(k, B2)
            self['avg']['Bt'][k] = flxAvg(k, Bt)
            self['avg']['Bt**2'][k] = flxAvg(k, Bt**2)
            #"""
            self['avg']['vp'][k] = (
                self._cocos['sigma_rhotp']
                * self._cocos['sigma_Bp']
                * np.sign(self['avg']['Bp'][k])
                * self.int_fluxexpansion_dl[k]
                * (2.0 * np.pi) ** (1.0 - self._cocos['exp_Bp'])
            )
            #"""
            self['avg']['q'][k] = (
                self._cocos['sigma_rhotp']
                * self._cocos['sigma_Bp']
                * self['avg']['vp'][k]
                * self['flux'][k]['F']
                * self['avg']['1/R**2'][k]
                / ((2 * np.pi) ** (2.0 - self._cocos['exp_Bp']))
            )
            #self['avg']['hf'][k] = flxAvg(k, (1.0 - np.sqrt(1.0 - bratio) * (1.0 + bratio / 2.0)) / bratio**2)

            # these quantites are calculated from Bx,By and hence are not that precise
            # if information is available about P, PPRIME, and F, FFPRIM then they will be substittuted
            """
            self['avg']['ip'][k] = self._cocos['sigma_rhotp'] * np.sum(self.dl[k] * Bp) / (4e-7 * np.pi)
            self['avg']['Jt'][k] = flxAvg(k, self['flux'][k]['Jt'])
            self['avg']['Jt/R'][k] = flxAvg(k, self['flux'][k]['Jt'] / self['flux'][k]['R'])
            self['avg']['F'][k] = self['flux'][k]['F']
            if hasattr(self, 'P'):
                self['avg']['P'][k] = self['flux'][k]['P']
            elif 'P' in self['avg']:
                del self['avg']['P']
            if hasattr(self, 'PPRIME'):
                self['avg']['PPRIME'][k] = self['flux'][k]['PPRIME']
            elif 'PPRIME' in self['avg']:
                del self['avg']['PPRIME']
            if hasattr(self, 'FFPRIM'):
                self['avg']['FFPRIM'][k] = self['flux'][k]['FFPRIM']
            elif 'FFPRIM' in self['avg']:
                del self['avg']['FFPRIM']
            """
            """
            ## The circulating particle fraction calculation has been converted from IDL to python
            ## following the fraction_circ.pro which is widely used at DIII-D
            ## Formula 4.54 of S.P. Hirshman and D.J. Sigmar 1981 Nucl. Fusion 21 1079
            # x=np.array([0.0387724175, 0.1160840706, 0.1926975807, 0.268152185, 0.3419940908, 0.4137792043, 0.4830758016, 0.549467125, 0.6125538896, 0.6719566846, 0.7273182551, 0.7783056514, 0.8246122308, 0.8659595032, 0.9020988069, 0.9328128082, 0.9579168192, 0.9772599499, 0.9907262386, 0.9982377097])
            # w=np.array([0.0775059479, 0.0770398181, 0.0761103619, 0.074723169, 0.0728865823, 0.0706116473, 0.0679120458, 0.0648040134, 0.0613062424, 0.057439769, 0.0532278469, 0.0486958076, 0.0438709081, 0.0387821679, 0.0334601952, 0.0279370069, 0.0222458491, 0.0164210583, 0.0104982845, 0.004521277])
            # lmbd   = 1-x**2
            # weight = 2.*w*np.sqrt(1. - lmbd)
            # denom  = np.zeros((len(lmbd)))
            # for n in range(len(lmbd)):
            #    denom[n] = flxAvg(k, np.sqrt(1.-lmbd[n]*bratio) )
            # integral=np.sum(weight*lmbd/denom)
            # self['avg']['fc'][k]        =0.75*self['avg']['Btot**2'][k]/np.max(B)**2*integral
            #
            # The above calculation is exactly equivalent to the Lin-Lu form of trapped particle fraction
            # article: Y.R. Lin-Liu and R.L. Miller, Phys. of Plamsas 2 (1995) 1666
            h = self['avg']['Btot'][k] / self['flux'][k]['Bmax']
            h2 = self['avg']['Btot**2'][k] / self['flux'][k]['Bmax'] ** 2
            # Equation 4
            ftu = 1.0 - h2 / (h**2) * (1.0 - np.sqrt(1.0 - h) * (1.0 + 0.5 * h))
            # Equation 7
            ftl = 1.0 - h2 * self['avg']['hf'][k]
            # Equation 18,19
            self['avg']['fc'][k] = 1 - (0.75 * ftu + 0.25 * ftl)
            """
            #grad_parallel = np.diff(B) / self.fluxexpansion_dl[k][1:] / B[1:]
            #self['avg']['grad_term'][k] = np.sum(self.fluxexpansion_dl[k][1:] * grad_parallel**2) / self.int_fluxexpansion_dl[k]

        # q on axis by extrapolation
        if self['levels'][0] == 0:
            x = self['levels'][1:]
            y = self['avg']['q'][1:]
            self['avg']['q'][0] = y[1] - ((y[1] - y[0]) / (x[1] - x[0])) * x[1]

        for k in range(self.nc):
            self['flux'][k]['q'] = self['avg']['q'][k]
        """
        if 'P' in self['avg'] and 'PPRIME' not in self['avg']:
            self['avg']['PPRIME'] = deriv(self['geo']['psi'], self['avg']['P'])

        if 'F' in self['avg'] and 'FFPRIM' not in self['avg']:
            self['avg']['FFPRIM'] = self['avg']['F'] * deriv(self['geo']['psi'], self['avg']['F'])

        if 'PPRIME' in self['avg'] and 'FFPRIM' in self['avg']:
            self['avg']['Jt/R'] = (
                -self._cocos['sigma_Bp']
                * (self['avg']['PPRIME'] + self['avg']['FFPRIM'] * self['avg']['1/R**2'] / (4 * np.pi * 1e-7))
                * (2.0 * np.pi) ** self._cocos['exp_Bp']
            )
        """
        """
        # calculate currents based on Grad-Shafranov if pressure information is available
        # TEMPORARILY DISABLED: issue at first knot when looking at Jeff which is near zero on axis
        if False and 'PPRIME' in self['avg'] and 'F' in self['avg'] and 'FFPRIM' in self['avg']:
            self['avg']['dip/dpsi'] = (
                -self._cocos['sigma_Bp']
                * self['avg']['vp']
                * (self['avg']['PPRIME'] + self['avg']['FFPRIM'] * self['avg']['1/R**2'] / (4e-7 * np.pi))
                / ((2 * np.pi) ** (1.0 - self._cocos['exp_Bp']))
            )
            self['avg']['ip'] = integrate.cumtrapz(self['avg']['dip/dpsi'], self['geo']['psi'], initial=0)
        else:
            self['avg']['dip/dpsi'] = deriv(self['geo']['psi'], self['avg']['ip'])
        self['avg']['Jeff'] = (
            self._cocos['sigma_Bp']
            * self._cocos['sigma_rhotp']
            * self['avg']['dip/dpsi']
            * self['BCENTR']
            / (self['avg']['q'] * (2 * np.pi) ** (1.0 - self._cocos['exp_Bp']))
        )
        self['CURRENT'] = self['avg']['ip'][-1]

        # calculate geometric quantities
        if not self.quiet:
            printi('  > Took {:}'.format(datetime.datetime.now() - t0))
        if not self.quiet:
            printi('Geometric quantities ...')
        """
        #t0 = datetime.datetime.now()

        for k in range(self.nc):
            geo = fluxSurface.fluxGeo(self['flux'][k]['R'], self['flux'][k]['Z'], lcfs=(k == (self.nc - 1)))
            for item in sorted(geo):
                if item not in self['geo']:
                    self['geo'][item] = np.zeros((self.nc))
                self['geo'][item][k] = geo[item]
        #self['geo']['vol'] = np.abs(self.volume_integral(1))
        #self['geo']['cxArea'] = np.abs(self.surface_integral(1))
        self['geo']['phi'] = (
            self._cocos['sigma_Bp']
            * self._cocos['sigma_rhotp']
            * integrate.cumtrapz(self['avg']['q'], self['geo']['psi'], initial=0)
            * (2.0 * np.pi) ** (1.0 - self._cocos['exp_Bp'])
        )
        # self['geo']['bunit']=(abs(self['avg']['q'])/self['geo']['a'])*( deriv(self['geo']['a'],self['geo']['psi']) )
        #self['geo']['bunit'] = deriv(self['geo']['a'], self['geo']['phi']) / (2.0 * np.pi * self['geo']['a'])

        # fix geometric quantities on axis
        if self['levels'][0] == 0:
            """
            self['geo']['delu'][0] = 0.0
            self['geo']['dell'][0] = 0.0
            self['geo']['delta'][0] = 0.0
            self['geo']['zeta'][0] = 0.0
            self['geo']['zetaou'][0] = 0.0
            self['geo']['zetaiu'][0] = 0.0
            self['geo']['zetail'][0] = 0.0
            self['geo']['zetaol'][0] = 0.0
            # linear extrapolation
            x = self['levels'][1:]
            for item in ['kapu', 'kapl']:#, 'bunit']:
                y = self['geo'][item][1:]
                self['geo'][item][0] = y[1] - ((y[1] - y[0]) / (x[1] - x[0])) * x[1]
            self['geo']['kap'][0] = 0.5 * self['geo']['kapu'][0] + 0.5 * self['geo']['kapl'][0]
            """
            #  calculate rho only if levels start from 0
            self['geo']['rho'] = np.sqrt(np.abs(self['geo']['phi'] / (np.pi * self['BCENTR'])))
        else:
            # the values of phi, rho have meaning only if I can integrate from the first flux surface on...
            if 'phi' in self['geo']:
                del self['geo']['phi']
            if 'rho' in self['geo']:
                del self['geo']['rho']

        # calculate betas
        """
        if 'P' in self['avg']:
            Btvac = self['BCENTR'] * self['RCENTR'] / self['geo']['R'][-1]
            self['avg']['beta_t'] = abs(
                self.volume_integral(self['avg']['P']) / (Btvac**2 / 2.0 / 4.0 / np.pi / 1e-7) / self['geo']['vol'][-1]
            )
            i = self['CURRENT'] / 1e6
            a = self['geo']['a'][-1]
            self['avg']['beta_n'] = self['avg']['beta_t'] / abs(i / a / Btvac) * 100
            Bpave = self['CURRENT'] * (4 * np.pi * 1e-7) / self['geo']['per'][-1]
            self['avg']['beta_p'] = abs(
                self.volume_integral(self['avg']['P']) / (Bpave**2 / 2.0 / 4.0 / np.pi / 1e-7) / self['geo']['vol'][-1]
            )
        """

        # the values of rhon has a meaning only if I have the value at the lcfs
        if 'rho' in self['geo'] and self['levels'][self.nc - 1] == 1.0:
            self['geo']['rhon'] = self['geo']['rho'] / max(self['geo']['rho'])

            # fcap, f(psilim)/f(psi)
            #self['avg']['fcap'] = np.zeros((self.nc))
            #for k in range(self.nc):
            #    self['avg']['fcap'][k] = self['flux'][self.nc - 1]['F'] / self['flux'][k]['F']

            # hcap, fcap / <R0**2/R**2>
            #self['avg']['hcap'] = self['avg']['fcap'] / (self['RCENTR'] ** 2 * self['avg']['1/R**2'])
            """
            # RHORZ (linear extrapolation for rho>1)
            def ext_arr_linear(x, y):
                dydx = (y[-1] - y[-2]) / (x[-1] - x[-2])
                extra_x = (x[-1] - x[-2]) * np.r_[1:1000] + x[-1]
                extra_y = (x[-1] - x[-2]) * np.r_[1:1000] * dydx + y[-1]
                x = np.hstack((x, extra_x))
                y = np.hstack((y, extra_y))
                return [x, y]

            [new_psi_mesh0, new_PHI] = ext_arr_linear(self['geo']['psi'], self['geo']['phi'])
            PHIRZ = interpolate.interp1d(new_psi_mesh0, new_PHI, kind='linear', bounds_error=False)(self.PSIin)
            RHORZ = np.sqrt(abs(PHIRZ / np.pi / self['BCENTR']))

            # gcap <(grad rho)**2*(R0/R)**2>
            dRHOdZ, dRHOdR = np.gradient(RHORZ, self.Zin[2] - self.Zin[1], self.Rin[2] - self.Rin[1])
            dPHI2 = dRHOdZ**2 + dRHOdR**2
            dp2fun = RectBivariateSplineNaN(self.Zin, self.Rin, dPHI2)
            self['avg']['gcap'] = np.zeros((self.nc))
            for k in range(self.nc):
                self['avg']['gcap'][k] = (
                    np.sum(self.fluxexpansion_dl[k] * dp2fun.ev(self['flux'][k]['Z'], self['flux'][k]['R']) / self['flux'][k]['R'] ** 2)
                    / self.int_fluxexpansion_dl[k]
                )
            self['avg']['gcap'] *= self['RCENTR'] ** 2  # * self['avg']['1/R**2']

            # linear extrapolation
            x = self['levels'][1:]
            for item in ['gcap', 'hcap', 'fcap']:
                y = self['avg'][item][1:]
                self['avg'][item][0] = y[1] - ((y[1] - y[0]) / (x[1] - x[0])) * x[1]
            """
        else:
            if 'rhon' in self['geo']:
                del self['geo']['rhon']
        """
        # midplane quantities
        self['midplane']['R'] = self['geo']['R'] + self['geo']['a']
        self['midplane']['Z'] = self['midplane']['R'] * 0 + self['Z0']

        Br, Bz = self._calcBrBz()
        self['midplane']['Br'] = RectBivariateSplineNaN(self.Zin, self.Rin, Br).ev(self['midplane']['Z'], self['midplane']['R'])
        self['midplane']['Bz'] = RectBivariateSplineNaN(self.Zin, self.Rin, Bz).ev(self['midplane']['Z'], self['midplane']['R'])

        signBp = -self._cocos['sigma_rhotp'] * self._cocos['sigma_RpZ'] * np.sign(self['midplane']['Bz'])
        self['midplane']['Bp'] = signBp * np.sqrt(self['midplane']['Br'] ** 2 + self['midplane']['Bz'] ** 2)

        self['midplane']['Bt'] = []
        for k in range(self.nc):
            self['midplane']['Bt'].append(self['flux'][k]['F'] / self['midplane']['R'][k])
        self['midplane']['Bt'] = np.array(self['midplane']['Bt'])
        """
        """
        # ============
        # extra infos
        # ============
        self['info'] = SortedDict()

        # Normlized plasma inductance
        # * calculated using {Inductive flux usage and its optimization in tokamak operation T.C.Luce et al.} EQ (A2,A3,A4)
        # * ITER IMAS li3
        ip = self['CURRENT']
        vol = self['geo']['vol'][-1]
        dpsi = np.abs(np.gradient(self['geo']['psi']))
        r_axis = self['R0']
        a = self['geo']['a'][-1]
        if self['RCENTR'] is None:
            printw('Using magnetic axis as RCENTR of vacuum field ( BCENTR = Fpol[-1] / RCENTR)')
            r_0 = self['R0']
        else:
            r_0 = self['RCENTR']
        kappa_x = self['geo']['kap'][-1]  # should be used if
        kappa_a = vol / (2.0 * np.pi * r_0 * np.pi * a * a)
        correction_factor = (1 + kappa_x**2) / (2.0 * kappa_a)
        Bp2_vol = 0
        for k in range(self.nc):  # loop over the flux surfaces
            Bp = np.sqrt(self['flux'][k]['Br'] ** 2 + self['flux'][k]['Bz'] ** 2)
            dl = np.sqrt(np.ediff1d(self['flux'][k]['R'], to_begin=0) ** 2 + np.ediff1d(self['flux'][k]['Z'], to_begin=0) ** 2)
            Bpl = np.sum(Bp * dl * 2 * np.pi)  # integral over flux surface
            Bp2_vol += Bpl * dpsi[k]  # integral over dpsi (making it <Bp**2> * V )
        circum = np.sum(dl)  # to calculate the length of the last closed flux surface
        li_from_definition = Bp2_vol / vol / constants.mu_0 / constants.mu_0 / ip / ip * circum * circum
        # li_3_TLUCE is the same as li_3_IMAS (by numbers)
        # ali_1_EFIT is the same as li_from_definition
        self['info']['internal_inductance'] = {
            "li_from_definition": li_from_definition,
            "li_(1)_TLUCE": li_from_definition / circum / circum * 2 * vol / r_0 * correction_factor,
            "li_(2)_TLUCE": li_from_definition / circum / circum * 2 * vol / r_axis,
            "li_(3)_TLUCE": li_from_definition / circum / circum * 2 * vol / r_0,
            "li_(1)_EFIT": circum * circum * Bp2_vol / (vol * constants.mu_0 * constants.mu_0 * ip * ip),
            "li_(3)_IMAS": 2 * Bp2_vol / r_0 / ip / ip / constants.mu_0 / constants.mu_0,
        }

        # EFIT current normalization
        self['info']['J_efit_norm'] = (
            (self['RCENTR'] * self['avg']['1/R']) * self['avg']['Jt'] / (self['CURRENT'] / self['geo']['cxArea'][-1])
        )

        # open separatrix
        if self.open_sep is not None:
            try:
                self['info']['open_separatrix'] = self.sol(levels=[1], open_flx={1: self.open_sep})[0][0]
            except Exception as _excp:
                printw('Error tracing open field-line separatrix: ' + repr(_excp))
                self['info']['open_separatrix'] = _excp
            else:
                ros = self['info']['open_separatrix']['R']
                istrk = np.array([0, -1] if ros[-1] > ros[0] else [-1, 0])  # Sort it so it goes inner, then outer strk pt
                self['info']['rvsin'], self['info']['rvsout'] = ros[istrk]
                self['info']['zvsin'], self['info']['zvsout'] = self['info']['open_separatrix']['Z'][istrk]

        # primary xpoint
        i = np.argmin(np.sqrt(self['flux'][self.nc - 1]['Br'] ** 2 + self['flux'][self.nc - 1]['Bz'] ** 2))
        self['info']['xpoint'] = np.array([self['flux'][self.nc - 1]['R'][i], self['flux'][self.nc - 1]['Z'][i]])

        # identify sol regions (works for single x-point >> do not do this for double-X-point or limited cases)
        if (
            'rvsin' in self['info']
            and 'zvsin' in self['info']
            and np.sign(self.open_sep[0, 1]) == np.sign(self.open_sep[-1, 1])
            and self.open_sep[0, 1] != self.open_sep[-1, 1]
            and self.open_sep[0, 0] != self.open_sep[-1, 0]
        ):
            rx, zx = self['info']['xpoint']

            # find minimum distance between legs of open separatrix used to estimate circle radius `a`
            k = int(len(self.open_sep) // 2)
            r0 = self.open_sep[:k, 0]
            z0 = self.open_sep[:k, 1]
            r1 = self.open_sep[k:, 0]
            z1 = self.open_sep[k:, 1]
            d0 = np.sqrt((r0 - rx) ** 2 + (z0 - zx) ** 2)
            i0 = np.argmin(d0)
            d1 = np.sqrt((r1 - rx) ** 2 + (z1 - zx) ** 2)
            i1 = np.argmin(d1) + k
            a = np.sqrt((self.open_sep[i0, 0] - self.open_sep[i1, 0]) ** 2 + (self.open_sep[i0, 1] - self.open_sep[i1, 1]) ** 2)
            a *= 3

            # circle
            t = np.linspace(0, 2 * np.pi, 101)[:-1]
            r = a * np.cos(t) + rx
            z = a * np.sin(t) + zx

            # intersect open separatrix with small circle around xpoint
            circle = line_intersect(np.array([self.open_sep[:, 0], self.open_sep[:, 1]]).T, np.array([r, z]).T)

            if len(circle) == 4:

                # always sort points so that they are in [inner_strike, outer_strike, outer_midplane, inner_midplane] order
                circle0 = circle - np.array([rx, zx])[np.newaxis, :]
                # clockwise for upper Xpoint
                if zx > 0 and np.sign(circle0[0, 0] * circle0[1, 1] - circle0[1, 0] * circle0[0, 1]) > 0:
                    circle = circle[::-1]
                # counter clockwise for lower Xpoint
                elif zx < 0 and np.sign(circle0[0, 0] * circle0[1, 1] - circle0[1, 0] * circle0[0, 1]) < 0:
                    circle = circle[::-1]
                # start numbering from inner strike wall
                index = np.argmin(np.sqrt((circle[:, 0] - self['info']['rvsin']) ** 2 + (circle[:, 1] - self['info']['zvsin']) ** 2))
                circle = np.vstack((circle, circle))[index : index + 4, :]
                for k, item in enumerate(['xpoint_inner_strike', 'xpoint_outer_strike', 'xpoint_outer_midplane', 'xpoint_inner_midplane']):
                    try:
                        self['info'][item] = circle[k]
                    except IndexError:
                        printe('Error parsing %s' % item)

                # regions are defined at midway points between the open separatrix points
                regions = circle + np.diff(np.vstack((circle, circle[0])), axis=0) / 2.0
                for k, item in enumerate(['xpoint_private_region', 'xpoint_outer_region', 'xpoint_core_region', 'xpoint_inner_region']):
                    try:
                        self['info'][item] = regions[k]
                    except IndexError:
                        printe('Error parsing %s' % item)

            # logic for secondary xpoint evaluation starts here
            # find where Bz=0 on the opposite side of the primary X-point: this is xpoint2_start
            Bz_sep = self['flux'][self.nc - 1]['Bz'].copy()
            mask = self['flux'][self.nc - 1]['Z'] * np.sign(self['info']['xpoint'][1]) > 0
            Bz_sep[mask] = np.nan
            index = np.nanargmin(abs(Bz_sep))
            xpoint2_start = [self['flux'][self.nc - 1]['R'][index], self['flux'][self.nc - 1]['Z'][index]]

            # trace Bz=0 contour and find the contour line that passes closest to xpoint2_start: this is the rz_divider line
            Bz0 = contourPaths(self.Rin, self.Zin, Bz, [0], remove_boundary_points=True, smooth_factor=1)[0]
            d = []
            for item in Bz0:
                d.append(np.min(np.sqrt((item.vertices[:, 0] - xpoint2_start[0]) ** 2 + (item.vertices[:, 1] - xpoint2_start[1]) ** 2)))
            rz_divider = Bz0[np.argmin(d)].vertices

            # evaluate Br along rz_divider line and consider only side opposite side of the primary X-point
            Br_divider = RectBivariateSplineNaN(self.Zin, self.Rin, Br).ev(rz_divider[:, 1], rz_divider[:, 0])
            mask = (rz_divider[:, 1] * np.sign(self['info']['xpoint'][1])) < -abs(self['info']['xpoint'][1]) / 10.0
            Br_divider = Br_divider[mask]
            rz_divider = rz_divider[mask, :]
            if abs(rz_divider[0, 1]) > abs(rz_divider[-1, 1]):
                rz_divider = rz_divider[::-1, :]
                Br_divider = Br_divider[::-1]

            # secondary xpoint where Br flips sign
            tmp = np.where(np.sign(Br_divider) != np.sign(Br_divider)[0])[0]
            if len(tmp):
                ix = tmp[0]
                self['info']['xpoint2'] = (rz_divider[ix - 1, :] + rz_divider[ix, :]) * 0.5
            else:
                self['info']['xpoint2'] = None

        # limiter
        if (
            hasattr(self, 'rlim')
            and self.rlim is not None
            and len(self.rlim) > 3
            and hasattr(self, 'zlim')
            and self.zlim is not None
            and len(self.zlim) > 3
        ):
            self['info']['rlim'] = self.rlim
            self['info']['zlim'] = self.zlim

        if not self.quiet:
            printi('  > Took {:}'.format(datetime.datetime.now() - t0))
        """
        print(f'{time.time() - startTime} to get through fluxSurfaces')







