from omfit_classes.omfit_eqdsk import OMFITgeqdsk
from fluxSurfacesMinimal import fluxSurfaces_minimal
import os 
import numpy as np
import matplotlib.pyplot as plt
from omfit_classes.sortedDict import SortedDict
from omfit_classes.omfit_ascii import OMFITascii
from scipy import interpolate
import time

class OMFITgeqdsk_fast(OMFITgeqdsk):

    def __init__(self, filename, **kw):
        OMFITascii.__init__(self, filename, **kw)
        SortedDict.__init__(self, caseInsensitive=True)
        self._cocos = 1
        self._AuxNamelistString = None
        self.dynaLoad = True


    def addFluxSurfaces(self, **kw):
        r"""
        Adds ['fluxSurface'] to the current object

        :param \**kw: keyword dictionary passed to fluxSurfaces class

        :return: fluxSurfaces object based on the current gEQDSK file
        """
        if self['CURRENT'] == 0.0:
            printw('Skipped tracing of fluxSurfaces for vacuum equilibrium')
            return

        options = {}
        options.update(kw)
        options['quiet'] = kw.pop('quiet', self['NW'] <= 129)
        options['levels'] = kw.pop('levels', True)
        options['resolution'] = kw.pop('resolution', 0)
        options['calculateAvgGeo'] = kw.pop('calculateAvgGeo', True)

        # N.B., the middle option accounts for the new version of CHEASE
        #       where self['CASE'][1] = 'OM CHEAS'
        if (
            self['CASE'] is not None
            and self['CASE'][0] is not None
            and self['CASE'][1] is not None
            and ('CHEASE' in self['CASE'][0] or 'CHEAS' in self['CASE'][1] or 'TRXPL' in self['CASE'][0])
        ):
            options['forceFindSeparatrix'] = kw.pop('forceFindSeparatrix', False)
        else:
            options['forceFindSeparatrix'] = kw.pop('forceFindSeparatrix', True)

        try:
            self['fluxSurfaces'] = fluxSurfaces_minimal(gEQDSK=self, **options)
        except Exception as _excp:
            warnings.warn('Error tracing flux surfaces: ' + repr(_excp))
            self['fluxSurfaces'] = OMFITerror('Error tracing flux surfaces: ' + repr(_excp))

        return self['fluxSurfaces']


