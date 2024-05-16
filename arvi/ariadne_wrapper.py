import os
import sys

try:
    from astroARIADNE.star import Star
    from astroARIADNE.fitter import Fitter
except ImportError:
    print('This module requires astroARIADNE. Install with `pip install astroARIADNE`')
    sys.exit(0)


def run_ariadne(self, fit=True, plot=True):
    if hasattr(self, 'gaia'):
        s = Star(self.star, self.gaia.ra, self.gaia.dec, g_id=self.gaia.dr3_id,
                 search_radius=1)
    else:
        s = Star(self.star, self.simbad.ra, self.simbad.dec, g_id=self.simbad.gaia_id,
                 search_radius=1)

    out_folder = f'{self.star}_ariadne'

    setup = dict(engine='dynesty', nlive=300, dlogz=1, 
                 bound='multi', sample='auto', threads=6, dynamic=False)
    setup = list(setup.values())

    models = [
            'phoenix',
            'btsettl',
            'btnextgen',
            'btcond',
            'kurucz',
            'ck04'
    ]

    f = Fitter()
    f.star = s
    f.setup = setup
    f.av_law = 'fitzpatrick'
    f.out_folder = out_folder
    f.bma = True
    f.models = models
    f.n_samples = 10_000

    f.prior_setup = {
            'teff': ('default'),
            'logg': ('default'),
            'z': ('default'),
            'dist': ('default'),
            'rad': ('default'),
            'Av': ('default')
    }

    f.initialize()
    if fit:
        f.fit_bma()

    if plot:
        from pkg_resources import resource_filename
        from astroARIADNE.plotter import SEDPlotter
        modelsdir = resource_filename('astroARIADNE', 'Datafiles/models')
        artist = SEDPlotter(os.path.join(out_folder, 'BMA.pkl'), out_folder, models_dir=modelsdir)

        artist.plot_SED_no_model()
        artist.plot_SED()
        artist.plot_bma_hist()
        artist.plot_bma_HR(10)
        artist.plot_corner()

    return s, f