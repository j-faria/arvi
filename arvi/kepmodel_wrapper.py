import io
from contextlib import redirect_stdout, contextmanager
from string import ascii_lowercase
from matplotlib import pyplot as plt
import numpy as np

from kepmodel.rv import RvModel
from spleaf.cov import merge_series
from spleaf.term import Error, InstrumentJitter

from .setup_logger import setup_logger
from .utils import timer, adjust_lightness


class model:
    logger = setup_logger()
    # periodogram settings
    Pmin, Pmax, nfreq = 1.5, 10_000, 100_000

    @contextmanager
    def ew(self):
        for name in self.model.keplerian:
            self.model.set_keplerian_param(name, param=['P', 'la0', 'K', 'e', 'w'])
        try:
            yield
        finally:
            for name in self.model.keplerian:
                self.model.set_keplerian_param(name, param=['P', 'la0', 'K', 'esinw', 'ecosw'])

    @property
    def nu0(self):
        return 2 * np.pi / self.Pmax
    
    @property
    def dnu(self):
        return (2 * np.pi / self.Pmin - self.nu0) / (self.nfreq - 1)

    def __init__(self, s):
        self.s = s
        self.instruments = s.instruments
        self.Pmax = 2 * np.ptp(s.time)
        ts = self.ts = self.s._mtime_sorter

        # t, y_ ye, series_index = merge_series(
        #     [_s.mtime for _s in s],
        #     [_s.mvrad for _s in s],
        #     [_s.msvrad for _s in s],
        # )

        inst_jit = self._get_jitters()

        self.model = RvModel(self.s.mtime[ts], self.s.mvrad[ts],
                             err=Error(self.s.msvrad[ts]), **inst_jit)
        self.np = 0
        self._add_means()

    def _add_means(self):
        for inst in self.s.instruments:
            # if inst not in self.s.instrument_array[self.s.mask]:
            #     continue
            mask = self.s.instrument_array[self.s.mask][self.ts] == inst
            self.model.add_lin(
                derivative=1.0 * mask,
                name=f"offset_{inst}",
                value=getattr(self.s, inst).mvrad.mean(),
            )
        self.model.fit_lin()
    
    def _get_jitters(self):
        inst_jit = {}
        for inst in self.s.instruments:
            inst_jit[f'jit_{inst}'] = InstrumentJitter(
                indices=self.s.instrument_array[self.s.mask] == inst,
                sig=self.s.svrad[self.s.mask].min()
            )
        return inst_jit

    def _set_jitters(self, value=0.0):
        for par in self.model.cov.param:
            if 'jit' in par:
                self.model.set_param(value, f'cov.{par}')
        # self.model.fit()
    
    def _equal_coralie_offsets(self):
        if self.s._check_instrument('CORALIE') is None:
            return
        mask = np.char.find(self.s.instrument_array, 'CORALIE') == 0
        mean = self.s.vrad[mask].mean()
        for inst in self.instruments:
            if 'CORALIE' in inst:
                self.model.set_param(mean, f'lin.offset_{inst}')


    def __repr__(self):
        with self.ew():
            with io.StringIO() as buf, redirect_stdout(buf):
                self.model.show_param()
                output = buf.getvalue()
        return output

    def to_table(self, **kwargs):
        from .utils import pretty_print_table
        lines = repr(self)
        lines = lines.replace(' [deg]', '_[deg]')
        lines = lines.encode().replace(b'\xc2\xb1', b'').decode()
        lines = lines.split('\n')
        lines = [line.split() for line in lines]
        lines = [[col.replace('_[deg]', ' [deg]') for col in line] for line in lines]
        pretty_print_table(lines[:-2], **kwargs)

    @property
    def fit_param(self):
        return self.model.fit_param

    def fit(self):
        # fit offsets
        self.model.fit_param = [f'lin.offset_{inst}' for inst in self.instruments]
        # fit jitters
        self.model.fit_param += [f'cov.{par}' for par in self.model.cov.param]
        # fit keplerian(s)
        self.model.fit_param += [
            f'kep.{k}.{p}' 
            for k, v in self.model.keplerian.items() 
            for p in v._param
        ]
        # if self.np == 0:
        #     self._set_jitters(0.1 * np.std(self.model.y - self.model.model()))
        try:
            self.model.fit()
        except Exception as e:
            print(e)
            

    def plot(self, **kwargs):
        fig, ax = self.s.plot(**kwargs)
        tt = self.s._tt()
        time_offset = 50000 if 'remove_50000' in kwargs else 0

        for i, inst in enumerate(self.s):
            inst_name = inst.instruments[0].replace('-', '_')
            val = self.model.get_param(f'lin.offset_{inst_name}')
            x = np.array([inst.mtime.min(), inst.mtime.max()]) - time_offset
            y = [val, val]
            ax.plot(x, y, ls='--', color=f'C{i}')
            mask = (tt > inst.mtime.min()) & (tt < inst.mtime.max())
            color = adjust_lightness(f'C{i}', 1.2)
            ax.plot(tt[mask] - time_offset,
                    val + self.model.keplerian_model(tt)[mask],
                    color=color)

        return fig, ax

    def plot_phasefolding(self, planets=None, ax=None):
        t = self.model.t
        res = self.model.residuals()
        sig = np.sqrt(self.model.cov.A)

        Msmooth = np.linspace(0, 360, 1000)

        if planets is None:
            planets = list(self.model.keplerian.keys())

        if ax is None:
            fig, axs = plt.subplots(
                1, len(planets), sharex=True, sharey=True, constrained_layout=True,
                squeeze=False
            )
        else:
            axs = np.atleast_1d(ax)
            fig = axs[0].figure

        for p, ax in zip(planets, axs.flat):
            self.model.set_keplerian_param(p, param=['P', 'la0', 'K', 'e', 'w'])
            kep = self.model.keplerian[p]
            P = self.model.get_param(f'kep.{p}.P')
            M0 = (180 / np.pi * (self.model.get_param(f'kep.{p}.la0') - self.model.get_param(f'kep.{p}.w')))
            M = (M0 + 360 / P * t) % 360
            reskep = res + kep.rv(t)
            tsmooth = (Msmooth - M0) * P / 360
            mod = kep.rv(tsmooth)

            ax.plot([0, 360], [0, 0], '--', c='gray', lw=1)
            ax.plot(Msmooth, mod, 'k-', lw=3, rasterized=True)
            for inst in self.instruments:
                sel = self.s.instrument_array[self.s.mask] == inst
                ax.errorbar(M[sel], reskep[sel], sig[sel], fmt='.',
                            rasterized=True, alpha=0.7)

            ax.set(ylabel='RV [m/s]', xlabel='Mean anomaly [deg]',
                   xticks=np.arange(0, 361, 90))
            ax.minorticks_on()
            self.model.set_keplerian_param(p, param=['P', 'la0', 'K', 'esinw', 'ecosw'])

        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5))
        return fig, axs

    def add_planet_from_period(self, period):
        self.model.add_keplerian_from_period(period, fit=True)
        self.model.fit()
        self.np += 1

    def _plot_periodogram(self, P=None, power=None, kmax=None, faplvl=None,
                          **kwargs):
        if P is None and power is None:
            with timer('periodogram'):
                nu, power = self.model.periodogram(self.nu0, self.dnu, self.nfreq)
                P = 2 * np.pi / nu

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
            fig = ax.figure
        else:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.semilogx(P, power, 'k', lw=1, rasterized=True)
        ax.set_ylim(0, 1.2 * power.max())
        ax.set(xlabel='Period [days]', ylabel='Normalized power')

        if kmax is None:
            kmax = np.argmax(power)
        ax.plot(P[kmax], power[kmax], 'or', ms=4)
        ax.text(P[kmax], power[kmax] * 1.1, f'{P[kmax]:.3f} d',
                ha='right', va='center', color='r')

        if faplvl is None:
            faplvl = self.model.fap(power[kmax], nu.max())
        ax.text(0.99, 0.95, f'FAP = {faplvl:.2g}', transform=ax.transAxes,
                ha='right', va='top')

        return fig, ax

    def add_keplerian_from_periodogram(self, fap_max=0.001, plot=False,
                                       fit_first=True):
        if fit_first and self.np == 0:
            self.fit()

        self._equal_coralie_offsets()

        with timer('periodogram'):
            nu, power = self.model.periodogram(self.nu0, self.dnu, self.nfreq)

        P = 2 * np.pi / nu
        # Compute FAP
        kmax = np.argmax(power)
        faplvl = self.model.fap(power[kmax], nu.max())
        self.logger.info('highest periodogram peak:')
        self.logger.info(f'P={P[kmax]:.4f} d, power={power[kmax]:.3f}, FAP={faplvl:.2e}')
        if plot:
            self._plot_periodogram(P, power, kmax, faplvl)

        if faplvl > fap_max:
            print('non-significant peak')
            self.fit()
            return False

        # add new planet
        letter = ascii_lowercase[1:][self.np]
        self.model.add_keplerian_from_period(P[kmax], name=letter,
                                             guess_kwargs={'emax': 0.8})
        # self.model.set_keplerian_param(letter, param=['P', 'la0', 'K', 'e', 'w'])
        self.model.set_keplerian_param(letter, param=['P', 'la0', 'K', 'esinw', 'ecosw'])
        self.np += 1
        self.fit()
        
        if plot:
            self.plot()

        return True
    
    @property
    def offsets(self):
        names = [f'lin.offset_{inst}' for inst in self.instruments]
        return {
            name.replace('lin.', ''): self.model.get_param(name)
            for name in names
        }

    @property
    def jitters(self):
        names = [f'cov.{par}' for par in self.model.cov.param]
        return {
            name.replace('cov.', '').replace('.sig', ''): self.model.get_param(name)
            for name in names
        }
    
    @property
    def keplerians(self):
        keps = {name: {} for name in self.model.keplerian.keys()}
        for name in keps:
            params = self.model.keplerian[name]._param
            pars = [f'kep.{name}.{p}' for p in params]
            keps[name] = {
                par.replace(f'kep.{name}.', ''): self.model.get_param(par)
                for par in pars
            }
        return keps