import os
from functools import partial, partialmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import mplcursors

from astropy.timeseries import LombScargle

from .setup_logger import logger
from . import config


def plot(self, ax=None, show_masked=False, instrument=None, time_offset=0,
         remove_50000=False, tooltips=True, label=None, N_in_label=False,
         versus_n=False, show_histogram=False, **kwargs):
    """ Plot the RVs

    Args:
        ax (Axes, optional):
            Axis to plot to. Defaults to None.
        show_masked (bool, optional):
            Show masked points. Defaults to False.
        instrument (str, optional):
            Which instrument to plot. Defaults to None, or plot all instruments.
        time_offset (int, optional):
            Value to subtract from time. Defaults to 0.
        remove_50000 (bool, optional):
            Whether to subtract 50000 from time. Defaults to False.
        tooltips (bool, optional):
            Show information upon clicking a point. Defaults to True.
        N_in_label (bool, optional):
            Show number of observations in legend. Defaults to False.
        versus_n (bool, optional):
            Plot RVs vs observation index instead of time. Defaults to False.
        show_histogram (bool, optional)
            Whether to show a panel with the RV histograms (per intrument).
            Defaults to False.

    Returns:
        Figure: the figure
        Axes: the axis
    """
    if self.N == 0:
        if self.verbose:
            logger.error('no data to plot')
        return

    if ax is None:
        if show_histogram:
            fig, (ax, axh) = plt.subplots(1, 2, constrained_layout=True, gridspec_kw={'width_ratios': [3, 1]})
        else:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        if show_histogram:
            ax, axh = ax
        fig = ax.figure

    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('ls', '')
    kwargs.setdefault('capsize', 0)
    kwargs.setdefault('ms', 4)

    if remove_50000:
        time_offset = 50000

    instruments = self._check_instrument(instrument)

    cursors = {}
    for inst in instruments:
        s = self if self._child else getattr(self, inst)
        if s.mask.sum() == 0:
            continue

        if label is None:
            _label = f'{inst:10s} ({s.N})' if N_in_label else inst
            if not self.only_latest_pipeline:
                i, p = _label.split('_', 1)
                p = p.replace('_', '.')
                _label = f'{i}-{p}'
        else:
            _label = label

        if versus_n:
            container = ax.errorbar(np.arange(1, s.mtime.size + 1),
                                    s.mvrad, s.msvrad, label=_label, picker=True, **kwargs)
        else:
            container = ax.errorbar(s.mtime - time_offset,
                                    s.mvrad, s.msvrad, label=_label, picker=True, **kwargs)

        if show_histogram:
            kw = dict(histtype='step', bins='doane', orientation='horizontal')
            hlabel = f'{s.mvrad.std():.2f} {self.units}'
            axh.hist(s.mvrad, **kw, label=hlabel)

        if tooltips:
            cursors[inst] = crsr = mplcursors.cursor(container, multiple=False)

            @crsr.connect("add")
            def _(sel):
                inst = sel.artist.get_label()
                _s = getattr(self, inst)
                vrad, svrad = _s.vrad[sel.index], _s.svrad[sel.index]
                sel.annotation.get_bbox_patch().set(fc="white")
                text = f'{inst}\n'
                text += f'BJD: {sel.target[0]:9.5f}\n'
                text += f'RV: {vrad:.3f} ± {svrad:.3f}'
                if fig.canvas.manager.toolmanager.get_tool('infotool').toggled:
                    text += '\n\n'
                    text += f'date: {_s.date_night[sel.index]}\n'
                    text += f'mask: {_s.ccf_mask[sel.index]}'
                sel.annotation.set_text(text)

        if show_masked:
            if versus_n:
                pass
            else:
                ax.errorbar(s.time[~s.mask] - time_offset, s.vrad[~s.mask], s.svrad[~s.mask],
                            **kwargs, color=container[0].get_color())

    if show_masked:
        if versus_n:
            pass
            # ax.errorbar(self.time[~self.mask] - time_offset, self.vrad[~self.mask], self.svrad[~self.mask],
            #             label='masked', fmt='x', ms=10, color='k', zorder=-2)
        else:
            ax.errorbar(self.time[~self.mask] - time_offset, self.vrad[~self.mask], self.svrad[~self.mask],
                        label='masked', fmt='x', ms=10, color='k', zorder=-2)

    ax.legend()
    if show_histogram:
        axh.legend()

    ax.minorticks_on()

    ax.set_ylabel(f'RV [{self.units}]')
    if versus_n:
        ax.set_xlabel('observation')
    else:
        if remove_50000:
            ax.set_xlabel('BJD - 2450000 [days]')
        else:
            ax.set_xlabel('BJD - 2400000 [days]')

    from matplotlib.backend_tools import ToolBase, ToolToggleBase
    tm = fig.canvas.manager.toolmanager

    class InfoTool(ToolToggleBase):
        description = "Show extra information about each observation"
        image = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'info.svg'))
        # def enable(self, *args, **kwargs):
        #     self.figure.add_axes([1, 0, 0.3, 1])
        #     self.figure.canvas.draw_idle()

    from PIL import UnidentifiedImageError
    try:
        tm.add_tool("infotool", InfoTool)
        fig.canvas.manager.toolbar.add_tool(tm.get_tool("infotool"), "toolgroup")
        raise UnidentifiedImageError
    except AttributeError:
        pass
    except UnidentifiedImageError:
        pass


    if config.return_self:
        return self
    
    if show_histogram:
        return fig, (ax, axh)
    
    return fig, ax


def plot_quantity(self, quantity, ax=None, show_masked=False, instrument=None,
                  time_offset=0, remove_50000=False, tooltips=False,
                  N_in_label=False, **kwargs):
    if self.N == 0:
        if self.verbose:
            logger.error('no data to plot')
        return

    if not hasattr(self, quantity):
        logger.error(f"cannot find '{quantity}' attribute")
        return

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        fig = ax.figure

    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('ls', '')
    kwargs.setdefault('capsize', 0)
    kwargs.setdefault('ms', 4)

    if remove_50000:
        time_offset = 50000

    instruments = self._check_instrument(instrument)

    for inst in instruments:
        s = self if self._child else getattr(self, inst)
        label = f'{inst:10s} ({s.N})' if N_in_label else inst
        if not self.only_latest_pipeline:
            i, p = label.split('_', 1)
            p = p.replace('_', '.')
            label = f'{i}-{p}'

        y = getattr(s, quantity)
        ye = getattr(s, quantity + '_err')

        if np.isnan(y).all() or np.isnan(ye).all():
            lines, *_ = ax.errorbar([], [], [],
                                    label=label, picker=True, **kwargs)
            continue

        lines, *_ = ax.errorbar(s.mtime - time_offset, y[s.mask], ye[s.mask],
                                label=label, picker=True, **kwargs)

        if show_masked:
            ax.errorbar(s.time[~s.mask] - time_offset, y[~s.mask], ye[~s.mask],
                        **kwargs, color=lines.get_color())

    if show_masked:
        ax.errorbar(self.time[~self.mask] - time_offset,
                    getattr(self, quantity)[~self.mask],
                    getattr(self, quantity + '_err')[~self.mask],
                    label='masked', fmt='x', ms=10, color='k', zorder=-2)

    ax.legend()
    ax.minorticks_on()

    if quantity == 'fwhm':
        ax.set_ylabel(f'FWHM [{self.units}]')
    elif quantity == 'bispan':
        ax.set_ylabel(f'BIS [{self.units}]')
    elif quantity == 'rhk':
        ax.set_ylabel(r"$\log$ R'$_{HK}$")

    if remove_50000:
        ax.set_xlabel('BJD - 2450000 [days]')
    else:
        ax.set_xlabel('BJD - 2400000 [days]')

    if config.return_self:
        return self
    else:
        return fig, ax


plot_fwhm = partialmethod(plot_quantity, quantity='fwhm')
plot_bis = partialmethod(plot_quantity, quantity='bispan')
plot_rhk = partialmethod(plot_quantity, quantity='rhk')


def gls(self, ax=None, label=None, fap=True, picker=True, instrument=None, **kwargs):
    if self.N == 0:
        if self.verbose:
            logger.error('no data to compute gls')
        return

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        fig = ax.figure

    if instrument is not None:
        instrument = self._check_instrument(instrument)
        if instrument is not None:
            instrument_mask = np.isin(self.instrument_array, instrument)
            t = self.time[instrument_mask & self.mask]
            y = self.vrad[instrument_mask & self.mask]
            e = self.svrad[instrument_mask & self.mask]
            if self.verbose:
                logger.info(f'calculating periodogram for instrument {instrument}')
    else:
        t = self.time[self.mask]
        y = self.vrad[self.mask]
        e = self.svrad[self.mask]

    self._gls = gls = LombScargle(t, y, e)
    maximum_frequency = kwargs.pop('maximum_frequency', 1.0)
    minimum_frequency = kwargs.pop('minimum_frequency', None)
    freq, power = gls.autopower(maximum_frequency=maximum_frequency,
                                minimum_frequency=minimum_frequency,
                                samples_per_peak=10)
    ax.semilogx(1/freq, power, picker=picker, label=label, **kwargs)

    if fap:
        ax.axhline(gls.false_alarm_level(0.01),
                   color='k', alpha=0.2, zorder=-1)

    ax.set(xlabel='Period [days]', ylabel='Normalized power', ylim=(0, None))
    ax.minorticks_on()

    if label is not None:
        ax.legend()


    if config.return_self:
        return self
    else:
        return fig, ax


def gls_quantity(self, quantity, ax=None, fap=True, picker=True):
    if not hasattr(self, quantity):
        logger.error(f"cannot find '{quantity}' attribute")
        return

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        fig = ax.figure

    t = self.mtime
    y = getattr(self, quantity)[self.mask]
    ye = getattr(self, quantity + '_err')[self.mask]

    if np.isnan(y).any():
        if self.verbose:
            logger.warning(f'{quantity} contains NaNs, ignoring them')
        m = np.isnan(y)
        t = t[~m]
        y = y[~m]
        ye = ye[~m]

    gls = LombScargle(t, y, ye)
    freq, power = gls.autopower(maximum_frequency=1.0)
    ax.semilogx(1/freq, power, picker=picker)

    if fap:
        ax.axhline(gls.false_alarm_level(0.01),
                   color='k', alpha=0.2, zorder=-1)

    ax.set(xlabel='Period [days]', ylabel='Normalized power', ylim=(0, None))
    ax.minorticks_on()

    if config.return_self:
        return self
    else:
        return fig, ax


gls_fwhm = partialmethod(gls_quantity, quantity='fwhm')
gls_bis = partialmethod(gls_quantity, quantity='bispan')
gls_rhk = partialmethod(gls_quantity, quantity='rhk')
