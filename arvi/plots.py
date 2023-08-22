from functools import partial, partialmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from astropy.timeseries import LombScargle

from .setup_logger import logger


def plot(self,
         ax=None,
         show_masked=False,
         time_offset=0,
         remove_50000=False,
         tooltips=True,
         N_in_label=False,
         **kwargs):
    """ Plot the RVs

    Args:
        ax (Axes, optional): Axis to plot to. Defaults to None.
        show_masked (bool, optional): Show masked points. Defaults to False.
        time_offset (int, optional): Value to subtract from time. Defaults to 0.
        remove_50000 (bool, optional): Whether to subtract 50000 from time. Defaults to False.
        tooltips (bool, optional): TBD. Defaults to True.
        N_in_label (bool, optional): Show number of observations in legend. Defaults to False.

    Returns:
        Figure: the figure
        Axes: the axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        fig = ax.figure

    kwargs.setdefault('fmt', 'o')
    kwargs.setdefault('capsize', 0)
    kwargs.setdefault('ms', 4)

    if remove_50000:
        time_offset = 50000

    all_lines = []
    for inst in self.instruments:
        s = self if self._child else getattr(self, inst)
        label = f'{inst:10s} ({s.N})' if N_in_label else inst
        lines, *_ = ax.errorbar(s.mtime - time_offset,
                                s.mvrad,
                                s.msvrad,
                                label=label,
                                picker=True,
                                **kwargs)
        all_lines.append(lines)
    if show_masked:
        ax.errorbar(self.time[~self.mask] - time_offset,
                    self.vrad[~self.mask],
                    self.svrad[~self.mask],
                    label='masked', fmt='x', color='k')

    ax.legend()

    ax.set_ylabel(f'RV [{self.units}]')
    if remove_50000:
        ax.set_xlabel('BJD - 2450000 [days]')
    else:
        ax.set_xlabel('BJD - 2400000 [days]')

    if tooltips:
        inds = []
        def onpick(event):
            if isinstance(event.artist, LineCollection):
                return
            xdata, ydata = event.artist.get_data()
            ind = event.ind
            if ind in inds:
                inds.remove(ind)
            else:
                inds.append(ind)

            try:
                reds.remove()
            except UnboundLocalError:
                pass

            if len(inds) > 0:
                reds = ax.plot(xdata[np.array(inds)], ydata[np.array(inds)],
                                'ro', ms=10, zorder=-1)
            fig.canvas.draw()
        fig.canvas.mpl_connect('pick_event', onpick)

    return fig, ax


def plot_quantity(self,
                  quantity,
                  ax=None,
                  time_offset=0,
                  remove_50000=False,
                  tooltips=True,
                  N_in_label=False,
                  **kwargs):

    if not hasattr(self, quantity):
        logger.error(f"cannot find '{quantity}' attribute")
        return

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        fig = ax.figure

    kwargs.setdefault('fmt', 'o')
    kwargs.setdefault('capsize', 0)
    kwargs.setdefault('ms', 4)

    if remove_50000:
        time_offset = 50000

    all_lines = []
    for inst in self.instruments:
        s = self if self._child else getattr(self, inst)
        label = f'{inst:10s} ({s.N})' if N_in_label else inst

        y = getattr(s, quantity)[s.mask]
        ye = getattr(s, quantity + '_err')[s.mask]

        if np.isnan(y).all() or np.isnan(ye).all():
            lines, *_ = ax.errorbar([], [], [],
                                    label=label, picker=True, **kwargs)
            continue

        lines, *_ = ax.errorbar(s.mtime - time_offset, y, ye,
                                label=label, picker=True, **kwargs)
        all_lines.append(lines)

    ax.legend()

    if quantity == 'fwhm':
        ax.set_ylabel(f'FWHM [{self.units}]')
    elif quantity == 'bispan':
        ax.set_ylabel(f'BIS [{self.units}]')
    elif quantity == 'rhk':
        ax.set_ylabel("$\log$ R'$_{HK}")

    if remove_50000:
        ax.set_xlabel('BJD - 2450000 [days]')
    else:
        ax.set_xlabel('BJD - 2400000 [days]')

    if tooltips:
        inds = []
        def onpick(event):
            if isinstance(event.artist, LineCollection):
                return
            xdata, ydata = event.artist.get_data()
            ind = event.ind
            if ind in inds:
                inds.remove(ind)
            else:
                inds.append(ind)

            try:
                reds.remove()
            except UnboundLocalError:
                pass

            if len(inds) > 0:
                reds = ax.plot(xdata[np.array(inds)], ydata[np.array(inds)],
                                'ro', ms=10, zorder=-1)
            fig.canvas.draw()
        fig.canvas.mpl_connect('pick_event', onpick)

    return fig, ax


plot_fwhm = partialmethod(plot_quantity, quantity='fwhm')
plot_bis = partialmethod(plot_quantity, quantity='bispan')


def gls(self, ax=None, fap=True, picker=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        fig = ax.figure

    gls = LombScargle(self.mtime, self.mvrad, self.msvrad)
    freq, power = gls.autopower(maximum_frequency=1.0)
    ax.semilogx(1/freq, power, picker=picker)
    if fap:
        ax.axhline(gls.false_alarm_level(0.01),
                   color='k',
                   alpha=0.2,
                   zorder=-1)
    ax.set(xlabel='Period [days]', ylabel='Normalized power')
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
                   color='k',
                   alpha=0.2,
                   zorder=-1)
    ax.set(xlabel='Period [days]', ylabel='Normalized power')
    return fig, ax


gls_fwhm = partialmethod(gls_quantity, quantity='fwhm')
gls_bis = partialmethod(gls_quantity, quantity='bispan')