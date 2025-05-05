from functools import partial
import numpy as np
from astropy.timeseries import LombScargle

from .setup_logger import setup_logger


sine_line = None
residual_gls = None
def sine_picker(event, self, fig, ax, ax1):
    from .timeseries import fit_sine
    global sine_line, residual_gls
    if sine_line is not None:
        sine_line[0].remove()
    if residual_gls is not None:
        residual_gls[0].remove()
    xdata, ydata = event.artist.get_data()
    ind = event.ind
    period = xdata[ind][0]
    p, sine = fit_sine(self.mtime, self.mvrad, self.msvrad, period=period)
    tt = np.linspace(self.mtime.min(), self.mtime.max(), 100)
    tt -= 50000
    sine_line = ax1.plot(tt, sine(tt), 'k')
    #
    f, p = LombScargle(self.mtime, self.mvrad - sine(self.mtime), self.msvrad).autopower()
    residual_gls = ax.semilogx(1/f, p, 'r')
    fig.canvas.draw_idle()


class REPORTS:
    def summary(self, add_ccf_mask=True, add_prog_id=False, **kwargs):
        from .utils import pretty_print_table
        if isinstance(self, list):
            selfs = self
        else:
            selfs = [self]
        
        rows = []
        for self in selfs:
            rows.append([self.star] + [''] * len(self.instruments) + [''])
            rows.append([''] + self.instruments + ['full'])
            rows.append(['N'] + list(self.NN.values()) + [self.N])
            rows.append(['RV span'] + [np.ptp(s.mvrad).round(3) for s in self] + [np.ptp(self.mvrad).round(3)])
            rows.append(['RV std'] + [s.mvrad.std().round(3) for s in self] + [self.mvrad.std().round(3)])
            rows.append(['eRV mean'] + [s.msvrad.mean().round(3) for s in self] + [self.msvrad.mean().round(3)])

            if add_ccf_mask:
                if hasattr(self, 'ccf_mask'):
                    row = ['CCF mask']
                    for inst in self.instruments:
                        row.append(', '.join(np.unique(getattr(self, inst).ccf_mask)))
                    row.append('')
                    rows.append(row)

            if add_prog_id:
                row = ['prog ID']
                for inst in self.instruments:
                    p = ', '.join(np.unique(getattr(self, inst).prog_id))
                    row.append(p)
                rows.append(row)

        return pretty_print_table(rows, **kwargs)


    def summary_one_instrument(self):
        from .utils import pretty_print_table
        if isinstance(self, list):
            selfs = self
        else:
            selfs = [self]

        rows = []
        rows.append(['', 'N', 'Δt', 'RV std', 'median σRV'])
        for self in selfs:
            rows.append([
                self.star, 
                self.N, 
                int(np.ptp(self.mtime).round(0)),
                np.std(self.mvrad).round(1),
                np.median(self.msvrad).round(1),
            ])

        pretty_print_table(rows)


    def summary_stellar(self, show_ra_dec=True, show_pm=True, **kwargs):
        from .utils import pretty_print_table, get_ra_sexagesimal, get_dec_sexagesimal
        if isinstance(self, list):
            selfs = self
        else:
            selfs = [self]
        


        rows = []
        rows.append(['star'] + [self.star for self in selfs])
        if show_ra_dec:
            rows.append(['RA'] + [get_ra_sexagesimal(self.simbad.ra) for self in selfs])
            rows.append(['DEC'] + [get_dec_sexagesimal(self.simbad.dec) for self in selfs])
        
        if show_pm:
            rows.append(['pm ra'] + [(self.gaia or self.simbad).pmra for self in selfs])
            rows.append(['pm dec'] + [(self.gaia or self.simbad).pmdec for self in selfs])

        rows.append(['π (mas)'] + [(self.gaia or self.simbad).plx for self in selfs])
        # 
        rows.append(['B (mag)'] + [self.simbad._B for self in selfs])
        rows.append(['V (mag)'] + [self.simbad._V for self in selfs])
        rows.append(['B-V'] + [self.simbad._B - self.simbad._V for self in selfs])
        rows.append(["log R'HK"] + [np.nanmean(self.rhk).round(2) for self in selfs])
            # rows.append([''] + self.instruments + ['full'])
            # rows.append(['N'] + list(self.NN.values()) + [self.N])
            # rows.append(['RV span'] + [np.ptp(s.mvrad).round(3) for s in self] + [np.ptp(self.mvrad).round(3)])
            # rows.append(['RV std'] + [s.mvrad.std().round(3) for s in self] + [self.mvrad.std().round(3)])
            # rows.append(['eRV mean'] + [s.msvrad.mean().round(3) for s in self] + [self.msvrad.mean().round(3)])

        return pretty_print_table(rows, **kwargs)




    def report(self, save=None):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.backends.backend_pdf import PdfPages
        logger = setup_logger()

        # size = A4
        size = 8.27, 11.69
        fig = plt.figure(figsize=size, constrained_layout=True)
        gs = gridspec.GridSpec(5, 3, figure=fig, height_ratios=[2, 2, 1, 1, 0.1])

        # first row, all columns
        ax1 = plt.subplot(gs[0, :])

        title = f'{self.star}'
        ax1.set_title(title, loc='left', fontsize=14)
        # ax1.set_title(r"\href{http://www.google.com}{link}", color='blue',
        #               loc='center')

        if self._did_adjust_means:
            title = '(instrument means subtracted) '
        else:
            title = ''
        title += f'V={self.simbad.V}, {self.simbad.sp_type}'
        ax1.set_title(title, loc='right', fontsize=12)

        self.plot(ax=ax1, N_in_label=True, tooltips=False, remove_50000=True)


        ax1.legend().remove()
        legend_ax = plt.subplot(gs[1, -1])
        legend_ax.axis('off')
        leg = plt.legend(*ax1.get_legend_handles_labels(),
                        prop={'family': 'monospace'})
        legend_ax.add_artist(leg)
        
        second_legend = f'rms  : {self.rms:.2f} {self.units}'
        second_legend += f'\nerror: {self.error:.2f} {self.units}'

        second_legend += f'\n\nCCF masks: {np.unique(self.ccf_mask)}'

        legend_ax.legend([],
                        title=second_legend,
                        loc='lower right', frameon=False,
                        prop={'family': 'monospace'})

        ax2 = plt.subplot(gs[1, :-1])
        self.gls(ax=ax2, picker=True)

        ax3 = plt.subplot(gs[2, :-1])
        self.plot_fwhm(ax=ax3, tooltips=False, remove_50000=True)
        ax3.legend().remove()
        ax3p = plt.subplot(gs[2, -1])
        self.gls_fwhm(ax=ax3p, picker=False)

        ax4 = plt.subplot(gs[3, :-1])
        self.plot_rhk(ax=ax4, tooltips=False, remove_50000=True)
        ax4.legend().remove()
        ax4p = plt.subplot(gs[3, -1])
        self.gls_rhk(ax=ax4p, picker=False)


        if save is None:
            fig.canvas.mpl_connect(
                'pick_event',
                partial(sine_picker, self=self, fig=fig, ax=ax2, ax1=ax1))

        if save is not None:
            if save is True:
                save = f'report_{"".join(self.star.split())}.pdf'

            if save.endswith('.png'):
                fig.savefig(save)
            else:
                with PdfPages(save) as pdf:
                    #pdf.attach_note('hello', positionRect=[5, 15, 20, 30])

                    if self.verbose:
                        logger.info(f'saving to {save}')
                    pdf.savefig(fig)
                # os.system(f'evince {save} &')

        return fig