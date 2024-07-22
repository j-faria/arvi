import os
import numpy as np
import matplotlib.pyplot as plt

from arvi.headers import get_headers
from barycorrpy import get_BC_vel
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy import constants as const
from astropy.timeseries import LombScargle
from tqdm import tqdm

from .setup_logger import logger
from . import config


def correct_rvs(self, simple=False, H=None, save_files=False, plot=True):
    """
    """
    import pickle

    if hasattr(self, '_did_correct_berv') and self._did_correct_berv:
        logger.info('Already corrected for the BERV! Not doing anything.')
        return

    path = os.path.dirname(__file__)
    path = os.path.join(path, 'data')
    pkl = os.path.join(path, 'berv_espresso_sine.pkl')
    berv_espresso = pickle.load(open(pkl, 'rb'))
    
    if simple:
        logger.info('Correcting RVs with a previously-fitted sinusoid function')
        _f = berv_espresso['func'].replace('lambda t: ', '')
        logger.info(f': {_f}')
        f = eval(berv_espresso['func'])
        if plot:
            _, ax = self.plot()
            ax.plot(self._tt, f(self._tt) + self.vrad.mean(), 'k')
            _, axgls = self.gls(label='before')
        
        self.vrad = self.vrad + f(self.time)

        if plot:
            self.gls(ax=axgls, label='after')

        return f(self.time)

    else:
        logger.info('Correcting RVs with actual difference between BERVs')
        logger.info('(basically, use BERV_barycorrpy for BERV correction)')

        old_vrad = self.vrad.copy()

        _, berv = BERV(self, H=H, use_gaia_meassurements=True, plx=self.gaia.plx, 
                       plot=False, ignore_mask=True)

        if plot:
            fig, axs = plt.subplots(2, 1, constrained_layout=True, height_ratios=(3, 1), sharex=True)
            _, ax = self.plot(ax=axs[0])
            _, axgls = self.gls(label='before')

        # undo secular acceleration, if it was done
        _did_secular_acceleration = self._did_secular_acceleration
        self._undo_secular_acceleration()

        # transform RVs: RV --> RV - BERVpipe + BERVbarycorrpy

        diff = berv[self.star]['berv_barycorrpy'] - berv[self.star]['berv_pipeline']

        if save_files:
            i_inst = np.hstack([np.arange(n) for n in self.NN.values()])
            with open(f'{self.star}_berv_correction.rdb', 'w') as rdb:
                rdb.write('# time\n')
                rdb.write('# vrad\n')
                rdb.write('# svrad\n')
                rdb.write('# berv - BERV value from header\n')
                rdb.write('# berv_pipe - BERV from header corrected for 1.55e-8 factor\n')
                rdb.write('# berv_barycorrpy - BERV value from barycorrpy\n')
                rdb.write('# diff - difference between berv_barycorrpy and berv_pipe\n')
                rdb.write('# vrad_berv_corrected = vrad + diff\n')
                rdb.write('# instrument\n')
                rdb.write('# i - index\n')
                rdb.write('# i_inst - index within the instrument\n')
                rdb.write('#\n')
                rdb.write('# --> TO CORRECT vrad, we ** add the diff column **\n')
                rdb.write('# --> the result of this operation is in column vrad_berv_corrected\n')
                rdb.write('# --> vrad_berv_corrected is already corrected for the secular acceleration, vrad is not\n')
                rdb.write('#\n')
                # 
                cols = [
                    'time', 'vrad', 'svrad', 
                    'berv', 'berv_pipe', 'berv_barycorrpy', 'diff', 'vrad_berv_corrected',
                    'instrument', 'i', 'i_inst'
                ]
                rdb.write('# ' + '\t'.join(cols) + '\n')
                for i, t in enumerate(self.time):
                    rdb.write(f'{t:11.5f}\t')
                    # if _did_secular_acceleration:
                    #     rdb.write(f'{old_vrad[i]:13.5f}\t')
                    # else:
                    rdb.write(f'{self.vrad[i]:13.7f}\t')
                    rdb.write(f'{self.svrad[i]:13.7f}\t')
                    rdb.write(f'{self.berv[i]:15.7f}\t')
                    rdb.write(f'{berv[self.star]["berv_pipeline"][i]/1e3:15.7f}\t')
                    rdb.write(f'{berv[self.star]["berv_barycorrpy"][i]/1e3:15.7f}\t')
                    rdb.write(f'{diff[i]:15.7f}\t')
                    rdb.write(f'{self.vrad[i] + diff[i]:13.7f}\t')
                    rdb.write(f'{self.instrument_array[i]}\t')
                    rdb.write(f'{i}\t')
                    rdb.write(f'{i_inst[i]}\t')
                    rdb.write('\n')

        self.add_to_vrad(diff)
        self._did_correct_berv = True
        self._did_secular_acceleration = True # "automatically", by using BERV_barycorrpy
        self._did_secular_acceleration_simbad = False
        self._did_secular_acceleration_epoch = Time('J2016').jd - 24e5

        # the secular acceleration hadn't been done, but it was introduced by
        # BERV_barycorrpy, so we need to undo it
        if not _did_secular_acceleration:
            self._undo_secular_acceleration()

        if plot:
            self.plot(ax=axs[0], marker='+', ms=5)
            axs[1].plot(self.time, old_vrad - self.vrad, '.k', label='old RV - new RV')
            ma = np.abs(axs[1].get_ylim()).max()
            axs[1].set(ylim=(-ma, ma), xlabel=axs[0].get_xlabel(), ylabel='RV difference [m/s]')
            self.gls(ax=axgls, label='after')

        return diff

def get_A_and_V_from_lesta(self, username=config.username):
    try:
        import paramiko
    except ImportError:
        raise ImportError("paramiko is not installed. Please install it with 'pip install paramiko'")

    logs = []
    for f in self.raw_file:
        f = f.replace('espresso/', '/projects/astro/ESPRESSODRS/')
        f = f.replace('nirps/', '/projects/astro/NIRPSDRS/')
        f = f.replace('.fits', '_SCIENCE_FP.log')
        f = f.replace('reduced', 'log')
        f = f.replace('r.ESPRE', 'ESPRESSO')
        logs.append(f)

    A, V = [], []

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect("lesta02.astro.unige.ch", username=username, timeout=5)
    except Exception as e:
        if 'getaddrinfo failed' in str(e):
            jump = paramiko.SSHClient()
            jump.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            jump.connect('login01.astro.unige.ch', username=username, timeout=5)
            jump_transport = jump.get_transport()
            jump_channel = jump_transport.open_channel('direct-tcpip', ('10.194.64.162', 22), ('129.194.64.20', 22))
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect('lesta02.astro.unige.ch', username=username, sock=jump_channel)
        else:
            raise e

    with ssh.open_sftp() as sftp:
        pbar = tqdm(logs, total=len(logs), unit='file', desc='Reading logs')
        for f in pbar:
            with sftp.open(f) as fp:
                pattern1 = 'Sun-Earth'
                pattern2 = "Barycentric Observer's Velocity"
                for line in fp:
                    if pattern1 in line:
                        value = line.strip().split(':')[-1].replace('\x1b[32m', '').replace('\x1b[0m', '').replace(' ', '')
                        A.append(float(value))
                    if pattern2 in line:
                        value = line.strip().split(':')[-1].replace('\x1b[32m', '').replace('\x1b[0m', '').replace(' ', '')
                        V.append(float(value))

    ssh.close()

    return np.array(A), np.array(V)


def BERV(self, H=None, use_gaia_meassurements=False, plx=None,
         A=None, V=None, plot=True, ignore_mask=False, verbose=False, dpi=None):
    """ Calculate Barycentric Radial Velocity with barycorr and compare with pipeline

    Args:
        H (list, optional):
            List of (CCF/S1D/etc) headers for the target. If None, try to
            download the CCF files to get the headers.
        use_gaia_meassurements (bool, optional):
            Use Gaia coordinates and proper motions instead of those in the headers.
        plx (float, optional):
            Value of stellar parallax [mas] to use in barycorr.
        A (array, optional):
            Earth-Sun distance [AU] for each BJD (found in the pipeline logs).
        V (array, optional):
            Earth's orbital velocity [km/s] for each BJD (found in the pipeline logs).
        plot (bool, optional):
            Plot the results.
    """
    if H is None:
        H = get_headers(self, check_lesta=False, check_exo2=False, instrument='ESPRE')

    if len(H) != self.N:
        raise ValueError(f'Expected {self.N} headers (in `H`), got {len(H)}')

    if 'HARPS' in H[0]['INSTRUME'] or 'NIRPS' in H[0]['INSTRUME']:
        obsname = 'lasilla'
    elif 'ESPRESSO' in H[0]['INSTRUME']:
        obsname = 'paranal'
    else:
        raise ValueError('unknown instrument')

    bjd = np.array([h['HIERARCH ESO QC BJD'] for h in H])
    bjd -= 24e5
    berv_pipeline = np.array([h['HIERARCH ESO QC BERV'] for h in H])

    # in the pipeline, the BERV is used to shift wavelenghts with this formula
    # berv_factor = (1 + 1.55e-8) * (1 + BERV/c)
    # The 1.55e-8 factor is an average of some relativistic effects, which are
    # probably already included in the BERV calculated from barycorrpy.
    # Therefore, we compute an "effective" BERV from the pipeline doing
    # (1 + 1.55e-8) * (1 + BERV/c) = 1 + effBERV/c 
    # => effBERV = ((1 + 1.55e-8) * (1 + BERV/c) - 1) * c

    if A is None and V is None:
        if verbose:
            logger.info("Using mean value for Earth-Sun distance and Earth's orbital velocity")

    if A is None:
        Φobs = const.G * const.M_sun / const.au + const.G * const.M_earth / const.R_earth
    else:
        A = np.atleast_1d(A) * u.km
        Φobs = const.G * const.M_sun / A + const.G * const.M_earth / const.R_earth

    if V is None:
        V = 29785 *u.m / u.second
    else:
        V = np.atleast_1d(V) * u.km / u.second

    f = 1 / (1 - Φobs / const.c**2 - V**2 / (2*const.c**2))
    c = const.c.to(u.km / u.second).value
    berv_pipeline = (f * (1 + berv_pipeline/c) - 1) * c


    tmmean = np.array([h['HIERARCH ESO QC TMMEAN USED'] for h in H])
    mjdobs = np.array([h['MJD-OBS'] for h in H])
    texp = np.array([h['EXPTIME'] for h in H])
    jd = mjdobs + 24e5 + 0.5 + (texp * tmmean)/60/60/24

    if verbose:
        logger.info(f"Unique exposure times: {np.unique(texp)}")

    berv = []
    if verbose:
        pbar = enumerate(jd)
    else:
        pbar = tqdm(enumerate(jd), total=len(jd),
                    unit='observation', desc='Computing BERV')

    for i, _jd in pbar:
        if use_gaia_meassurements:
            if not hasattr(self, 'gaia'):
                raise ValueError('No Gaia data available')
            
            target = self.gaia.coords
            pmra = self.gaia.pmra
            pmdec = self.gaia.pmdec
            epoch = Time('J2016').jd
        else:
            ra = H[i]['* TARG ALPHA'][0]
            ra = f'{ra:09.2f}'
            ra = ra[:2] + 'h' + ra[2:4] + 'm' + ra[4:] + 's'

            dec = H[i]['* TARG DELTA'][0]
            if dec < 0:
                dec = f'{dec:010.2f}'
            else:
                dec = f'{dec:09.2f}'
            if dec.startswith('-'):
                dec = dec[:3] + 'd' + dec[3:5] + 'm' + dec[5:] + 's'
            else:
                dec = dec[:2] + 'd' + dec[2:4] + 'm' + dec[4:] + 's'

            target = SkyCoord(ra, dec)
            pmra = H[i]['* TARG PMA'][0] * 1e3
            pmdec = H[i]['* TARG PMD'][0] * 1e3
            epoch = Time('J2000').jd

        if verbose:
            logger.info(f'jd: {_jd}')
            logger.info(f'\t ra: {target.ra}')
            logger.info(f'\t dec: {target.dec}')
            logger.info(f'\t pmra: {pmra}')
            logger.info(f'\t pmdec: {pmdec}')
                        

        px = plx or 0.0
        out = get_BC_vel(_jd, obsname=obsname, rv=0.0, px=px, zmeas=0.0, epoch=epoch,
                         ra=target.ra.value, dec=target.dec.value, pmra=pmra, pmdec=pmdec)
        # print(out[1][3])
        berv.append(out[0])
    
    berv = np.array(berv).flatten()

    if ignore_mask: # ignore the system's masked points
        pass
    else: # mask points in the BERV output as well
        bjd = bjd[self.mask]
        berv = berv[self.mask]
        berv_pipeline = berv_pipeline[self.mask]

    fig = None
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), dpi=dpi, sharex=True,
                                constrained_layout=True)
    
        axs[0].set_title(f'{self.star}', loc='right')
        axs[0].plot(bjd, berv_pipeline*1e3, '.', label='pipeline', alpha=0.5)
        axs[0].plot(bjd, berv, '.', label='barycorrpy', alpha=0.5)
        axs[0].legend(bbox_to_anchor=(0.0, 1.15), loc=2, borderaxespad=0., ncol=2)
        axs[0].set(xlabel='BJD - 2450000', ylabel='BERV [m/s]')


        if plx is not None:
            epoch = 55500
            sa = self.secular_acceleration(just_compute=True)
            print('sa:', sa)
            sec_acc = sa.value * (bjd - epoch) / 365.25

            axs[0].plot(bjd, sec_acc)

            # fitp = np.polyfit(bjd - epoch, diff, 1)
            # axs[1].plot(bjd, np.polyval(fitp, bjd - epoch))
            # axs[1].plot(bjd, np.mean(diff) + diff - np.polyval(fitp, bjd - epoch), '.')

        if plx is None:
            diff = berv - berv_pipeline*1e3
            label=r'BERV$_{\rm barycorrpy}$ - BERV$_{\rm pipeline}$'
        else:
            diff = berv + sec_acc - berv_pipeline*1e3
            label=r'BERV$_{\rm barycorrpy}$ (+SA) - BERV$_{\rm pipeline}$'

        axs[1].plot(bjd, diff, 'k.', label=label)
        axs[1].axhline(np.mean(diff), ls='--', c='k', alpha=0.1)

        from adjustText import adjust_text
        text = axs[1].text(bjd.max(), diff.min() + 0.1*diff.ptp(), 
                           f'ptp: {diff.ptp()*1e2:.2f} cm/s',
                           ha='right', va='bottom', color='g', alpha=0.8)
        axs[1].plot([bjd[np.argmax(diff)], bjd.max() + 0.05 * bjd.ptp()], 
                    [np.max(diff), np.max(diff)], 'g--', alpha=0.3)
        axs[1].plot([bjd[np.argmin(diff)], bjd.max() + 0.05 * bjd.ptp()], 
                    [np.min(diff), np.min(diff)], 'g--', alpha=0.3)

        ax = axs[1].twinx()
        diff_cms = 1e2*(diff - np.mean(diff))
        ax.plot(bjd, diff_cms, alpha=0)
        ma = np.max(np.abs(ax.get_ylim()))
        ax.set_ylim(-1 - 5*round(ma/5), 1 + 5*round(ma/5))
        ax.set(ylabel='diff - mean(diff) [cm/s]')
        axs[1].set_ylim(np.mean(diff)-ma/100, np.mean(diff)+ma/100)

        axs[1].legend(bbox_to_anchor=(0.0, 1.15), loc=2, borderaxespad=0.)
        axs[1].set(xlabel='BJD - 2450000', ylabel='diff [m/s]')
    
        # adjust_text([text], va='bottom')

    return fig, {
        self.star: {
            'bjd': bjd,
            'berv_pipeline': berv_pipeline*1e3,
            'berv_barycorrpy': berv
        }
    }


def plot_BERV_correction(self, H, A, V, berv2=None, berv6=None,
                         inset=False, inset_range=(3, 5)):
    fig, axs = plt.subplot_mosaic('ab\ncc\ndd\nee', constrained_layout=True, figsize=(2*3.57, 10))

    if berv2 is None:
        _, berv2 = BERV(self, H, plot=False)
    if berv6 is None:
        _, berv6 = BERV(self, H, A=A, V=V, plot=False)

    self.plot(ax=axs['a'], ms=2)
    axs['a'].set_title('original', loc='right')
    self.gls(ax=axs['e'], label='original', color='r', alpha=0.5,
             fill_between=True, samples_per_peak=20)

    temp_vrad = self.vrad.copy()
    self.vrad[self.mask] = self.vrad[self.mask] - berv2[self.star]['berv_pipeline'].value + berv6[self.star]['berv_pipeline'].value

    self.plot(ax=axs['b'], ms=2)
    axs['b'].set_title('after correction', loc='right')

    diff = temp_vrad[self.mask] - self.vrad[self.mask]

    axs['c'].plot(self.mtime, diff, 'k.')
    axs['c'].set_title('RV difference', loc='right')
    axs['c'].set(xlabel='BJD - 2450000', ylabel='RV diff [m/s]')

    text = axs['c'].text(self.mtime.max(), diff.min() + 0.1*diff.ptp(), 
                         f'ptp: {diff.ptp()*1e2:.2f} cm/s',
                         ha='right', va='bottom', color='g', alpha=0.8)
    axs['c'].plot([self.mtime[np.argmax(diff)], self.mtime.max() + 0.05 * self.mtime.ptp()], 
                [np.max(diff), np.max(diff)], 'g--', alpha=0.3)
    axs['c'].plot([self.mtime[np.argmin(diff)], self.mtime.max() + 0.05 * self.mtime.ptp()], 
                [np.min(diff), np.min(diff)], 'g--', alpha=0.3)


    f, p = LombScargle(self.mtime, diff).autopower(maximum_frequency=1.0, samples_per_peak=10)
    axs['d'].semilogx(1/f, p, color='k', alpha=0.8)
    axs['d'].vlines([365.25, 365.25/2], 0, 1, color='k', ls='--', alpha=0.3)
    axs['d'].set(xlabel='Period [days]', ylabel='normalized power', ylim=(0, 1))
    axs['d'].set_title('GLS of RV difference', loc='right')

    if inset:
        inset = axs['d'].inset_axes(bounds=[0.15, 0.3, 0.3, 0.6])
        m = (1/f > inset_range[0]) & (1/f < inset_range[1])
        inset.plot(1/f[m], p[m], color='k', alpha=0.8)
        inset.set(xlim=inset_range, yticks=[])
        inset.minorticks_on()

    self.gls(ax=axs['e'], label='after correction', color='g', alpha=1,
             lw=0.8, samples_per_peak=20)
    axs['e'].set(xlabel='Period [days]', ylabel='normalized power')
    axs['e'].sharex(axs['d'])

    self.vrad = temp_vrad
    return fig