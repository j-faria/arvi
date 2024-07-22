from tqdm import tqdm
from astropy.io import fits
import iCCF

from . import config

def get_headers(self, check_lesta=False, lesta_username=config.username,
                check_exo2=False, instrument=None):
    try:
        import paramiko
    except ImportError:
        raise ImportError("paramiko is not installed. Please install it with 'pip install paramiko'")

    H = []

    if check_lesta:
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect("lesta02.astro.unige.ch", username=lesta_username)
            sftp = ssh.open_sftp()

            pbar = tqdm(self.raw_file, total=len(self.raw_file), unit='file', desc='Reading headers')
            for f in pbar:
                f = f.replace('espresso/', '/projects/astro/ESPRESSODRS/')
                f = f.replace('nirps/', '/projects/astro/NIRPSDRS/')
                f = f.replace('.fits', '_CCF_A.fits')#.replace(':', r'\:')
                with sftp.open(f) as fp:
                    header = fits.getheader(fp)
                    H.append(header)

    if len(H) == 0 and check_exo2:
        raise NotImplementedError('getting headers from exo2 not yet implemented')

    if len(H) == 0:
        self.download_ccf()
        if instrument is None:
            I = iCCF.from_file(f'{self.star}_downloads/*CCF_A.fits')
        else:
            I = iCCF.from_file(f'{self.star}_downloads/r.{instrument}.*CCF_A.fits',
                               guess_instrument='HARPS' not in instrument)
        H = [i.HDU[0].header for i in I]

    # sort by BJD
    H = sorted(H, key=lambda x: x['HIERARCH ESO QC BJD'])

    return H

