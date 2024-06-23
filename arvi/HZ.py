import numpy as np
from astropy.constants import G
from astropy import units

# Coeffcients to be used in the analytical expression to calculate habitable
# zone flux boundaries
seffsun = [1.776, 1.107, 0.356, 0.320, 1.188, 0.99]
a = [2.136e-4, 1.332e-4, 6.171e-5, 5.547e-5, 1.433e-4, 1.209e-4]
b = [2.533e-8, 1.580e-8, 1.698e-9, 1.526e-9, 1.707e-8, 1.404e-8]
c = [-1.332e-11, -8.308e-12, -3.198e-12, -2.874e-12, -8.968e-12, -7.418e-12]
d = [-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16, -2.084e-15, -1.713e-15]
seffsun = np.array(seffsun)
a, b, c, d = np.array([a, b, c, d])


def getHZ(teff, lum, which='conservative'):
    """ 
    Calculate the limits of the HZ, according to Kopparapu et al.(2013).

    Parameters
    ----------
    teff : float
        Stellar effective temperature
    lum : float
        Stellar luminosity (in units of the solar luminosity)
    which : str
        Either 'conservative' or 'optimistic' habitable zone

    Returns
    -------
    innerHZf, outerHZf : floats
        Inner and outer limits of the HZ [in stellar flux compared to the Sun].
    innerHZd, outerHZd : floats
        Inner and outer limits of the HZ [in AU].
    """
    tstar = teff - 5780
    seff = seffsun + a * tstar + b * tstar**2 + c * tstar**3 + d * tstar**4
    dist = np.sqrt(lum / seff)

    # the order of the coefficients is
    # recentVenus
    # runawayGreenhouse
    # maxGreenhouse
    # earlyMars
    # runaway5Me
    # runaway10Me

    if which in ('conservative', 'con'):
        return (seff[1], seff[2]), (dist[1], dist[2])
    elif which in ('optimistic', 'opt'):
        return (seff[0], seff[3]), (dist[0], dist[3])
    else:
        raise ValueError(f'Could not recognise `which={which}`')


def getHZ_period(teff, Mstar, Mplanet, lum=1, Mplanet_units='earth',
                 which='conservative'):
    """
    Calculate the period limits of the HZ.

    Parameters
    ----------
    teff : float
        Stellar effective temperature
    Mstar : float
        Mass of the star
    Mplanet : float
        Mass of the planet
    lum : float
        Stellar luminosity (in units of the solar luminosity)
    Mplanet_units : str
        Units of the planet mass, 'earth' or 'jupiter'
    which : str
        Either 'conservative' or 'optimistic' habitable zone
    
    Returns
    -------
    innerHZd, outerHZd : floats
        Inner and outer limits of the HZ [in days].
    """
    # this function is nothing more than Kepler's 3rd law

    f = 4 * np.pi**2 / G
    
    HZa = np.array(getHZ(teff, lum, which)[1]) * units.AU

    Mstar = Mstar * units.solMass

    if Mplanet_units.lower() == 'earth':
        Mplanet = Mplanet * units.earthMass
    elif Mplanet_units.lower() in ('jupiter', 'jup'):
        Mplanet = Mplanet * units.jupiterMass

    return np.sqrt(f * HZa**3 / (Mstar + Mplanet)).to(units.day)