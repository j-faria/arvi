import os
import numpy as np

from .setup_logger import logger

try:
    import kima
    from kima.pykima.utils import chdir
    from kima import distributions
    from kima import RVData, RVmodel
    kima_available = True
except ImportError:
    kima_available = False


def try_to_guess_prior(model, prior):
    if 'jitter' in prior:
        return 'Jprior'
    if 'vsys' in prior:
        return 'Cprior'
    return None


def run_kima(self, run=False, load=False, run_directory=None, priors={}, **kwargs):
    if not kima_available:
        raise ImportError('kima not available, please install with `pip install kima`')

    instruments = [inst for inst in self.instruments if self.NN[inst] > 1]
    time = [getattr(self, inst).mtime for inst in instruments]
    vrad = [getattr(self, inst).mvrad for inst in instruments]
    err = [getattr(self, inst).msvrad for inst in instruments]
    data = RVData(time, vrad, err, instruments=instruments)

    fix = kwargs.pop('fix', False)
    npmax = kwargs.pop('npmax', 1)
    model = RVmodel(fix=fix, npmax=npmax, data=data)

    model.trend = kwargs.pop('trend', False)
    model.degree = kwargs.pop('degree', 0)

    model.studentt = kwargs.pop('studentt', False)
    model.enforce_stability = kwargs.pop('enforce_stability', False)
    model.star_mass = kwargs.pop('star_mass', 1.0)

    if kwargs.pop('gaussian_priors_individual_offsets', False):
        from kima.pykima.utils import get_gaussian_priors_individual_offsets
        model.individual_offset_prior = get_gaussian_priors_individual_offsets(data, use_std=True)

    if kwargs.pop('kuma', False):
        model.conditional.eprior = distributions.Kumaraswamy(0.867, 3.03)

    for k, v in priors.items():
        try:
            if 'conditional' in k:
                setattr(model.conditional, k.replace('conditional.', ''), v)
            else:
                setattr(model, k, v)

        except AttributeError:
            msg = f'`RVmodel` has no attribute `{k}`, '
            if guess := try_to_guess_prior(model, k):
                msg += f'did you mean `{guess}`?'
            logger.warning(msg)
            return

    if run_directory is None:
        run_directory = os.getcwd()

    if run:
        
        # TODO: use signature of kima.run to pop the correct kwargs
        # model_name = model.__class__.__name__
        # model_name = f'kima.{model_name}.{model_name}'
        # signature, defaults = [sig for sig in kima.run.__nb_signature__ if model_name in sig[0]]

        with chdir(run_directory):
            kima.run(model, **kwargs)
        
    if load:
        with chdir(run_directory):
            res = kima.load_results(model)
        return data, model, res

    return data, model