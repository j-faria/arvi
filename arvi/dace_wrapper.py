import numpy as np
from dace_query.spectroscopy import Spectroscopy
from .setup_logger import logger


def get_arrays(result, latest_pipeline=True):
    arrays = []
    instruments = list(result.keys())
    for inst in instruments:
        pipelines = list(result[inst].keys())
        if latest_pipeline:
            pipelines = [pipelines[-1]]
        for pipe in pipelines:
            modes = list(result[inst][pipe].keys())
            for mode in modes:
                if 'rjd' not in result[inst][pipe][mode]:
                    logger.error(f"No 'rjd' key for {inst} - {pipe}")
                    raise ValueError

                arrays.append(
                    ((inst, pipe, mode), result[inst][pipe][mode])
                )

    return arrays


def get_observations(star, save_rdb=False, verbose=True):
    result = Spectroscopy.get_timeseries(target=star,
                                         sorted_by_instrument=True,
                                         output_format='numpy')
    instruments = list(result.keys())

    if verbose:
        logger.info('RVs available from')
        with logger.contextualize(indent='   '):
            for inst in instruments:
                pipelines = list(result[inst].keys())
                for pipe in pipelines:
                    mode = list(result[inst][pipe].keys())[0]
                    N = len(result[inst][pipe][mode]['rjd'])
                    # LOG
                    logger.info(f'{inst:12s} {pipe:10s} ({N} observations)')

    return result


def save_rdb(data, columns):
    pass