__all__ = ['RV']

from .timeseries import RV

## OLD
# # the __getattr__ function is always called twice, so we need this
# # to only build and return the RV object on the second time
# _ran_once = False

def __getattr__(name: str):
    if name in (
        '_ipython_canary_method_should_not_exist_',
        '_ipython_display_',
        '_repr_mimebundle_',
        '__wrapped__'
    ):
        return

    try:
        globals()[name] = RV(name)
        return globals()[name]
    except ValueError as e:
        raise ImportError(e) from None
        # raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    ## OLD
    # # can't do it any other way :(
    # global _ran_once

    # if _ran_once:
    #     _ran_once = False
    #     return RV(name)
    # else:
    #     _ran_once = True
