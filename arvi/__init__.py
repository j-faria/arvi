__version__ = '0.0.2'
__all__ = []

from .timeseries import RV

_ran_once = False
def __getattr__(name: str):
    global _ran_once
    if _ran_once:
        return RV(name)
    else:
        _ran_once = True


# from importlib import import_module

# HACK_IMPORTS = True

# if HACK_IMPORTS:
#     timeseries = import_module('.timeseries', 'arvi')
#     class FakeModule(object):
#         def __getattribute__(self, name):
#             print(name)
#             r = RV(name)
#             return r
#     # first, load (some) subpackages
#     # then, replace the 'arvi' module to allow for dynamic imports
#     # of the type "from arvi import HD1"
#     import sys
#     sys.modules['arvi'] = FakeModule()
