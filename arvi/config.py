
def instancer(cls):
    return cls()

@instancer
class config:
    # configuration values
    __conf = {
        # whether to return self from (some) RV methods
        'return_self': False,
        # whether to adjust instrument means before gls by default
        'adjust_means_gls': True,
        # whether to check internet connection before querying DACE
        'check_internet': False,
        # make all DACE requests without using a .dacerc file
        'request_as_public': False,
        # username for DACE servers
        'username': 'desousaj',
        # debug
        'debug': False,
    }
    # all, for now
    __setters = list(__conf.keys())

    def __getattr__(self, name):
        if name in ('__custom_documentations__', ):
            # return {'return_self': 'help!'}
            return {}

        return self.__conf[name]

    def __setattr__(self, name, value):
        if name in config.__setters:
            self.__conf[name] = value
        else:
            raise NameError(f"unknown configuration name '{name}'")
