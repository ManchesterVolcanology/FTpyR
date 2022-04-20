import copy
import numpy as np
from collections import OrderedDict


class Parameters(OrderedDict):
    """."""

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def add(self, name, value=0.0, vary=True, species=None, layer=None):
        """."""
        self.__setitem__(name, Parameter(name=name,
                                         value=value,
                                         vary=vary,
                                         species=species,
                                         layer=layer))

    def extract_gases(self, layer_key=None):
        """Return only gas parameters based on layer key."""
        if layer_key is not None:
            return OrderedDict(
                (k, v) for [k, v] in self.items()
                if v.species is not None and v.layer == layer_key
            )
        else:
            return OrderedDict(
                (k, v) for [k, v] in self.items()
                if v.species is not None
            )

    def valuesdict(self):
        """Return an ordered dictionary of all parameter values."""
        return OrderedDict((p.name, p.value) for p in self.values())

    def fittedvaluesdict(self):
        """Return an ordered dictionary of fitted parameter values."""
        return OrderedDict((p.name, p.value) for p in self.values() if p.vary)

    def popt_dict(self):
        """Return a dictionary of the optimised parameters."""
        return OrderedDict((p.name, p.fit_val)
                           for p in self.values() if p.vary)

    def valueslist(self):
        """Return a list of all parameter values."""
        return [(p.value) for p in self.values()]

    def fittedvalueslist(self):
        """Return a list of the fitted parameter values."""
        return [(p.value) for p in self.values() if p.vary]

    def popt_list(self):
        """Return a list of the optimised parameters."""
        return [(p.fit_val) for p in self.values() if p.vary]

    def make_copy(self):
        """Return a deep copy of the Parameters object."""
        return copy.deepcopy(self)

    def pretty_print(self, mincolwidth=7, precision=4, cols='basic'):
        """Print the parameters in a nice way.

        Parameters
        ----------
        mincolwidth : int, optional
            Minimum width of the columns. Default is 7
        precision : int, optional
            Number of significant figures to print to. Default is 4
        cols : str or list, optional
            The columns to be printed. Either "all" for all columns, "basic"
            for the name, value and if it is fixed or a list of the desired
            column names. Default is "basic"

        Returns
        -------
        msg : str
            The formatted message to print
        """
        # Set default column choices
        def_cols = {
            'all': ['name', 'value', 'vary', 'layer', 'species', 'fit_val',
                    'fit_err'],
            'basic': ['name', 'value', 'vary']
        }

        # Make list of columns
        if cols == 'all' or cols == 'basic':
            cols = def_cols[cols]

        colwidth = [mincolwidth] * (len(cols))

        if 'name' in cols:
            i = cols.index('name')
            colwidth[i] = max([len(name) for name in self]) + 2

        if 'value' in cols:
            i = cols.index('value')
            colwidth[i] = max([len(f'{p.value:.{precision}g}')
                               for p in self.values()]) + 2

        if 'vary' in cols:
            i = cols.index('vary')
            colwidth[i] = mincolwidth

        if 'layer' in cols:
            i = cols.index('layer')
            colwidth[i] = mincolwidth

        if 'species' in cols:
            i = cols.index('species')
            colwidth[i] = mincolwidth

        if 'fit_val' in cols:
            i = cols.index('fit_val')
            colwidth[i] = max([len(f'{p.fit_val:.{precision}g}')
                               for p in self.values()]) + 2

        if 'fit_err' in cols:
            i = cols.index('fit_err')
            colwidth[i] = max([len(f'{p.fit_err:.{precision}g}')
                               for p in self.values()]) + 2

        for n, w in enumerate(colwidth):
            if w < mincolwidth:
                colwidth[n] = mincolwidth

        # Generate the string
        title = ''
        for n, c in enumerate(cols):
            title += f'|{c:^{colwidth[n]}}'
        title += '|'

        msg = f'\n{"MODEL PARAMETERS":^{len(title)}}\n{title}\n' + \
              f'{"-"*len(title)}\n'

        for name, p in self.items():
            d = {'name': f'{p.name}',
                 'value': f'{p.value:.{precision}g}',
                 'layer': f'{p.layer}',
                 'species': f'{p.species}',
                 'fit_val': f'{p.fit_val:.{precision}g}',
                 'fit_err': f'{p.fit_err:.{precision}g}',
                 'vary': f'{p.vary}'
                 }

            for col in cols:
                msg += f'|{d[col]:^{colwidth[cols.index(col)]}}'

            msg += '|\n'

        return(msg)


class Parameter(object):
    """."""

    def __init__(self, name, value, vary=True, species=None, layer=None):
        """."""
        self.name = str(name)
        self.value = float(value)
        self.vary = bool(vary)
        if species is not None:
            self.species = str(species).upper()
            self.layer = str(layer)
            self.original_od = None
            self.original_amt = None
        else:
            self.species = None
            self.layer = None

        self.fit_val = np.nan
        self.fit_err = np.nan
