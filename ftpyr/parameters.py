import copy
import logging
import numpy as np
from collections import OrderedDict

logger = logging.getLogger(__name__)


class Parameters(OrderedDict):
    """."""

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def add(self, name, value=0.0, vary=True, species=None, path=None,
            pres=None, temp=None, lo_bound=-np.inf, hi_bound=np.inf):
        """."""
        self.__setitem__(
            name,
            Parameter(
                name=name, value=value, vary=vary, species=species, path=path,
                pres=pres, temp=temp, lo_bound=lo_bound, hi_bound=hi_bound
            )
        )

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

    def get_bounds(self):
        """Return a list of parameter bounds."""
        return [
            [p.lo_bound for p in self.values() if p.vary],
            [p.hi_bound for p in self.values() if p.vary]
        ]

    def make_copy(self):
        """Return a deep copy of the Parameters object."""
        return copy.deepcopy(self)

    def pretty_print(self, mincolwidth=10, precision=4, cols='basic'):
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
            'all': [
                'name', 'value', 'vary', 'species', 'temp', 'pres', 'path',
                'fit_val', 'fit_err', 'lo_bound', 'hi_bound'
            ],
            'basic': ['name', 'value', 'vary', 'lo_bound', 'hi_bound']
        }

        # Make list of columns
        if cols == 'all' or cols == 'basic':
            cols = def_cols[cols]

        colwidth = np.zeros(len(cols))

        if 'name' in cols:
            i = cols.index('name')
            colwidth[i] = max([len(name) for name in self]) + 2

        if 'value' in cols:
            i = cols.index('value')
            colwidth[i] = max(
                [len(f'{p.value:.{precision}g}') for p in self.values()]
            ) + 2

        if 'vary' in cols:
            i = cols.index('vary')
            colwidth[i] = mincolwidth

        if 'lo_bound' in cols:
            i = cols.index('lo_bound')
            colwidth[i] = max(
                [len(f'{p.lo_bound:.{precision}g}') for p in self.values()]
            ) + 2

        if 'hi_bound' in cols:
            i = cols.index('hi_bound')
            colwidth[i] = max(
                [len(f'{p.hi_bound:.{precision}g}') for p in self.values()]
            ) + 2

        if 'species' in cols:
            i = cols.index('species')
            colwidth[i] = max([len(str(p.species)) for p in self.values()]) + 2

        if 'temp' in cols:
            i = cols.index('temp')
            colwidth[i] = max([len(f'{p.temp:.{precision}g}')
                               for p in self.values()]) + 2

        if 'pres' in cols:
            i = cols.index('pres')
            colwidth[i] = max(
                [len(f'{p.pres:.{precision}g}') for p in self.values()]
            ) + 2

        if 'path' in cols:
            i = cols.index('path')
            colwidth[i] = max(
                [len(f'{p.path:.{precision}g}') for p in self.values()]
            ) + 2

        if 'fit_val' in cols:
            i = cols.index('fit_val')
            colwidth[i] = max(
                [len(f'{p.fit_val:.{precision}g}') for p in self.values()]
            ) + 2

        if 'fit_err' in cols:
            i = cols.index('fit_err')
            colwidth[i] = max(
                [len(f'{p.fit_err:.{precision}g}') for p in self.values()]
            ) + 2

        # Make sure no widths are below the minimum
        colwidth = [
            int(mincolwidth) if colwidth[i] < mincolwidth
            else int(colwidth[i]) for i in range(len(cols))
        ]

        # Generate the title string
        title = ''
        for n, c in enumerate(cols):
            title += f'|{c:^{colwidth[n]}}'
        title += '|'

        # Generate the header row
        msg = f'\n{"MODEL PARAMETERS":^{len(title)}}\n{title}\n' + \
              f'{"-"*len(title)}\n'

        # Write a row for each parameter
        for name, p in self.items():
            d = {
                'name': f'{p.name}',
                'value': f'{p.value:.{precision}g}',
                'vary': f'{p.vary}',
                'lo_bound': f'{p.lo_bound:.{precision}g}',
                'hi_bound': f'{p.hi_bound:.{precision}g}',
                'species': f'{p.species}',
                'temp': f'{p.temp:.{precision}g}',
                'pres': f'{p.pres:.{precision}g}',
                'path': f'{p.path:.{precision}g}',
                'fit_val': f'{p.fit_val:.{precision}g}',
                'fit_err': f'{p.fit_err:.{precision}g}'
             }

            for col in cols:
                msg += f'|{d[col]:^{colwidth[cols.index(col)]}}'

            msg += '|\n'

        return(msg)


class Parameter(object):
    """."""

    def __init__(self, name, value, vary=True, species=None, path=None,
                 pres=None, temp=None, lo_bound=-np.inf, hi_bound=np.inf):
        """."""
        self.name = str(name)
        self.value = float(value)
        self.vary = bool(vary)
        self.lo_bound = lo_bound
        self.hi_bound = hi_bound
        if species is not None:
            self.species = str(species)
            self.path = float(path)
            self.pres = float(pres)
            self.temp = float(temp)
            self.original_od = None
            self.original_amt = None
            self.xsec_od = None
        else:
            self.species = None
            self.temp = np.nan
            self.pres = np.nan
            self.path = np.nan

        self.fit_val = np.nan
        self.fit_err = np.nan

    def value_to_ppmm(self):
        """Return the initial value in ppm.m."""
        if self.species is not None:
            return self.value * self.temp / (7.243e14 * self.pres)
        else:
            logger.warning(f'{self.name} parameter is not a gas!')

    def fit_val_to_ppmm(self):
        """Return the fit value in ppm.m."""
        if self.species is not None:
            return self.fit_val * self.temp / (7.243e14 * self.pres)
        else:
            logger.warning(f'{self.name} parameter is not a gas!')

    def fit_err_to_ppmm(self):
        """Return the fit error in ppm.m."""
        if self.species is not None:
            return self.fit_err * self.temp / (7.243e14 * self.pres)
        else:
            logger.warning(f'{self.name} parameter is not a gas!')
