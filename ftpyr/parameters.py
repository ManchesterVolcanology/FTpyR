import copy
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Parameters(object):
    """
    Parameters class to hold fitting parameters in least squares minimisation.
    """

    def __init__(self):
        self.layers = {}
        self.variables = {}

    def add(self, name, value=0.0, vary=True, bounds=[-np.inf, np.inf]):
        """Add a Parameter.

        Parameters
        ----------
        name : string
            Parameter unique ID string.
        value : float, optional
            Initial value for the parameter. Used as a priori estimate in fit,
            by default 0.0
        vary : bool, optional
            If True, then the parameter is fitted, otherwise it is fixed to
            value, by default True
        bounds : list, optional
            Lower and upper bounds for the parameter to be passed to the
            minimisation algorithm, by default [-np.inf, np.inf]
        """
        self.variables[name] = Parameter(
            name=name, value=value, vary=vary, bounds=bounds
        )

    def add_layer(self, layer):
        """Add a layer to the Paramaters.

        Parameters
        ----------
        layer : Layer object
            Layer to add.

        Raises
        ------
        ValueError
            Raises if multiples of the same layer ID are added
        """
        if layer.layer_id in self.layers:
            raise ValueError('Cannot have multiple layers with the same ID!')
        self.layers[layer.layer_id] = layer

    def get_fit_values_list(self):
        """Return a list of the fitted parameter values."""
        vals_list = []
        for layer in self.layers.values():
            vals_list += [gas.value for gas in layer.gases.values() if gas.vary]

        vals_list += [(p.value) for p in self.variables.values() if p.vary]

        return vals_list

    def get_values_list(self):
        """Return a list of all parameter values."""
        vals_list = []
        for layer in self.layers.values():
            vals_list += [gas.value for gas in layer.gases.values()]

        vals_list += [(p.value) for p in self.variables.values()]

        return vals_list

    def get_fit_values_dict(self):
        """Return a dictionary of the fitted parameter values."""
        vals_dict = {}
        for layer in self.layers.values():
            for key, gas in layer.gases.items():
                if gas.vary:
                    vals_dict[f'{layer.layer_id}_{key}'] = gas.value

        for key, param in self.variables.items():
            if param.vary:
                vals_dict[key] = param.value

        return vals_dict

    def get_values_dict(self):
        """Return a dictionary of all parameter values."""
        vals_dict = {}
        for layer in self.layers.values():
            for key, gas in layer.gases.items():
                vals_dict[f'{layer.layer_id}_{key}'] = gas.value

        for key, param in self.variables.items():
            vals_dict[key] = param.value

        return vals_dict

    def get_popt_dict(self):
        """Return a dictionary of the optimised parameters."""
        popt_dict = {}
        for layer in self.layers.values():
            for key, gas in layer.gases.items():
                if gas.vary:
                    popt_dict[f'{layer.layer_id}_{key}'] = gas.fit_value

        for key, param in self.variables.items():
            if param.vary:
                popt_dict[key] = param.fit_value

        return popt_dict

    def get_popt_list(self):
        """Return a list of the optimised parameters."""
        vals_list = []
        for layer in self.layers.values():
            vals_list += [
                gas.fit_value for gas in layer.gases.values() if gas.vary
            ]

        vals_list += [(p.fit_value) for p in self.variables.values() if p.vary]

        return vals_list

    def get_all_parameters(self):
        """Get a dictionary of all parameters."""
        params_dict = {}
        for layer in self.layers.values():
            for key, gas in layer.gases.items():
                params_dict[f'{layer.layer_id}_{key}'] = gas

        for key, param in self.variables.items():
            params_dict[key] = param

        return params_dict

    def get_bounds(self):
        """Return a list of parameter bounds."""
        lo_bounds = []
        hi_bounds = []

        for layer in self.layers.values():
            lo_bounds += [
                gas.bounds[0] for gas in layer.gases.values() if gas.vary
            ]
            hi_bounds += [
                gas.bounds[1] for gas in layer.gases.values() if gas.vary
            ]

        lo_bounds += [(p.bounds[0]) for p in self.variables.values() if p.vary]
        hi_bounds += [(p.bounds[1]) for p in self.variables.values() if p.vary]

        return [lo_bounds, hi_bounds]

    def make_copy(self):
        """Return a deep copy of the Parameters object."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        """Nice printing."""

        msg = 'FTpyR Parameters object:'

        # Add layer information
        for layer in self.layers.values():
            msg += f'\n{layer.layer_id} layer:'
            msg += f'\n    Temperature: {layer.temperature} K (fixed)'
            msg += f'\n    Pressure: {layer.pressure} mb (fixed)'
            msg += f'\n    Path Length: {layer.path_length} m (fixed)'

            # Add layer gases
            for gas in layer.gases.values():
                msg += f'\n    {gas.name}: {layer.get_value(gas.name):.4g}'
                if gas.vary:
                    bounds = layer.get_bounds(gas.name)
                    msg += f' (free [{bounds[0]} / {bounds[1]}])'
                else:
                    msg += ' (fixed)'

                if not np.isnan(gas.fit_value):
                    fit_value = layer.get_fit_value(gas.name)
                    fit_error = layer.get_fit_error(gas.name)
                    msg += f' -> {fit_value:.4g} (± {fit_error:.4g})'

        # Add other parameters
        msg += '\nOther Parameters:'
        for key, param in self.variables.items():
            msg += f'\n    {key}: {param.value}'
            if param.vary:
                msg += f' (free [{param.bounds[0]} / {param.bounds[1]}])'
            else:
                msg += ' (fixed)'

            if not np.isnan(param.fit_value):
                msg += f' -> {param.fit_value:.4g} (± {param.fit_error:.4g})'

        return msg


class Parameter(object):
    """Individual parameter class."""

    def __init__(self, name, value, vary=True, bounds=[-np.inf, np.inf],
                 layer_id=None):
        """Initialise object

        Parameters
        ----------
        name : string
            Unique id string for the parameter.
        value : float
            Initial value for the parameter. Used as a priori estimate in fit.
        vary : bool, optional
            If True, then the parameter is fitted, otherwise it is fixed to
            value, by default True
        bounds : list, optional
            Lower and upper bounds for the parameter to be passed to the
            minimisation algorithm, by default [-np.inf, np.inf]
        """
        self.name = str(name)
        self.value = float(value)
        self.vary = bool(vary)
        self.bounds = bounds
        self.layer_id = layer_id
        self.fit_value = np.nan
        self.fit_error = np.nan

    def __repr__(self):
        """Nice printing behaviour."""
        msg = f'{self.name}'
        if self.layer_id is not None:
            msg += f' ({self.layer_id})'

        msg += f': {self.value:.4g}'

        if self.vary:
            msg += f' (free [{self.bounds[0]} / {self.bounds[1]}])'
        else:
            msg += ' (fixed)'

        if not np.isnan(self.fit_value):
            msg += f' -> {self.fit_value:.4g} (± {self.fit_error:.4g})'

        return msg


class Layer(object):
    """Layer class to hold settings for individual atmospheric layers."""

    def __init__(self, layer_id, temperature=293.15, pressure=1013.25,
                 path_length=100, atmos_flag=False):
        """Initialise the object.

        Parameters
        ----------
        layer_id : string
            Unique ID string for the layer
        temperature : float, optional
            Layer temperature (in degrees K), by default 293.15
        pressure : float, optional
            Layer pressure (in millibars), by default 1013.25
        path_length : int, optional
            Layer path length (in meters), by default 100
        atmos_flag : bool, optional
            If True, then sets up a whole atmosphere layer (for use in solar
            occultation measurements), by default False
        """

        self.gases = {}
        self.optical_depths = {}
        self.path_amounts = {}

        self.layer_id = layer_id
        self.atmos_flag = atmos_flag

        if not atmos_flag:
            self.temperature = temperature
            self.pressure = pressure
            self.path_length = path_length
        else:
            raise Exception('Whole atmosphere layers not setup yet!')

    def add_gas(self, species, value=1.0, vary=True, bounds=[-np.inf, np.inf]):
        """Add a gas to the layer.

        Parameters
        ----------
        species : string
            The species name (must match those used in RFM, e.g. H2O, SO2...)
        value : float, optional
            The initial value to fit, by default 1.0. Note that this is a
            multiple of the path amount computed by RFM, not the absolute path
            amount (for computational stability).
        vary : bool, optional
            If True, then the gas amount is varied from value, by default True
        bounds : list, optional
            Lower and upper bounds on the fit value, by default
            [-np.inf, np.inf]

        Raises
        ------
        ValueError
            Raises if multiples of the same species are added to the same layer
        """
        if species in self.gases:
            raise ValueError(f'Layer already contains {species}!')
        self.gases[species] = Parameter(
            name=species, value=value, vary=vary, bounds=bounds,
            layer_id=self.layer_id
        )

    def get_value(self, gas):
        """Return the scaled gas initial amount."""
        try:
            return self.gases[gas].value * self.path_amounts[gas]
        except KeyError:
            return self.gases[gas].value

    def get_bounds(self, gas):
        """Return the scaled gas bounds."""
        try:
            return np.multiply(self.gases[gas].bounds, self.path_amounts[gas])
        except KeyError:
            return self.gases[gas].bounds

    def get_fit_value(self, gas):
        """Return the scaled gas fitted amount."""
        try:
            return self.gases[gas].fit_value * self.path_amounts[gas]
        except KeyError:
            return self.gases[gas].fit_value

    def get_fit_error(self, gas):
        """Return the scaled gas fitted amount."""
        try:
            return self.gases[gas].fit_error * self.path_amounts[gas]
        except KeyError:
            self.gases[gas].fit_error

    def __repr__(self):
        """Nice printing behaviour."""
        output = str(
            f'Layer "{self.layer_id}": '
            f'Temperature = {self.temperature} K, '
            f'Pressure = {self.pressure} mb, '
            f'Path length = {self.path_length} m\n'
            f'Gases:'
        )
        for gas in self.gases.keys():
            output += f' {gas}'
        return output
