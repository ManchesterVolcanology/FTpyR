import copy
import logging
import numpy as np
from scipy.interpolate import CubicSpline

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

    def get_free_values_list(self):
        """Return a list of the fitted parameter values."""
        vals_list = []
        for layer in self.layers.values():
            if layer.temperature.vary:
                vals_list += [layer.temperature.value]
            vals_list += [gas.value for gas in layer.gases.values() if gas.vary]

        vals_list += [(p.value) for p in self.variables.values() if p.vary]

        return vals_list

    def get_values_list(self):
        """Return a list of all parameter values."""
        vals_list = []
        for layer in self.layers.values():
            vals_list += [layer.temperature.value]
            vals_list += [gas.value for gas in layer.gases.values()]

        vals_list += [(p.value) for p in self.variables.values()]

        return vals_list

    def get_free_values_dict(self):
        """Return a dictionary of the fitted parameter values."""
        vals_dict = {}
        for layer in self.layers.values():
            layer_id = layer.layer_id
            if layer.temperature.vary:
                vals_dict[f'{layer_id}_temperature'] = layer.temperature.value
            for key, gas in layer.gases.items():
                if gas.vary:
                    vals_dict[f'{layer_id}_{key}'] = gas.value

        for key, param in self.variables.items():
            if param.vary:
                vals_dict[key] = param.value

        return vals_dict

    def get_values_dict(self):
        """Return a dictionary of all parameter values."""
        vals_dict = {}
        for layer in self.layers.values():
            layer_id = layer.layer_id
            vals_dict[f'{layer_id}_temperature'] = layer.temperature.value
            for key, gas in layer.gases.items():
                vals_dict[f'{layer_id}_{key}'] = gas.value

        for key, param in self.variables.items():
            vals_dict[key] = param.value

        return vals_dict

    def get_popt_dict(self):
        """Return a dictionary of the optimised parameters."""
        poptdict = {}
        for layer in self.layers.values():
            layer_id = layer.layer_id
            if layer.temperature.vary:
                poptdict[f'{layer_id}_temperature'] = layer.temperature.fit_value
            for key, gas in layer.gases.items():
                if gas.vary:
                    poptdict[f'{layer.layer_id}_{key}'] = gas.fit_value

        for key, param in self.variables.items():
            if param.vary:
                poptdict[key] = param.fit_value

        return poptdict

    def get_popt_list(self):
        """Return a list of the optimised parameters."""
        poptlist = []
        for layer in self.layers.values():
            if layer.temperature.vary:
                poptlist += [layer.temperature.fit_value]
            poptlist += [
                gas.fit_value for gas in layer.gases.values() if gas.vary
            ]

        poptlist += [(p.fit_value) for p in self.variables.values() if p.vary]

        return poptlist

    def get_all_parameters(self):
        """Get a dictionary of all parameters."""
        params_dict = {}
        for layer in self.layers.values():
            params_dict[f'{layer.layer_id}_temperature'] = layer.temperature
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
            if layer.temperature.vary:
                lo_bounds += [layer.temperature.bounds[0]]
                hi_bounds += [layer.temperature.bounds[1]]

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

        msg = 'FTpyR Parameters:'

        # Add layer information
        for layer in self.layers.values():
            msg += f'\n{layer.layer_id} layer:'
            msg += f'\n    Temperature: {layer.temperature.value} K'
            if layer.temperature.vary:
                b1, b2 = layer.temperature.bounds
                msg += f' (free [{b1} / {b2}])'
                if not np.isnan(layer.temperature.fit_value):
                    fit_value = layer.temperature.fit_value
                    fit_error = layer.temperature.fit_error
                    msg += f' -> {fit_value:.5g} (± {fit_error:.5g})'
            else:
                msg += ' (fixed)'


            msg += f'\n    Pressure: {layer.pressure} mb (fixed)'
            msg += f'\n    Path Length: {layer.path_length} m (fixed)'

            # Add layer gases
            for gas in layer.gases.values():
                msg += f'\n    {gas.name}: {layer.get_value(gas.name):.4g}'
                if gas.vary:
                    b1, b2 = layer.get_bounds(gas.name)
                    msg += f' (free [{b1} / {b2}])'
                else:
                    msg += ' (fixed)'

                if not np.isnan(gas.fit_value):
                    fit_value = layer.get_fit_value(gas.name)
                    fit_error = layer.get_fit_error(gas.name)
                    msg += f' -> {fit_value:.4} (± {fit_error:.4})'

        # Add other parameters
        msg += '\nOther Parameters:'
        for key, param in self.variables.items():
            msg += f'\n    {key}: {param.value}'
            if param.vary:
                msg += f' (free [{param.bounds[0]} / {param.bounds[1]}])'
            else:
                msg += ' (fixed)'

            if not np.isnan(param.fit_value):
                msg += f' -> {param.fit_value:.4} (± {param.fit_error:.4})'

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

    def __init__(self, layer_id, temperature=298.15, pressure=1013.25,
                 path_length=100, atmos_flag=False, vary_temperature=False,
                 temperature_bounds=[273.15, 323.15], temperature_step=1):
        """Initialise the object.

        Parameters
        ----------
        layer_id : string
            Unique ID string for the layer
        temperature : float, optional
            Layer temperature (in degrees K), by default 293
        pressure : float, optional
            Layer pressure (in millibars), by default 1013.25
        path_length : int, optional
            Layer path length (in meters), by default 100
        atmos_flag : bool, optional
            If True, then sets up a whole atmosphere layer (for use in solar
            occultation measurements), by default False
        vary_temperature: bool, optional
            If True, then temperature is included in the fit, by default False
        temperature_bounds : tuple, optional
            If vary_temperature is True, then temperature_bounds sets the fit
            low and high bounds (in Kelvin) for the temperature Parameter, by
            default [273.15, 323.15]
        temperature_step : float, optional
            Sets the step (in Kelvin) in temperature to use when calculating the
            optical depth array for each gas in the layer, by default 1
        """

        self.gases = {}
        self.cross_sections = {}
        self.path_amounts = {}
        self.interpolators = {}
        self.layer_id = layer_id
        self.atmos_flag = atmos_flag

        if not atmos_flag:
            self.temperature = Parameter(
                name='temperature', value=temperature, vary=vary_temperature,
                bounds=temperature_bounds
            )
            self.temperature_step = temperature_step
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
            path_amt = self.cross_sections[gas].path_amount.data[0]
            return self.gases[gas].value * path_amt
        except KeyError:
            return self.gases[gas].value

    def get_bounds(self, gas):
        """Return the scaled gas bounds."""
        try:
            path_amt = self.cross_sections[gas].path_amount.data[0]
            return np.multiply(self.gases[gas].bounds, path_amt)
        except KeyError:
            return self.gases[gas].bounds

    def get_value(self, gas):
        """Return the scaled gas initial amount."""
        try:
            path_amt = self.cross_sections[gas].path_amount.data[0]
            return self.gases[gas].value * path_amt
        except KeyError:
            return self.gases[gas].value

    def get_fit_value(self, gas):
        """Return the scaled gas fitted amount."""
        try:
            path_amt = self.cross_sections[gas].path_amount.data[0]
            return self.gases[gas].fit_value * path_amt
        except KeyError:
            return self.gases[gas].fit_value

    def get_fit_error(self, gas):
        """Return the scaled gas fitted amount."""
        try:
            path_amt = self.cross_sections[gas].path_amount.data[0]
            return self.gases[gas].fit_error * path_amt
        except KeyError:
            self.gases[gas].fit_error

    def set_cross_section(self, gas, cross_section):

        # Store the cross section and path amount for that gas
        self.cross_sections[gas] = cross_section

        # If temperature is varying, perform a cubic interpolation and store the
        # interpolator for future use
        if self.temperature.vary:
            self.interpolators[gas] = CubicSpline(
                cross_section.temperature.data,
                cross_section.optical_depth.data
            )

    def get_cross_section(self, gas, temperature=None):
        # If varying the temperature, interpolate between stored cross-sections
        if self.temperature.vary:
            return self.interpolators[gas](temperature)

        # Otherwise just return the given optical depth cross-section
        else:
            return self.cross_sections[gas].optical_depth.data[0]

    def __repr__(self):
        """Nice printing behaviour."""
        output = str(
            f'Layer "{self.layer_id}":'
            f'\nTemperature = {self.temperature}'
            f'\nPressure = {self.pressure}'
            f'\nPath length = {self.path_length}\n'
            f'Gases:'
        )
        for gas in self.gases.keys():
            output += f' {gas}'
        return output
