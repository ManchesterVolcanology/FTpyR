import os
import logging
import numpy as np
import xarray as xr
from scipy import fft
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from ftpyr.rfm_control import RFM

logger = logging.getLogger(__name__)


class Analyser(object):
    """Controls the spectral fit for OP-FTIR spectra.

    Parameters
    ----------
    params : parameters.Parameters object
        Contains the parameter information, including the different gas
        species, background polynomial, intensity offset, wavenmumber shift,
        optical path difference and field of view. Note the following parameter
        names are reserved for specific variables:
            - *bg_poly*: the background polynomial coefficients
            - *shift*: the wavenumber shift coefficients
            - *offset*: the intensity offset coefficients
            - fov: the spectrometer field of view (milliradians)
            - opd: the spectrometer optical path difference (in cm)
        Note also all gas parameters must have a "species" variable defined
    rfm_path : str
        The path to the Reference Forward Model executable, either as a full
        path or with respect to the current working directory
    hitran_path : str
        The path to the HITRAN database, either as a full path or with respect
        to the RFM executable
    wn_start : float
        The start wavenumber of the fit window (cm-1)
    wn_stop : float
        The stop waveumber of the fit window (cm-1)
    solar_flag : bool, optional
        If True, then the calculation assumes a solar occultation measurement.
        Default is False.
    obs_height : float, optional
        The observer height, in meters above sea level. Only used if
        solar_flag=True. Default is 0.0.
    update_params : bool, optional
        If True, then the fit results will be used as the first guess for the
        next fit. Default is True.
    residual_limit : float, optional
        The residual limit to set as a "good" spectrum. Spectra with a max
        residual greater than residual_limit will have a fit_quality=2 and will
        not be used as the next first guess (if update_params==True). Default
        is 10.
    model_padding : float, optional
        The amount of padding of the fit window, in cm-1, to avoid convolution
        edge effects and allow a wavenumber shift. Default is 50.
    zero_fill_factor : int, optional
        If greater than zero then the spectra are zero-filled before analysis
        to artificially increase the sampling frequency. Increasing numbers
        use increassing zero-filling, determined by the next_fast_len function
        of the scipy.fft library. Default is 0.
    pts_per_cm : int, optional
        The number of points per cm-1 to use in the model wavenumber grid.
        Default is 100.
    apod_function : str, optional
        The apodisation function to use when generating the instrument ILS.
        Must be one of triangular, boxcar, NB_weak, NB_medium, NB_strong.
        Default is NB_medium.
    outfile : str, optional
        The output filename. If None, then no file is written. Default is None.
    tolerance : float, optional
        The fit tolerance to use in scipy.curve_fit. Default is 0.01.
    output_ppmm_flag : bool, optional
        If True then the output gas values are converted to ppm.m from
        molecules/cm2. Default is False.
    gas_auto_apriori : bool, optional
        If True, then the apriori values for gas species are set to the column
        amount calculate in RFM (assuming provided path length) or to zero if
        Parameter.vary=False. Default is True.

    Methods
    -------
    fit(spectrum, calc_od='all')
        Fit the provided spectrum

    fwd_model(x, *p0)
        The forward model used to fit spectra
    """

    def __init__(self, params, rfm_path, hitran_path, wn_start, wn_stop,
                 solar_flag=False, obs_height=0.0, update_params=True,
                 residual_limit=10, zero_fill_factor=0, model_padding=50,
                 pts_per_cm=25, apod_function='NB_medium', outfile=None,
                 tolerance=1e-8, output_ppmm_flag=False):
        """Initialise the Analyser."""
        # Generate the RFM object
        logger.debug('Setting up RFM')
        self.rfm = RFM(
            exe_path=rfm_path,
            hitran_path=hitran_path,
            wn_start=wn_start,
            wn_stop=wn_stop,
            model_padding=model_padding,
            solar_flag=solar_flag,
            obs_height=obs_height,
            pts_per_cm=pts_per_cm
        )

        # Calculate the optical depths of the layers
        self.params = self.rfm.calc_optical_depths(params=params)

        # Pull the fitted parameters
        self.p0 = self.params.get_free_values_list()

        # Store the fit window information
        self.wn_start = float(wn_start)
        self.wn_stop = float(wn_stop)
        self.pts_per_cm = int(pts_per_cm)

        # Store the quality check settings
        self.update_params = update_params
        self.residual_limit = residual_limit
        self.tolerance = tolerance

        # Add zero fill factor
        self.zero_fill_factor = zero_fill_factor

        # Add the output units
        self.output_ppmm_flag = output_ppmm_flag
        if self.output_ppmm_flag:
            gas_units = 'ppm.m'
        else:
            gas_units = 'molecules.cm-2'

        # Calculate the model x-grid
        # This includes a 1 cm-1 padding on either side to allow shifts
        npts_cm = int(self.wn_stop - self.wn_start) + model_padding*2
        self.xgrid_npts = self.pts_per_cm*(npts_cm) + 1
        self.model_grid = np.linspace(
            self.wn_start-model_padding,
            self.wn_stop+model_padding,
            self.xgrid_npts
        )

        # Generate the ILS
        ils_flag = np.any([
            self.params.variables['opd'].vary,
            self.params.variables['fov'].vary
        ])
        if ils_flag:
            self.regenerate_ils_flag = True
            self.apod_function = apod_function
        else:
            logger.info('Pre-generating initial ILS')
            self.ils = self.make_ils(
                max_opd=params.variables['opd'].value,
                fov=params.variables['fov'].value,
                nper_wn=self.pts_per_cm,
                wn=(self.model_grid.max() - self.model_grid.min()) / 2,
                apod_function=apod_function
            )
            self.regenerate_ils_flag = False

        # If an output file is defined, create it
        self.outfile = outfile
        if self.outfile is not None:
            # Make sure the parent directory exists
            head, tail = os.path.split(self.outfile)
            if head != '' and not os.path.isdir(head):
                os.makedirs(head)

            # Create the output file and write the header info
            with open(self.outfile, 'w') as ofile:
                # Write file header
                ofile.write(
                    f'#,FTpyR Output file: {outfile}\n'
                    f'#,StartWavenumber(cm-1),{wn_start}\n'
                    f'#,StopWavenumber(cm-1),{wn_stop}\n'
                    f'#,WavenumberPadding(cm-1),{model_padding}\n'
                    f'#,PointsPercm,{pts_per_cm}\n'
                    f'#,ZeroFillFactor,{zero_fill_factor}\n'
                    f'#,SolarFlag,{solar_flag}\n'
                    f'#,ObserverHeight(m),{obs_height}\n'
                    f'#,Apodisation,{apod_function}\n'
                    f'#,Tolerance,{tolerance}\n'
                    f'#,Apodisation,{apod_function}\n'
                    f'#,RFM,{rfm_path}\n'
                    f'#,HITRAN,{hitran_path}\n'
                    f'#,GasOutputUnits,{gas_units}\n'
                    '#,Layer,Temperature(K),Pressure(mb),PathLength(m),Gases\n'
                )

                # Write layer details
                for layer in params.layers.values():
                    ofile.write(
                        f'#,{layer.layer_id},{layer.temperature},'
                        f'{layer.pressure},{layer.path_length},'
                    )
                    for gas in layer.gases:
                        ofile.write(f'{gas}; ')
                    ofile.write('\n')

                # Write the fit results header
                ofile.write('Filename,Timestamp')
                for key in params.get_all_parameters():
                    ofile.write(f',{key},{key}_err')
                ofile.write(',FitQuality,MaxResidual,StdevResidual\n')

    def fit(self, spectrum):
        """Fit the provided spectrum.

        Parameters
        ---------
        spectrum : xarray.DataArray
            The spectrum to fit. Must be a DataArray with a single coords named
            "wavenumber" and with the following attrs:
                filename: the spectrum filename
                timestamp: the spectrum timestamp

        Returns
        -------
        FitResult object
            Holds the fit results and associated metadata
        """
        # Apply zero-filling
        if self.zero_fill_factor:
            spectrum = zero_fill(spectrum, self.zero_fill_factor)

        # Extract the region we are interested in
        sub_spec = spectrum.sel(wavenumber=slice(self.wn_start, self.wn_stop))

        # Perform the fit
        try:

            # Run fit and calculate parameter error
            popt, pcov = curve_fit(
                self.fwd_model,
                sub_spec.wavenumber,
                sub_spec.to_numpy(),
                self.p0,
                bounds=self.params.get_bounds(),
                ftol=self.tolerance
            )
            perr = np.sqrt(np.diag(pcov))

            # Set error code
            nerr = 0

        except RuntimeError:
            popt = np.full(len(self.p0), np.nan)
            perr = np.full(len(self.p0), np.nan)
            pcov = np.full((len(self.p0), len(self.p0)), np.nan)
            nerr = 1

        # Put the results into a FitResult object
        fit = FitResult(
            self, sub_spec, popt, pcov, perr, nerr, self.residual_limit
        )

        # Update the initial fit parameters
        if self.update_params and not fit.nerr:
            self.p0 = popt
        else:
            self.p0 = self.params.get_free_values_list()

        # Write fit results
        if self.outfile is not None:
            with open(self.outfile, 'a') as ofile:

                # Write the filename and timestamp
                metadata = spectrum.attrs
                ofile.write(f'{metadata["filename"]},{metadata["timestamp"]}')

                # Write the fitted parameters
                for param in self.params.get_all_parameters().values():

                    # If it is a gas parameter, convert if neccissary
                    if param.layer_id is not None:
                        layer = self.params.layers[param.layer_id]
                        if param.vary:
                            val = layer.get_fit_value(param.name)
                        else:
                            val = layer.get_value(param.name)
                        err = layer.get_fit_error(param.name)

                        if self.output_ppmm_flag:
                            val = molecm2_to_ppmm(
                                val, layer.temperature, layer.pressure
                            )
                            err = molecm2_to_ppmm(
                                err, layer.temperature, layer.pressure
                            )

                    # Otherwise just pull the parameter
                    else:
                        if param.vary:
                            val = param.fit_value
                        else:
                            val = param.value
                        err = param.fit_error

                    # Write the fit value and error
                    ofile.write(f',{val},{err}')

                # Add fit quality info and start a new line
                ofile.write(
                    f',{fit.nerr},'
                    f'{fit.data.residual.data.max()},'
                    f'{fit.data.residual.data.std()}\n'
                )

        return fit

    def fwd_model(self, x, *p0):
        """The forward model used to fit the measured spectrum.

        Parameters
        ----------
        x : numpy array
            The wavenumber grid of the measurement (cm-1).
        *p0 : floats
            The fitted parameters.

        Returns
        -------
        numpy array
            The fitted model spectrum, interpolated onto the measurement grid
        """
        # Get dictionary of fitted parameters
        par_vals = self.params.get_values_dict()

        # Update the fitted parameter values with those supplied to the forward
        # model
        for i, key in enumerate(self.params.get_free_values_dict().keys()):
            par_vals[key] = p0[i]

        # Unpack polynomial parameters
        bg_poly_coefs = [par_vals[n] for n in par_vals if 'bg_poly' in n]
        shift_coefs = [par_vals[n] for n in par_vals if 'shift' in n]
        offset_coefs = [par_vals[n] for n in par_vals if 'offset' in n]

        # Construct background polynomial
        bg_poly = np.polyval(np.flip(bg_poly_coefs), self.model_grid)

        # Calculate the gas optical depths for each layer
        layer_ods = np.zeros(
            [len(self.params.layers), len(self.model_grid)]
        )
        for n, layer in enumerate(self.params.layers.values()):

            # Pull the layer temperature
            temp = par_vals[f'{layer.layer_id}_temperature']

            # Calculate the layer optical depths
            layer_ods[n] = np.asarray([
                np.multiply(
                    layer.get_cross_section(gas, temp),
                    par_vals[f'{layer.layer_id}_{gas}']
                )
                for gas in layer.gases
            ]).sum(axis=0)

        # Summ to get the total optical depth
        total_od = np.sum(layer_ods, axis=0)

        # Convert to transmission
        total_trans = np.exp(-total_od)

        # Multiply by the background
        raw_spec = np.multiply(bg_poly, total_trans)

        # Add the offset
        offset = np.polyval(np.flip(offset_coefs), self.model_grid)
        raw_spec = np.add(raw_spec, offset)

        # Generate the ILS is any ILS parameters are being fit
        if self.regenerate_ils_flag:
            self.ils = self.make_ils(
                max_opd=par_vals['opd'],
                fov=par_vals['fov'],
                nper_wn=self.pts_per_cm,
                wn=(self.model_grid.max() - self.model_grid.min()) / 2,
                apod_function=self.apod_function
            )

        # Convolve with the ILS
        spec = np.convolve(raw_spec, self.ils, mode='same')

        # Apply shift and stretch to the model_grid
        zero_grid = self.model_grid - min(self.model_grid)
        wl_shift = np.polyval(np.flip(shift_coefs), zero_grid)
        shift_model_grid = np.add(self.model_grid, wl_shift)

        # Interpolate onto the measurement grid
        interp_spec = griddata(shift_model_grid, spec, x)

        return interp_spec

    def make_ils(self, max_opd=1.8, fov=30, nper_wn=25, wn=1000,
                 apod_function='NB medium'):

        # Relabel variables
        L = max_opd

        # Total length of 1-sided interferometer is the resolution
        total_opd = nper_wn

        # Set available apodisation functions
        apod_functions = [
            'Boxcar', 'Uniform', 'Triangular', 'Blackman-Harris',
            'Happ-Genzel', 'Hamming', 'Lorenz', 'Gaussian',
            'NB weak', 'NB medium', 'NB strong', 'Cosine'
        ]
        apod_function = apod_function.lower()

        # ----- Build grid based on max OPD -----
        n = int(L * nper_wn)
        if n % 2 == 0:
            n = n + 1
        L_grid = np.linspace(0, L, n)
        n_tot = int(total_opd * nper_wn)
        filler = np.zeros(n_tot - n)

        # ----- Build apodization functions -----
        # Boxcar function (no apodization, sampling function should be perfect
        # sinc)
        if 'boxcar' in apod_function or 'uniform' in apod_function:
            apod = np.ones(n)
        # Triangular function
        elif 'triang' in apod_function:
            apod = 1 - np.abs(L_grid) / L
        # Norton-Beer functions (NB)
        elif 'nb' in apod_function or 'norton' in apod_function or 'beer' in apod_function:
            # 'NB weak'
            if 'weak' in apod_function:
                c = [0.348093, -0.087577, 0.703484, 0.0]
            # NB strong
            elif 'strong' in apod_function:
                c = [0.045335, 0.0, 0.554883, 0.399782]
            # NB medium (default if nothing is specified)
            else:
                c = [0.152442, -0.136176, 0.983734, 0]
            # # Now build
            apod = np.zeros(n)
            for i in range(4):
                apod = apod + c[i] * (1 - (L_grid / L) ** 2) ** i

        # Hamming function (also known as Happ-Genzel function)
        elif 'hamming' in apod_function or 'happ' in apod_function or 'genzel' in apod_function:
            apod = 0.54 + 0.46 * np.cos(np.pi * L_grid / L)
        # Blackman-Harris function
        elif 'blackman' in apod_function or 'harris' in apod_function:
            apod = 0.42 + 0.5 * np.cos(np.pi * L_grid / L) + 0.08 * np.cos(2 * np.pi * L_grid / L)
        # Cosine function
        elif 'cos' in apod_function:
            apod = np.cos(np.pi * L_grid / (2 * L))
        # Lorenz function
        elif 'lorenz' in apod_function:
            apod = np.exp(-np.abs(L_grid) / L)
        elif 'gauss' in apod_function:
            apod = np.exp(-(2.24 * L_grid / L) ** 2)

        else:
            ValueError(
                'Invalid keyword for "apod_function". '
                'Please choose from (case insensitive) % s' % apod_functions
            )

        # ----- Take Fourier Transform and get sampling kernel -----
        # Add zeros to simulate infinite length and take FT
        kernel = np.fft.fft(np.concatenate((apod, filler))) .real
        middle = n_tot // 2

        # Split in the middle and mirror around wn_grid
        kernel = np.concatenate((kernel[middle:], kernel[0:middle]))
        offset = kernel[0:middle // 2].mean(axis=0)
        kernel = kernel - offset        # Remove offset in spectrum

        # Create the wavenumber grid centred on zero
        wn_grid = np.linspace(-total_opd / 2, total_opd / 2, n_tot)

        # Build boxcar filter for FOV effect
        fov_width = wn * (fov/1000/2) ** 2 / 2  # in [cm^-1]
        n_box = int(np.ceil(fov_width * nper_wn))
        if n_box < 0 or n_box > middle:
            n_box = 5
        boxcar = np.ones(n_box) * 1 / n_box
        kernel = np.convolve(kernel, boxcar, 'same')

        # Apply shift due to FOV effect
        shift = fov_width / 2
        # shift = 0.01
        shift_wn_grid = wn_grid - shift
        kernel = np.interp(wn_grid, shift_wn_grid, kernel)
        kernel = kernel / kernel.sum(axis=0)

        # Only keep the first 5 limbs
        idx = np.abs(wn_grid) <= 5 / L
        wn_grid = wn_grid[idx]
        kernel = kernel[idx]

        return kernel


def molecm2_to_ppmm(value, temperature, pressure):
    """Convert value in molecules/cm2 to ppmm using temperature and pressure."""
    return value * temperature / (7.243e14 * pressure)


def zero_fill(spectrum, zero_fill_factor):
    """Zero-fill the provided spectrum to increase the sampling."""
    # Unpack the x and  data from the spectrum
    grid = spectrum.coords['wavenumber'].to_numpy()
    spec = spectrum.to_numpy()

    # Calculate fast_npts, the first 2^x after npts
    target_npts = fft.next_fast_len(len(spec)) * zero_fill_factor

    # FFT the spectrum to space domain
    space_spec = fft.rfft(spec)

    # Re-FFT the space spectrum back to frequency space, padding with zeros
    filled_spec = fft.irfft(space_spec, target_npts)

    # Compute the new grid
    filled_grid = np.linspace(grid[0], grid[-1], target_npts)

    # Normalise to original max value
    norm_factor = max(spec) / max(filled_spec[1000:-1000])
    norm_filled_spec = filled_spec * norm_factor

    # Form the output DataArray
    filled_spectrum = xr.DataArray(
        data=norm_filled_spec,
        coords={'wavenumber': filled_grid},
        attrs=spectrum.attrs
    )

    return filled_spectrum


class FitResult(object):
    """Contains the results of a fit by the Analyser.

    Parameters
    ----------
    analyser : Analyser
        The analyser used to fit the spectrum
    spectrum : xarray.DataArray
        The spectrum fitted by the Analyser
    popt : numpy array
        The optimised parameters
    pcov : numpy array
        The covariance array
    perr : numpy array
        The error on the optimised parameters
    nerr : int
        The error flag. 0 = no error, 1 = fit failed.
    residual_limit : float
        The maximum fit residual to allow a "good" fit

    Methods
    -------
    None

    Attributes
    ----------
    params : Parameters
        The Parameters used in the fit, with the fit_val and fit_err values
        populated
    grid : numpy array
        The measurement grid in the fit window
    spec : numpy array
        The measurement spectrum in the fit window
    meas_od : dict
        Contains, for each gas, the optical depth difference between the
        measured spectrum and the fitted spectrum without the contribution from
        that gas. Useful for determining fit quality
    fit_od : dict
        Contains, for each gas, the fitted optical depth spectrum.
    """

    def __init__(self, analyser, spectrum, popt, pcov, perr, nerr,
                 residual_limit):
        """Initialise the FItResult."""

        self.analyser = analyser
        self.params = analyser.params

        coords = spectrum.coords

        data_vars = {'spectrum': spectrum}
        self.popt = popt
        self.pcov = pcov
        self.perr = perr
        self.nerr = nerr

        # Get dictionary of fitted parameters
        popt_dict = {
            key: popt[i]
            for i, key in enumerate(self.params.get_free_values_dict())
        }
        perr_dict = {
            key: perr[i]
            for i, key in enumerate(self.params.get_free_values_dict())
        }

        # Add the fit results to each parameter, starting with layers
        for layer in self.params.layers.values():

            # Apply temperature
            if layer.temperature.vary:
                key = f'{layer.layer_id}_temperature'
                if key in popt_dict:
                    layer.temperature.fit_value = popt_dict[key]
                    layer.temperature.fit_error = perr_dict[key]

            # Apply gas parameters
            for gas in layer.gases:
                key = f'{layer.layer_id}_{gas}'
                if key in popt_dict:
                    layer.gases[gas].fit_value = popt_dict[key]
                    layer.gases[gas].fit_error = perr_dict[key]

        # And then for other parameters
        for key in self.params.variables:
            if key in popt_dict:
                self.params.variables[key].fit_value = popt_dict[key]
                self.params.variables[key].fit_error = perr_dict[key]

        # If the fit is successful then calculate the fit and residual
        if not self.nerr:

            # Generate the fit spectrum
            fit = analyser.fwd_model(spectrum.wavenumber, *self.popt)
            data_vars['fit'] = xr.DataArray(data=fit, coords=coords)

            # Calculate the residual
            resid = (spectrum - fit)/spectrum * 100
            data_vars['residual'] = xr.DataArray(data=resid, coords=coords)

            # Calculate max residual value
            fit_qual_flag = np.nanmax(
                np.abs((spectrum - fit) / max(spectrum) * 100)
            ) > residual_limit

            # Check the fit quality
            if residual_limit is not None and fit_qual_flag:
                logger.info('High residual detected')
                self.nerr = 2

            # Calculate the background polynomial
            bg_poly = np.polyval(
                np.flip([
                    p.fit_value for p in self.params.variables.values()
                    if 'bg_poly' in p.name
                ]),
                spectrum.wavenumber
            )
            data_vars['bg_poly'] = xr.DataArray(data=bg_poly, coords=coords)

            # Calculate the intensity offset
            offset_polyvars = np.flip([
                p.fit_value for p in self.params.variables.values()
                if 'offset' in p.name
            ])
            if len(offset_polyvars) > 0:
                offset = np.polyval(offset_polyvars, spectrum.wavenumber)
            else:
                offset = np.full(len(spectrum), np.nan)
            data_vars['offset'] = xr.DataArray(data=offset, coords=coords)

            # Calculate optical depth spectra
            for key, layer in self.params.layers.items():
                for gas in layer.gases.values():
                    meas_od, fit_od = self._calc_od(spectrum, gas, layer)

                    data_vars[f'{key}_{gas}_measured_od'] = xr.DataArray(
                        data=meas_od, coords=coords
                    )
                    data_vars[f'{key}_{gas}_fitted_od'] = xr.DataArray(
                        data=fit_od, coords=coords,
                        attrs={'value': gas.fit_value}
                    )

        # If not then return nans
        else:
            logger.warn('Fit failed!')
            nan_array = xr.DataArray(
                data=np.full(len(spectrum), np.nan), coords=coords
            )
            data_vars = {
                **data_vars,
                'fit': nan_array,
                'residual': nan_array,
                'bg_poly': nan_array,
                'offset': nan_array
            }
            for key, layer in self.params.layers.items():
                for gas in layer.gases.values():
                    data_vars[f'{key}_{gas}_measured_od'] = xr.DataArray(
                        data=nan_array, coords=coords
                    )
                    data_vars[f'{key}_{gas}_fitted_od'] = xr.DataArray(
                        data=nan_array, coords=coords,
                        attrs={'value': gas.fit_value}
                    )

        self.data = xr.Dataset(data_vars=data_vars)

    def _calc_od(self, spectrum, gas, layer):
        """Calculate the given gas optical depth spectrum."""
        # Get a dictionary of the fit results
        popt_dict = self.params.get_popt_dict()

        # Set the parameter and any offset coefficients to zero
        par_name = f'{layer.layer_id}_{gas.name}'
        popt_dict[par_name] = 0
        for key in popt_dict.keys():
            if 'offset' in key:
                popt_dict[key] = 0

        # Get a list of param values
        popt_list = [val for val in popt_dict.values()]

        # Calculate the fit without the parameter
        fit = self.analyser.fwd_model(spectrum.wavenumber, *popt_list)

        # Calculate the shifted model grid
        shift_coefs = [popt_dict[n] for n in popt_dict if 'shift' in n]
        zero_grid = self.analyser.model_grid - min(self.analyser.model_grid)
        wl_shift = np.polyval(np.flip(shift_coefs), zero_grid)
        shift_model_grid = np.add(self.analyser.model_grid, wl_shift)

        # Calculate the intensity offset
        offset_coefs = [popt_dict[n] for n in popt_dict if 'offset' in n]
        offset = np.polyval(np.flip(offset_coefs), self.analyser.model_grid)
        offset = griddata(
            shift_model_grid, offset, spectrum.wavenumber, method='cubic'
        )

        # Get the layer temperature
        if layer.temperature.vary:
            temp = layer.temperature.fit_value
        else:
            temp = layer.temperature.value

        # Calculate the parameter od
        par_od = np.multiply(
            layer.get_cross_section(gas.name, temp),
            layer.gases[gas.name].fit_value
        )

        # Convolve with the ILS and interpolate onto the measurement grid
        par_T = np.exp(-par_od)
        ils_par_T = np.convolve(par_T, self.analyser.ils, mode='same')
        ils_par_od = -np.log(ils_par_T)
        par_od = griddata(shift_model_grid, ils_par_od, spectrum.wavenumber)

        meas_od = -np.log(np.divide(spectrum-offset, fit))
        fit_od = par_od

        return meas_od, fit_od
