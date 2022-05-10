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
    wn_pad : float, optional
        The amount of padding of the fit window, in cm-1, to avoid convolution
        edge effects and allow a wavenumber shift. Default is 50.
    zero_fill_factor : int, optional
        If greater than zero then the spectra are zero-filled before analysis
        to artificially increase the sampling frequency. Increasing numbers
        use increassing zero-filling, determined by the next_fast_len function
        of the scipy.fft library. Default is 0.
    npts_per_cm : int, optional
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
    bg_behaviour : str, optional
        How the background spectrum is handled, if it is provided. Must be
        either subtract or divide. Default is subtract.
    bg_spectrum : xarray.DataArray, optional
        The spectrum to use in the background correction. Default is None.

    Methods
    -------
    fit(spectrum, calc_od='all')
        Fit the provided spectrum

    fwd_model(x, *p0)
        The forward model used to fit spectra
    """

    def __init__(self, params, rfm_path, hitran_path, wn_start, wn_stop,
                 solar_flag=False, obs_height=0.0, update_params=True,
                 residual_limit=10, wn_pad=50, zero_fill_factor=0,
                 npts_per_cm=100, apod_function='NB_medium', outfile=None,
                 tolerance=0.001, bg_behaviour='subtract', bg_spectrum=None):
        """Initialise the Analyser."""
        # Generate the RFM object
        logger.debug('Setting up RFM')
        self.rfm = RFM(
            exe_path=rfm_path,
            hitran_path=hitran_path,
            wn_start=wn_start,
            wn_stop=wn_stop,
            wn_pad=wn_pad,
            solar_flag=solar_flag,
            obs_height=obs_height,
            npts_per_cm=npts_per_cm
        )

        # Calculate the optical depths
        self.params = self.rfm.calc_optical_depths(params=params)

        # Pull the fitted parameters
        self.p0 = self.params.fittedvalueslist()

        # Store the fit window information
        self.wn_start = float(wn_start)
        self.wn_stop = float(wn_stop)
        self.npts_per_cm = int(npts_per_cm)

        # Store the quality check settings
        self.update_params = update_params
        self.residual_limit = residual_limit
        self.tolerance = tolerance

        # Add zero fill factor
        self.zero_fill_factor = zero_fill_factor

        # Process the background spectrum
        self.bg_behaviour = bg_behaviour
        self.bg_spectrum = bg_spectrum

        # Apply zero-filling to the background if required
        if self.zero_fill_factor:
            self.bg_spectrum = self._zero_fill(
                self.bg_spectrum, self.zero_fill_factor)

        # Calculate the model x-grid
        # This includes a 1 cm-1 padding on either side to allow shifts
        npts_cm = int(self.wn_stop - self.wn_start) + wn_pad*2
        self.xgrid_npts = self.npts_per_cm*(npts_cm) + 1
        self.model_grid = np.linspace(
            self.wn_start-wn_pad,
            self.wn_stop+wn_pad,
            self.xgrid_npts
        )

        # Generate the ILS
        logger.info('Generating initial ILS')
        try:
            self.apod_function = apod_function
            self._make_ils(
                params['opd'].value,
                params['fov'].value,
                self.apod_function
            )
        except KeyError:
            logger.error('Ensure both "opd" and "fov" are defined in params!')

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
                    '#,FTpyR Output file\n'
                    f'#,StartWavenumber(cm-1),{wn_start}\n'
                    f'#,StopWavenumber(cm-1),{wn_stop}\n'
                    f'#,WavenumberPadding(cm-1),{wn_pad}\n'
                    f'#,SolarMeasurement,{solar_flag}\n'
                    f'#,ObserverHeight(m),{obs_height}\n'
                    f'#,Apodisation,{apod_function}\n'
                    '#,GasUnits,molecules.cm-2\n'
                    '#,Name,Gas,Temperature(K),Pressure(mb),PathLength(m)\n'
                )

                # Write gas details
                for p in params.values():
                    if p.species is not None:
                        ofile.write(
                            f'#,{p.name},{p.species},{p.temp},{p.pres},'
                            f'{p.path}\n'
                        )

                # Write the fit results header
                ofile.write('Filename,Timestamp')
                for p in params.values():
                    ofile.write(f',{p.name},{p.name}_err')
                ofile.write(',FitQuality,MaxResidual,StdevResidual\n')

    def fit(self, spectrum, calc_od='all'):
        """Fit the provided spectrum.

        Parameters
        ---------
        spectrum : xarray.DataArray
            The spectrum to fit. Must be a DataArray with a single coords named
            Wavenumber and with the following attrs:
                filename: the spectrum filename
                timestamp: the spectrum timestamp
        calc_od : str or list, optional
            The gas parameters for which to calculate the optical depths. If
            all, then all gases are calculated. Default is all.

        Returns
        -------
        FitResult object
            Holds the fit results and associated metadata
        """
        # Apply zero-filling
        if self.zero_fill_factor:
            spectrum = self._zero_fill(spectrum, self.zero_fill_factor)

        # If a background spectrum is given, perform the background correction
        if self.bg_spectrum is not None:
            if self.bg_behaviour == 'subtract':
                spectrum = xr.DataArray(
                    data=np.subtract(spectrum, self.bg_spectrum),
                    coords=spectrum.coords,
                    attrs=spectrum.attrs
                )
            elif self.bg_behaviour == 'divide':
                spectrum = xr.DataArray(
                    data=np.divide(spectrum, self.bg_spectrum),
                    coords=spectrum.coords,
                    attrs=spectrum.attrs
                )

        # Extract the region we are interested in
        full_xgrid = spectrum.coords['Wavenumber'].to_numpy()
        idx = np.logical_and(
            full_xgrid >= self.wn_start,
            full_xgrid <= self.wn_stop
        )
        self.grid = full_xgrid[idx != 0]
        self.spec = spectrum.to_numpy()[idx != 0]

        # Perform the fit
        try:
            # Reset iteration counter
            self.iter_count = 0

            # Run fit and calculate parameter error
            popt, pcov = curve_fit(
                self.fwd_model,
                self.grid,
                self.spec,
                self.p0,
                xtol=self.tolerance,
                ftol=self.tolerance
            )
            perr = np.sqrt(np.diag(pcov))

            # Set error code
            nerr = 0

        except RuntimeError:
            popt = np.full(len(self.p0), np.nan)
            perr = np.full(len(self.p0), np.nan)
            nerr = 1

        # Put the results into a FitResult object
        fit = FitResult(self, [self.grid, self.spec], popt, perr, nerr,
                        self.iter_count, self.residual_limit, calc_od)

        # Update the initial fit parameters
        if self.update_params and not fit.nerr:
            self.p0 = popt
        else:
            self.p0 = self.params.fittedvalueslist()

        # Write fit results
        if self.outfile is not None:
            with open(self.outfile, 'a') as ofile:

                # Write the filename and timestamp
                metadata = spectrum.attrs
                ofile.write(f'{metadata["filename"]},{metadata["timestamp"]}')

                # Write the fitted values
                for p in self.params.values():
                    ofile.write(f',{p.fit_val},{p.fit_err}')

                # New line
                ofile.write(
                    f',{fit.nerr},{fit.max_residual},{fit.std_residual}\n'
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
        # Get dicitionary of fitted parameters
        p = self.params.valuesdict()

        # Update the fitted parameter values with those supplied to the forward
        # model
        i = 0
        for par in self.params.values():
            if par.vary:
                p[par.name] = p0[i]
                i += 1
            else:
                p[par.name] = par.value

        # Unpack polynomial parameters
        bg_poly_coefs = [p[n] for n in p if 'bg_poly' in n]
        shift_coefs = [p[n] for n in p if 'shift' in n]
        offset_coefs = [p[n] for n in p if 'offset' in n]

        # Construct background polynomial
        bg_poly = np.polyval(np.flip(bg_poly_coefs), self.model_grid)

        # Calculate the gas optical depths
        od_arr = np.asarray(
            [par.xsec_od * p[par.name] for par in self.params.values()
             if par.species is not None]
        )

        # Convert to transmission
        trans_arr = np.exp(-od_arr)

        # Mulitply all transmittances
        total_trans = np.prod(trans_arr, axis=0)

        # Multiply by the background
        raw_spec = np.multiply(bg_poly, total_trans)

        # Add the offset
        offset = np.polyval(np.flip(offset_coefs), self.model_grid)
        raw_spec = np.add(raw_spec, offset)

        # Generate the ILS is any ILS parameters are being fit
        if self.params['opd'].vary or self.params['fov'].vary:
            self._make_ils(abs(p['opd']), abs(p['fov']))

        # Convole with the ILS
        spec = np.convolve(raw_spec, self.ils, mode='same')

        # Apply shift and stretch to the model_grid
        zero_grid = self.model_grid - min(self.model_grid)
        wl_shift = np.polyval(np.flip(shift_coefs), zero_grid)
        shift_model_grid = np.add(self.model_grid, wl_shift)

        # Interpolate onto the measurement grid
        interp_spec = griddata(shift_model_grid, spec, x)

        # Progress the iteration counter
        self.iter_count += 1

        return interp_spec

    def _make_ils(self, optical_path_diff, fov, apod_function='NB_medium'):
        """Generate the ILS to use in the fit."""
        # Define the total optical path difference as the sampling frequency
        total_opd = self.npts_per_cm

        # Convert fov from milliradians to radians
        fov = fov / 1000

        # Define the total number of points in the ftir igm
        total_igm_npts = int(total_opd * self.npts_per_cm)
        ftir_igm_npts = int(optical_path_diff * self.npts_per_cm)
        filler_igm_npts = int(total_igm_npts - ftir_igm_npts)

        # Calculate the FTIR IGM grid
        ftir_igm_grid = optical_path_diff * \
            np.arange(ftir_igm_npts) / (ftir_igm_npts - 1)

        # Initialise interferograms
        ftir_igm = np.zeros(ftir_igm_npts)
        filler_igm = np.zeros(filler_igm_npts)

        # If using a triangular apod function, then generate
        if apod_function == 'triangular':
            ftir_igm = np.flip(np.arange(total_igm_npts) / (total_igm_npts-1))
            igm = np.concatenate(ftir_igm, filler_igm)

        # Otherwise compute the boxcar/NB function
        elif apod_function in ['boxcar', 'NB_weak', 'NB_medium', 'NB_strong']:

            # Define apodisation parameters depending on the function
            apod_param_dict = {
                'boxcar':     [1.0,       0.0,      0.0,      0.0],
                'NB_weak':    [0.348093, -0.087577, 0.703484, 0.0],
                'NB_medium':  [0.152442, -0.136176, 0.983734, 0.0],
                'NB_strong':  [0.045335,  0.0,      0.554883, 0.399782]
            }

            c = apod_param_dict[apod_function]

            for i in range(4):
                ftir_igm += c[i] * (
                    (1 - (ftir_igm_grid / optical_path_diff)**2)**i
                )
            igm = np.concatenate([ftir_igm, filler_igm])

        else:
            logger.error('Apodisation function not recognised!')
            raise ValueError

        # FFT the interferogram to length space
        spc = fft.ifft(igm).real

        n2 = int(total_igm_npts / 2)
        spc1 = spc[:n2]
        spc2 = spc[n2:]
        spc = np.concatenate([spc2, spc1])

        # Remove the offset
        spc = spc - np.mean(spc)

        # Apply the FOV affect, like a boxcar in freq space with
        # width = wavenumber * (1 - cos(fov / 2))
        fov_width = self.wn_start * (1 - np.cos(fov / 2))

        # Calculate the smoothing factor in number of points
        fov_smooth_factor = int(self.npts_per_cm * fov_width)

        # Catch bad smooth factors
        if fov_smooth_factor <= 0 or fov_smooth_factor > n2:
            fov_smooth_factor = 5

        # Smooth the spc
        w = np.ones(fov_smooth_factor)
        spc_fov = np.convolve(spc, w, mode='same')

        # Select the middle, +/- 5 wavenumbers and normalise
        k_size = 5 * self.npts_per_cm
        kernel = spc_fov[n2-k_size:n2+k_size]
        norm_kernel = kernel/np.nansum(kernel)

        self.ils = norm_kernel

        return norm_kernel

    def _zero_fill(self, spectrum, zero_fill_factor):
        """Zero-fill the provided spectrum to increase the sampling."""
        # Unpack the x and  data from the spectrum
        grid = spectrum.coords['Wavenumber'].to_numpy()
        spec = spectrum.to_numpy()

        # Get npts
        orig_npts = len(grid)

        # calculate fast_npts the first 2^x after npts
        fast_npts = fft.next_fast_len(len(spec))

        # Add on the required number of zeros to the spectrum to get
        # fast_npts; make it go to zero at the end
        extra_npts = fast_npts - orig_npts
        extra = spec[orig_npts-1]*np.arange(extra_npts)/(extra_npts-1)
        extra = np.flip(extra)
        spec_extra = np.concatenate([spec, extra])

        # FFT the spectrum to frequency space
        fft_spec = fft.fft(spec_extra)

        # Rearrange it
        n = int(fast_npts/2)
        fft_spec = np.concatenate([fft_spec[n:], fft_spec[:n]])

        # Multiply fast_npts by zero_fill_factor to obtain required zero-filled
        # array
        req_npts = fast_npts * zero_fill_factor
        added_npts = req_npts - fast_npts
        added_npts_left = int(added_npts/2)
        added_npts_right = added_npts - added_npts_left
        extra_left = np.linspace(0, fft_spec[0], added_npts_left)
        extra_right = np.linspace(0, fft_spec[-1], added_npts_right)

        # Determine new, zerofilled interferogram
        fft_filled_spec = np.concatenate([extra_left, fft_spec, extra_right])

        # Back to length space
        filled_spec = abs(fft.ifft(fft_filled_spec))
        filled_grid = np.linspace(min(grid), max(grid), req_npts)

        # normalise to original max value
        norm_factor = max(spec) / max(filled_spec[1000:-1000])
        norm_filled_spec = filled_spec * norm_factor

        # Form the output DataArray
        filled_spectrum = xr.DataArray(
            data=norm_filled_spec,
            coords={'Wavenumber': filled_grid},
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
    perr : numpy array
        The error on the optimised parameters
    nerr : int
        The error flag. 0 = no error, 1 = fit failed.
    iter_count : int
        The number of iterations taken to converge.
    residual_limit : float
        The maximum fit residual to allow a "good" fit
    calc_od : str or list
        The gas parameters for which to calculate the optical depths. If all,
        then all gases are calculated.

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

    def __init__(self, analyser, spectrum, popt, perr, nerr, iter_count,
                 residual_limit, calc_od):
        """Initialise the FItResult."""

        self.analyser = analyser
        self.params = analyser.params
        self.grid, self.spec = spectrum
        self.popt = popt
        self.perr = perr
        self.nerr = nerr
        self.iter_count = iter_count
        self.meas_od = {}
        self.fit_od = {}

        # Add the fit results to each parameter
        i = 0
        for par in self.params.values():

            if par.vary:
                par.fit_val = self.popt[i]
                par.fit_err = self.perr[i]
                i += 1
            else:
                par.fit_val = par.value
                par.fit_err = 0

        # If the fit is successful then calculate the fit and residual
        if not self.nerr:

            # Generate the fit spectrum
            self.fit = analyser.fwd_model(self.grid, *self.popt)

            # Calculate the residual
            self.residual = (self.spec - self.fit)/self.spec * 100

            # Calculate residual values
            self.max_residual = np.nanmax(np.abs(self.residual))
            self.std_residual = np.nanstd(self.residual)

            # Check the fit quality
            if residual_limit is not None \
                    and max(abs(self.residual)) > residual_limit:
                logger.info('High residual detected')
                self.nerr = 2

            # Calculate the background polynomial
            self.bg_poly = np.polyval(
                np.flip(
                    [p.fit_val for p in self.params.values()
                     if 'bg_poly' in p.name]
                ),
                self.grid
            )

            # Calculate optical depth spectra
            if calc_od == 'all':
                calc_od = [par.name for par in self.params.values()
                           if par.species is not None]
            for par in calc_od:
                if par in self.params:
                    self._calc_od(par)

        # If not then return nans
        else:
            logger.warn('Fit failed!')
            self.fit = np.full(len(self.spec), np.nan)
            self.residual = np.full(len(self.spec), np.nan)
            self.max_residual = np.nan
            self.std_residual = np.nan
            self.bg_poly = np.full(len(self.spec), np.nan)
            for par in calc_od:
                if par in self.params:
                    self.meas_od[par] = np.full(len(self.spec), np.nan)
                    self.fit_od[par] = np.full(len(self.spec), np.nan)

    def _calc_od(self, par_name):
        """Calculate the given gas optical depth spectrum."""
        # Make a copy of the parameters to use in the OD calculation
        params = self.params.make_copy()

        # Set the parameter and any offset coefficients to zero
        params[par_name].fit_val = 0
        for par in params:
            if 'offset' in par:
                params[par].fit_val = 0

        # Calculate the fit without the parameter
        fit_params = params.popt_list()
        p = self.params.popt_dict()
        fit = self.analyser.fwd_model(self.grid, *fit_params)

        # Calculate the shifted model grid
        shift_coefs = [p[n] for n in p if 'shift' in n]
        zero_grid = self.analyser.model_grid - min(self.analyser.model_grid)
        wl_shift = np.polyval(np.flip(shift_coefs), zero_grid)
        shift_model_grid = np.add(self.analyser.model_grid, wl_shift)

        # Calculate the intensity offset
        offset_coefs = [p[n] for n in p if 'offset' in n]
        offset = np.polyval(offset_coefs, self.analyser.model_grid)
        offset = griddata(shift_model_grid,
                          offset,
                          self.grid,
                          method='cubic')

        # Calculate the parameter od
        par_od = np.multiply(params[par_name].xsec_od, p[par_name])

        # Convolve with the ILS and interpolate onto the measurement grid
        par_od = griddata(shift_model_grid,
                          np.convolve(par_od, self.analyser.ils, mode='same'),
                          self.grid)

        # Add to self
        self.meas_od[par_name] = -np.log(np.divide(self.spec-offset, fit))
        self.fit_od[par_name] = par_od

        return self.meas_od[par_name], self.fit_od[par_name]
