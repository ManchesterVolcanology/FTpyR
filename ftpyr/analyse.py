import os
import logging
import numpy as np
import xarray as xr
from scipy import fft
from scipy.interpolate import griddata
from scipy.optimize import curve_fit, differential_evolution

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
    model_pts_per_cm : int, optional
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

    # Define apodisation parameters depending on the function
    apod_param_dict = {
        'boxcar':     [1.0,       0.0,      0.0,      0.0],
        'NB_weak':    [0.348093, -0.087577, 0.703484, 0.0],
        'NB_medium':  [0.152442, -0.136176, 0.983734, 0.0],
        'NB_strong':  [0.045335,  0.0,      0.554883, 0.399782]
    }

    def __init__(self, params, rfm_path, hitran_path, wn_start, wn_stop,
                 solar_flag=False, obs_height=0.0, update_params=True,
                 residual_limit=10, zero_fill_factor=0, model_padding=50,
                 model_pts_per_cm=100, apod_function='NB_medium', outfile=None,
                 tolerance=1e-8, output_ppmm_flag=False,
                 gas_auto_apriori=True):
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
            model_pts_per_cm=model_pts_per_cm
        )

        # Calculate the optical depths
        self.params = self.rfm.calc_optical_depths(params=params)

        # If using the gas apriori, update the parameters
        if gas_auto_apriori:
            for gas, par in self.params.items():
                if par.species is not None and par.vary:
                    self.params[gas].value = par.original_amt
                elif par.species is not None and not par.vary:
                    self.params[gas].value = 0

        # Pull the fitted parameters
        self.p0 = self.params.fittedvalueslist()

        # Store the fit window information
        self.wn_start = float(wn_start)
        self.wn_stop = float(wn_stop)
        self.model_pts_per_cm = int(model_pts_per_cm)

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
        self.xgrid_npts = self.model_pts_per_cm*(npts_cm) + 1
        self.model_grid = np.linspace(
            self.wn_start-model_padding,
            self.wn_stop+model_padding,
            self.xgrid_npts
        )

        # Generate the ILS
        logger.info('Generating initial ILS')
        try:
            self.apod_function = apod_function
            self.make_ils_jf(
                max_opd=params['opd'].value,
                fov=params['fov'].value,
                nper_wn=self.model_pts_per_cm,
                wn=(self.model_grid.max() - self.model_grid.min()) / 2,
                apod_function=self.apod_function
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
                    f'#,FTpyR Output file: {outfile}\n'
                    f'#,StartWavenumber(cm-1),{wn_start}\n'
                    f'#,StopWavenumber(cm-1),{wn_stop}\n'
                    f'#,WavenumberPadding(cm-1),{model_padding}\n'
                    f'#,PointsPercm,{model_pts_per_cm}\n'
                    f'#,ZeroFillFactor,{zero_fill_factor}\n'
                    f'#,SolarFlag,{solar_flag}\n'
                    f'#,ObserverHeight(m),{obs_height}\n'
                    f'#,Apodisation,{apod_function}\n'
                    f'#,Tolerance,{tolerance}\n'
                    f'#,Apodisation,{apod_function}\n'
                    f'#,RFM,{rfm_path}\n'
                    f'#,HITRAN,{hitran_path}\n'
                    f'#,GasOutputUnits,{gas_units}\n'
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
            spectrum = zero_fill(spectrum, self.zero_fill_factor)

        # Extract the region we are interested in
        full_xgrid = spectrum.coords['Wavenumber'].to_numpy()
        idx = np.logical_and(
            full_xgrid >= self.wn_start,
            full_xgrid <= self.wn_stop
        )
        self.grid = full_xgrid[idx != 0]
        self.spec = spectrum.to_numpy()[idx != 0]

        # Pull the fit bounds
        bounds = self.params.get_bounds()

        # Perform the fit
        try:

            # p0 = differential_evolution(
            #     self.fwd_model,
            #     bounds=np.transpose(bounds)
            # ).x

            # print(p0)

            # Run fit and calculate parameter error
            popt, pcov = curve_fit(
                self.fwd_model,
                self.grid,
                self.spec,
                self.p0,
                bounds=bounds,
                # xtol=self.tolerance,
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
        fit = FitResult(self, [self.grid, self.spec], popt, pcov, perr, nerr,
                        self.residual_limit, calc_od)

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

                    # Check if gases require conversion to ppm.m
                    if p.species is not None and self.output_ppmm_flag:
                        val = p.fit_val_to_ppmm()
                        err = p.fit_err_to_ppmm()
                    else:
                        val = p.fit_val
                        err = p.fit_err

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
            [par.original_od * p[par.name] for par in self.params.values()
             if par.species is not None]
        )

        # Convert to transmission
        trans_arr = np.exp(-od_arr)

        # Multiply all transmittances
        total_trans = np.prod(trans_arr, axis=0)

        # Multiply by the background
        raw_spec = np.multiply(bg_poly, total_trans)

        # Add the offset
        offset = np.polyval(np.flip(offset_coefs), self.model_grid)
        raw_spec = np.add(raw_spec, offset)

        # Generate the ILS is any ILS parameters are being fit
        if self.params['opd'].vary or self.params['fov'].vary:
            self.make_ils_jf(
                max_opd=p['opd'],
                fov=p['fov'],
                nper_wn=self.model_pts_per_cm,
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

    def _make_ils_old(self, optical_path_diff, fov, apod_function='NB_medium'):
        """Generate the ILS to use in the fit."""
        # Define the total optical path difference as the sampling frequency
        total_opd = self.model_pts_per_cm

        # Convert fov from milliradians to radians
        fov = fov / 1000

        # Define the total number of points in the ftir igm
        total_igm_npts = int(total_opd * self.model_pts_per_cm)
        ftir_igm_npts = int(optical_path_diff * self.model_pts_per_cm)
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
        spc = spc - np.min(spc)

        # Apply the FOV affect, like a boxcar in freq space with
        # width = wavenumber * (1 - cos(fov / 2))
        fov_width = self.wn_start * (1 - np.cos(fov / 2))

        # Calculate the smoothing factor in number of points
        fov_smooth_factor = int(self.model_pts_per_cm * fov_width)

        # Catch bad smooth factors
        if fov_smooth_factor <= 0 or fov_smooth_factor > n2:
            fov_smooth_factor = 5

        # Smooth the spc
        w = np.ones(fov_smooth_factor)
        spc_fov = np.convolve(spc, w, mode='same')

        # Select the middle, +/- 5 wavenumbers and normalise
        k_size = 5 * self.model_pts_per_cm
        kernel = spc_fov[n2-k_size:n2+k_size]
        norm_kernel = kernel/np.nansum(kernel)

        self.ils = norm_kernel

        return norm_kernel

    def _make_ils(self, opd, fov, apod_function='NB_medium'):
        """."""
        # Make a the apodisation function grid 10 cm-1 wide
        npts = 20*self.model_pts_per_cm + 1
        grid = np.linspace(-10, 10, npts)

        # Pull the apodisation parameters
        c_params = self.apod_param_dict[apod_function]

        # Compute the apodisation function
        apod = np.sum(
            [c * ((1 - (grid / opd)**2))**i for i, c in enumerate(c_params)],
            axis=0
        )

        # Apply fourier transform to frequency space
        apod_func = fft.ifft(apod).real

        # Mirror around 0
        n = npts // 2
        apod_func = np.concatenate([apod_func[n:], apod_func[:n]])

        # Apply the FOV affect, like a boxcar in freq space with
        # width = wavenumber * (1 - cos(fov / 2))
        av_wn = (self.wn_stop + self.wn_start) / 2
        fov_width = av_wn * (1 - np.cos(fov/1000 / 2))

        # Compute the fov boxhat function
        fov_box = np.zeros(npts)
        fov_box[np.abs(grid) < fov_width] = 1

        ils = np.convolve(apod_func, fov_box, mode='same')

        # Normalise the ILS
        norm_ils = np.divide(ils, np.sum(ils))

        self.ils = norm_ils

        return norm_ils

    def _make_ils_new(self, opd, fov, apod_function='NB_medium'):
        """."""
        # Convert fov from milliradians to radians
        fov = fov / 1000

        # --------------- DEFINE STARTING PARAMETERS  ---------------------
        # Define the total OPD as [in cm] the sampling frequency
        n_per_wave = self.model_pts_per_cm
        total_opd = n_per_wave

        # Set the number of points in the interferograms (IGM)
        # Total IGM
        n_total_igm = int(total_opd * n_per_wave)

        # FTIR IGM
        n_ftir_igm = int(opd * n_per_wave)

        # Filler IGM (difference between the two)
        n_filler_igm = int(n_total_igm - n_ftir_igm)

        # Create the IGM grids (in cm)
        # total_igm_grid = np.linspace(0, total_opd, n_ftir_igm)
        ftir_igm_grid = np.linspace(0, opd, n_ftir_igm)

        # Create empty IGM arrays
        ftir_igm = np.zeros(n_ftir_igm)
        filler_igm = np.zeros(n_filler_igm)

        # ---------------------- GENERATE THE IGMS ------------------------
        # Get the apod parameters
        c = self.apod_param_dict[apod_function]

        # Now build
        for i in range(4):
            add_igm = c[i] * ((1 - (ftir_igm_grid / opd) ** 2)) ** i
            ftir_igm = ftir_igm + add_igm

        # Fill the rest of the signal with zeros
        igm = np.concatenate((ftir_igm, filler_igm))

        # ---------------------- RECONSTRUCT KERNEL ------------------------
        # Apply Fourier Transform to reconstruct spectrum
        spc = np.fft.fft(igm).real

        # Split the spectrum in the middle and mirror around axis
        # Find middle point
        middle = n_total_igm // 2

        # Beginning of the signal is the tapering edge
        # (right side of the kernel)
        spc_right = spc[0:middle]

        # End of the signal is the rising edge (left side of the kernel)
        spc_left = spc[middle:]

        # Reconstruct spectrum around the middle point
        spc = np.concatenate((spc_left, spc_right))

        # Remove offset in spectrum
        offset = spc[0:middle//2].mean(axis=0)
        spc = spc - offset

        # FOV effect is like a boxcar in freq space
        fov_width = self.wn_start * (1 - np.cos(fov/2))      # in [cm^-1]
        shift = self.wn_start * (1 - np.cos(fov/2)) * 0.5    # in [cm^-1]
        n_box = int(np.ceil(fov_width * n_per_wave))
        if n_box < 0 or n_box > middle:
            n_box = 5
        boxcar = np.ones(n_box) * 1 / n_box
        spc_fov = np.convolve(spc, boxcar, 'same')
        spc_fov = spc_fov / spc_fov.sum()

        # Create the wavenumber grid centred on zero
        wave = np.linspace(-total_opd / 2, total_opd / 2, n_total_igm)

        # select the middle +/- 5 wavenumbers
        # (minimum window size is then 10 cm^-1)
        start = int(middle - 5 * n_per_wave)
        stop = int(middle + 5 * n_per_wave)
        kernel = spc_fov[start:stop]
        kernel = kernel / kernel.sum(axis=0)
        grid = wave[start:stop]

        self.ils = kernel

        return kernel

    def make_ils_jf(self, max_opd=1.8, fov=30, nper_wn=25, wn=1000,
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
        shift_wn_grid = wn_grid - shift
        kernel = np.interp(wn_grid, shift_wn_grid, kernel)
        kernel = kernel / kernel.sum(axis=0)

        # Only keep the first 5 limbs
        idx = np.abs(wn_grid) <= 5 / L
        wn_grid = wn_grid[idx]
        kernel = kernel[idx]

        self.ils = kernel

        return kernel


def zero_fill(spectrum, zero_fill_factor):
    """Zero-fill the provided spectrum to increase the sampling."""
    # Unpack the x and  data from the spectrum
    grid = spectrum.coords['Wavenumber'].to_numpy()
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
    pcov : numpy array
        The covariance array
    perr : numpy array
        The error on the optimised parameters
    nerr : int
        The error flag. 0 = no error, 1 = fit failed.
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

    def __init__(self, analyser, spectrum, popt, pcov, perr, nerr,
                 residual_limit, calc_od):
        """Initialise the FItResult."""

        self.analyser = analyser
        self.params = analyser.params

        grid, spec = spectrum
        coords = {'wavenumber': grid}

        data_vars = {'spectrum': xr.DataArray(data=spec, coords=coords)}
        self.popt = popt
        self.pcov = pcov
        self.perr = perr
        self.nerr = nerr

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
            fit = analyser.fwd_model(grid, *self.popt)
            data_vars['fit'] = xr.DataArray(data=fit, coords=coords)

            # Calculate the residual
            resid = (spec - fit)/spec * 100
            data_vars['residual'] = xr.DataArray(data=resid, coords=coords)

            # Calculate max residual value
            fit_qual_flag = np.nanmax(
                np.abs((spec - fit) / max(spec) * 100)
            ) > residual_limit

            # Check the fit quality
            if residual_limit is not None and fit_qual_flag:
                logger.info('High residual detected')
                self.nerr = 2

            # Calculate the background polynomial
            bg_poly = np.polyval(
                np.flip(
                    [p.fit_val for p in self.params.values()
                     if 'bg_poly' in p.name]
                ),
                grid
            )
            data_vars['bg_poly'] = xr.DataArray(data=bg_poly, coords=coords)

            # Calculate the intensity offset
            offset_polyvars = np.flip(
                [p.fit_val for p in self.params.values() if 'offset' in p.name]
            )
            if len(offset_polyvars) > 0:
                offset = np.polyval(offset_polyvars, grid)
            else:
                offset = np.full(len(spec), np.nan)
            data_vars['offset'] = xr.DataArray(data=offset, coords=coords)

            # Calculate optical depth spectra
            if calc_od == 'all':
                calc_od = [
                    par.name for par in self.params.values()
                    if par.species is not None
                ]

            for par in calc_od:
                if par in self.params:
                    meas_od, fit_od = self._calc_od(grid, spec, par)
                    data_vars[f'{par}_measured_od'] = xr.DataArray(
                        data=meas_od, coords=coords
                    )
                    data_vars[f'{par}_fitted_od'] = xr.DataArray(
                        data=fit_od, coords=coords,
                        attrs={'value': self.params[par].fit_val}
                    )

        # If not then return nans
        else:
            logger.warn('Fit failed!')
            nan_array = xr.DataArray(
                data=np.full(len(self.spec), np.nan), coords=coords
            )
            data_vars = {
                **data_vars,
                'fit': nan_array,
                'residual': nan_array,
                'bg_poly': nan_array,
                'offset': nan_array
            }
            for par in calc_od:
                if par in self.params:
                    data_vars[f'{par}_measured_od'] = xr.DataArray(
                        data=nan_array, coords=coords
                    )
                    data_vars[f'{par}_fitted_od'] = xr.DataArray(
                        data=nan_array, coords=coords,
                        attrs={'value': self.params[par].fit_val}
                    )

        self.data = xr.Dataset(data_vars=data_vars)

    def _calc_od(self, grid, spec, par_name):
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
        fit = self.analyser.fwd_model(grid, *fit_params)

        # Calculate the shifted model grid
        shift_coefs = [p[n] for n in p if 'shift' in n]
        zero_grid = self.analyser.model_grid - min(self.analyser.model_grid)
        wl_shift = np.polyval(np.flip(shift_coefs), zero_grid)
        shift_model_grid = np.add(self.analyser.model_grid, wl_shift)

        # Calculate the intensity offset
        offset_coefs = [p[n] for n in p if 'offset' in n]
        offset = np.polyval(np.flip(offset_coefs), self.analyser.model_grid)
        offset = griddata(
            shift_model_grid, offset, grid, method='cubic'
        )

        # Calculate the parameter od
        par_od = np.multiply(
            self.params[par_name].xsec_od,
            self.params[par_name].fit_val
        )

        # Convolve with the ILS and interpolate onto the measurement grid
        par_T = np.exp(-par_od)
        ils_par_T = np.convolve(par_T, self.analyser.ils, mode='same')
        ils_par_od = -np.log(ils_par_T)
        par_od = griddata(shift_model_grid, ils_par_od, grid)

        meas_od = -np.log(np.divide(spec-offset, fit))
        fit_od = par_od

        return meas_od, fit_od
