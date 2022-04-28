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
    """."""

    def __init__(self, params, rfm_path, hitran_path, wn_start, wn_stop,
                 solar_flag=False, obs_height=0.0, rfm_id=None,
                 update_params=True, residual_limit=None, wn_pad=50,
                 npts_per_cm=100, apod_function='NB_medium', outfile=None,
                 tolerance=0.01):
        """."""
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
            rfm_id=rfm_id,
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
                ofile.write('\n')

    def fit(self, spectrum, calc_od=[], zero_fill_factor=0):
        """."""
        # Apply zero-filling
        if zero_fill_factor:
            spectrum = self._zero_fill(spectrum, zero_fill_factor)

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
                ofile.write('\n')

        return fit

    def fwd_model(self, x, *p0):
        """."""
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

        # See if offset is in the parameters
        if 'offset' in p.keys():
            offset = p['offset']
        else:
            offset = 0

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

        # Add the offset
        interp_offset_spec = np.add(interp_spec, offset)

        # Progress the iteration counter
        self.iter_count += 1

        return interp_offset_spec

    def _make_ils(self, optical_path_diff, fov, apod_function='NB_medium'):
        """."""
        # Define the total optical path difference as the sampling frequency
        total_opd = self.npts_per_cm

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

            for i in range(3):
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
        if fov_smooth_factor < 0 or fov_smooth_factor > n2:
            fov_smooth_factor = 5

        # Smooth the spc
        w = np.ones(fov_smooth_factor)
        spc_fov = np.convolve(spc, w, mode='same')

        # Select the middle, +/- 5 wavenumbers and normalise
        k_size = 5 * self.npts_per_cm
        kernel = spc_fov[n2-k_size:n2+k_size]
        norm_kernel = kernel/np.sum(kernel)

        self.ils = norm_kernel

        return norm_kernel

    def _zero_fill(self, spectrum, zero_fill_factor):
        """."""
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
    """."""

    def __init__(self, analyser, spectrum, popt, perr, nerr, iter_count,
                 residual_limit, calc_od):
        """."""

        self.analyser = analyser
        self.params = analyser.params
        self.grid, self.spec = spectrum
        self.popt = popt
        self.perr = perr
        self.nerr = nerr
        self.iter_count = iter_count
        self.meas_od = {}
        self.synth_od = {}

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
            for par in calc_od:
                if par in self.params:
                    self.calc_od(par)

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
                    self.synth_od[par] = np.full(len(self.spec), np.nan)

    def calc_od(self, par_name):
        """."""
        # Make a copy of the parameters to use in the OD calculation
        params = self.params.make_copy()

        # Set the parameter and any offset coefficients to zero
        params[par_name].fit_val = 0
        if 'offset' in params.keys():
            offset = params['offset'].fit_val
            params['offset'].fit_val = 0
        else:
            offset = 0

        # Calculate the fit without the parameter
        fit_params = params.popt_list()
        p = self.params.popt_dict()
        fit = self.analyser.fwd_model(self.grid, *fit_params)

        # Calculate the shifted model grid
        shift_coefs = [p[n] for n in p if 'shift' in n]
        zero_grid = self.analyser.model_grid - min(self.analyser.model_grid)
        wl_shift = np.polyval(np.flip(shift_coefs), zero_grid)
        shift_model_grid = np.add(self.analyser.model_grid, wl_shift)

        # Calculate the parameter od
        par_od = np.multiply(params[par_name].xsec_od, p[par_name])

        # Convolve with the ILS and interpolate onto the measurement grid
        par_od = griddata(shift_model_grid,
                          np.convolve(par_od, self.analyser.ils, mode='same'),
                          self.grid)

        # Add to self
        self.meas_od[par_name] = -np.log(np.divide(self.spec-offset, fit))
        self.synth_od[par_name] = par_od

        return self.meas_od[par_name], self.synth_od[par_name]
