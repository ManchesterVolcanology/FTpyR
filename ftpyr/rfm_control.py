import os
import yaml
import logging
import numpy as np
import xarray as xr
from tqdm import tqdm
from subprocess import Popen, PIPE, CalledProcessError

logger = logging.getLogger(__name__)


class RFM(object):
    """Wrapper to call the Reference Forward Model (RFM).

    Uses a precompiled RFM executable and the HITRAN line database to calculate
    requested gas optical depth spectra in order to fit measured OP-FTIR
    spectra. Gas optical depths are calculated one by one for both the volcanic
    plume and atmosphere layers.

    More details on the RFM can be found in doi:10.1016/j.jqsrt.2016.06.018 and
    online at http://eodg.atm.ox.ac.uk/RFM/

    Note that this has been designed to run using RFM v5.12.

    Parameters
    ----------
    exe_path : str
        The path to the RFM executable
    hitran_path : str
        The path to the HITRAN line database, relative to the RFM working
        directory
    wn_start : float
        The start wavenumber of the fit window in cm-1
    wn_stop : float
        The end wavenumber of the fit window in cm-1
    model_padding : float, optional
        The padding of the fit window in cm-1. Default is 50cm-1
    solar_flag : bool, optional
        If True, then calculation is perfromed using a full atmosphere.
        Requires obs_height to be defined. Default is False.
    obs_height : float, optional
        The altitude of the instrument in meters above sea level, used if
        solar_flag is True. Default is None
    pts_per_cm : int, optional
        The number of points per cm-1 in the forward model. Default is 100

    Methods
    -------
    calc_optical_depths(parmas)
        Calculate the gas optical depths using the RFM executable.
    """

    def __init__(self, exe_path, hitran_path, wn_start, wn_stop,
                 model_padding=50, solar_flag=False, obs_height=0.0,
                 pts_per_cm=25, vmr_file='databases/atm_layer.yml',
                 progress_bars=True):
        """."""
        # Assign object variables =============================================
        self.exe_path = str(exe_path)
        self.wd = str(os.path.split(self.exe_path)[0])
        self.hitran_path = str(hitran_path)
        self.wn_start = float(wn_start)
        self.wn_stop = float(wn_stop)
        self.model_padding = float(model_padding)
        self.solar_flag = bool(solar_flag)
        self.pts_per_cm = int(pts_per_cm)
        self.obs_height = float(obs_height)
        self.progress_bars = progress_bars

        # Perform input checks ================================================
        # Check the RFM working directory isn't the current one
        if self.wd == '':
            self.wd = '.'

        # Make sure the cache folder exists
        self.cache_folder = f'{self.wd}/cache'
        if not os.path.isdir(self.cache_folder):
            os.makedirs(self.cache_folder)

        # Check the the start wavenumber is smaller than the end
        if self.wn_start >= self.wn_stop:
            logger.error(
                'Start wavenumber must be lower than stop wavenumber!'
            )

        # Read in the volume mixing ratios
        with open(vmr_file, 'r') as ymlfile:
            self.vmr = yaml.load(ymlfile, Loader=yaml.FullLoader)

    def calc_optical_depths(self, params):
        """Calculate gas optical depths using RFM.

        Parameters
        ----------
        params : parameters.Parameters object
            Contains the fit parameters used in the retrieval

        Returns
        -------
        Parameters object
            A copy of the input parameters, with the original_od and
            original_amt variables populated from the RFM run
        """

        # Iterate through the desired gases
        logger.info('Calculating gas optical depths')

        # Iterate layer by layer
        for layer in params.layers.values():

            # Check if fitting temperature or not
            if layer.temperature.vary:

                # Set the temperatures to model
                temperatures = np.arange(
                    layer.temperature.bounds[0],
                    layer.temperature.bounds[1] + layer.temperature_step,
                    layer.temperature_step
                )

                # Form an id string to use in the file name
                temp_str = str(
                    f'{layer.temperature.bounds[0]}-'
                    f'{layer.temperature.bounds[1]}-'
                    f'{layer.temperature_step}'
                )

            else:
                temperatures = np.array([layer.temperature.value])
                temp_str = str(layer.temperature.value)

            # Create gas progress bar
            if self.progress_bars:
                gas_pbar = tqdm(
                    total=len(layer.gases), desc=layer.layer_id, leave=False
                )

            # Iterate through gases in the layer
            for gas in layer.gases:

                # Pull gas mixing ratio
                gas_vmr = self.vmr[gas]

                # Form cache filename
                wn_lo_limit = self.wn_start - self.model_padding
                wn_hi_limit = self.wn_stop + self.model_padding
                fname = str(
                    f'{gas}_'
                    f'{wn_lo_limit}_{wn_hi_limit}_'
                    f'{layer.pressure}_{temp_str}_{layer.path_length}_'
                    f'{self.pts_per_cm}_{gas_vmr:.3E}'
                ).replace('.', 'p') + '.nc'

                # Generate a flag to control if the calculation is run or not
                calc_od_flag = True

                # If the cache file exists, read it in. Otherwise run RFM
                cache_fname = f'{self.cache_folder}/{fname}'
                if os.path.isfile(cache_fname):
                    logger.info(f'Cache file found for {gas}')
                    with xr.open_dataset(cache_fname) as cross_section:

                        # Check that the precise temperature, pressure and
                        # window match the requested ones
                        attrs = cross_section.attrs
                        checks = np.array([
                            gas == attrs['species'],
                            np.all(
                                temperatures == cross_section.temperature.data
                            ),
                            layer.pressure == attrs['pressure'],
                            layer.path_length == attrs['path_length'],
                            self.pts_per_cm == attrs['pts_per_cm'],
                            wn_lo_limit == attrs['wn_lo_limit'],
                            wn_hi_limit == attrs['wn_hi_limit'],
                            gas_vmr == attrs['initial_vmr']
                        ])

                    if checks.all():
                        calc_od_flag = False
                        cross_sections = xr.load_dataset(cache_fname)

                    else:
                        logger.info(
                            'Consistency check failed, '
                            're-calculating optical depth'
                        )

                if calc_od_flag:

                    # Create progress bar
                    if self.progress_bars and layer.temperature.vary:
                        temp_pbar = tqdm(
                            total=len(temperatures), desc=gas, leave=False
                        )

                    # Cycle through temperatures to model
                    for i, temp in enumerate(temperatures):

                        # Write the atmosphere profile and driver table for this
                        # gas
                        self._write_profile(
                            layer.pressure, temp, layer.temperature.value
                        )
                        self._write_driver_table(gas, layer)

                        # Scale the vmr so that the path amount is constant wrt
                        # the a priori temperature
                        scaled_vmr = gas_vmr * layer.temperature.value / temp

                        if not self.progress_bars:
                            logger.info(
                                f'Running RFM for {gas} '
                                f'at {temp} K, '
                                f'{layer.pressure} mb, '
                                f'path length {layer.path_length} m '
                                f'and VMR {scaled_vmr:.3E} ppm'
                            )

                        # Execute RFM
                        kwargs = dict(
                            stdout=PIPE, bufsize=1, universal_newlines=True,
                            cwd=self.wd
                        )
                        with Popen(self.exe_path, **kwargs) as proc:
                            for line in proc.stdout:
                                logger.debug(line)

                        # Report error in case of a failure
                        if proc.returncode != 0:
                            logger.error('Error running RFM!')
                            raise CalledProcessError(proc.returncode, proc.args)

                        # Read in the RFM results
                        wn_grid, optical_depth = self._read_od_output()
                        path_amt = self._read_path_output(gas)

                        # Initialise output arrays on the first iteration
                        if i == 0:
                            vmrs = np.zeros(len(temperatures))
                            amts = np.zeros(len(temperatures))
                            xsec = np.zeros((len(temperatures), len(wn_grid)))

                        # Add to output arrays
                        vmrs[i] = scaled_vmr
                        amts[i] = path_amt
                        xsec[i] = optical_depth

                        if self.progress_bars and layer.temperature.vary:
                            temp_pbar.update()

                    if self.progress_bars and layer.temperature.vary:
                        temp_pbar.close()

                    # Form as a dataset and save to the cache file
                    cross_sections = xr.Dataset(
                        data_vars={
                            'optical_depth': xr.DataArray(
                                data=xsec,
                                coords={
                                    'temperature': temperatures,
                                    'wavenumber': wn_grid
                                },
                                attrs={'long_name': 'Optical Depth'}
                            ),
                            'vmr': xr.DataArray(
                                data=vmrs, coords={'temperature': temperatures},
                                attrs={
                                    'long_name': 'Volume Mixing Ratio',
                                    'units': 'ppm'
                                }
                            ),
                            'path_amount': xr.DataArray(
                                data=amts, coords={'temperature': temperatures},
                                attrs={
                                    'long_name': 'Path Amount',
                                    'units': 'molecules/cm2'
                                }
                            )
                        },
                        attrs={
                            'pressure': layer.pressure,
                            'species': gas,
                            'path_length': layer.path_length,
                            'wn_lo_limit': wn_lo_limit,
                            'wn_hi_limit': wn_hi_limit,
                            'pts_per_cm': self.pts_per_cm,
                            'initial_temperature': layer.temperature.value,
                            'initial_vmr': gas_vmr
                        }
                    )
                    cross_sections.to_netcdf(cache_fname)

                layer.set_cross_section(gas, cross_sections)

                if self.progress_bars:
                    gas_pbar.update()

            if self.progress_bars:
                gas_pbar.close()

        return params

    def _write_driver_table(self, gas, layer):
        """."""
        # Set up strings depending on the mode
        # If running plume gas then write the plume info
        # if gas.layer == 'plume':
        #     head_str = 'Single layer atmosphere calculation\n'
        #     flag_str = 'HOM OPT CTM MIX PTH'
        #     atm_str = 'atm/layer.atm'
        #     tan_str = f'{self.plume_path_length/1000}'
        #
        # # Otherwise write the atmosphere info, either for solar or layer mode
        # else:
        #     if self.solar_flag:
        #         head_str = '50-layer Earth atmosphere calculation\n'
        #         flag_str = 'ZEN OBS OPT CTM MIX PTH'
        #         atm_str = 'atm/mid_lat_summer.atm'
        #         tan_str = '1.0'
        #     else:
        #         head_str = 'Single layer atmosphere calculation\n'
        #         flag_str = 'HOM OPT CTM MIX PTH'
        #         atm_str = 'atm/layer.atm'
        #         tan_str = f'{self.atmos_path_length/1000}'

        head_str = 'Single layer atmosphere calculation\n'
        flag_str = 'HOM OPT CTM MIX PTH'
        atm_str = 'atm/layer.atm'
        tan_str = f'{layer.path_length/1000}'

        # Open the driver file
        with open(f'{self.wd}/rfm.drv', 'w') as rfmdrv:

            # Write the title
            rfmdrv.write('! RFM driver table for transmittance calculation\n')

            # Write header line for info
            rfmdrv.write(f'*HDR\n{head_str}\n')

            # Write flags depending on retrieval mode to tell RFM how to run
            rfmdrv.write(f'*FLG\n{flag_str}\n')

            # Write grid details: start wavenumber, stop wavenumber and number
            # of points/cm-1
            rfmdrv.write(
                f'*SPC\n{self.wn_start - self.model_padding} '
                f'{self.wn_stop + self.model_padding} '
                f'{self.pts_per_cm}\n'
            )

            rfmdrv.write(
                f'*GAS\n{gas}\n'      # Write gas name
                f'*ATM\n{atm_str}\n'  # Write path to atmosphere file
                f'*TAN\n{tan_str}\n'  # Write the pathlength or air mass factor
            )

            # If solar, write the observer hight in km
            if self.solar_flag:
                rfmdrv.write(f'*OBS\n{self.obs_height/1000}\n')

            # Write output filenames
            rfmdrv.write(
                '*OUT\n  '
                'OPTFIL=rfm.out\n  '
                'PTHFIL=rfm_pth.out\n'
            )

            # Write HITRAN file location
            rfmdrv.write(f'*HIT\n{self.hitran_path}\n')

            # Write the file end
            rfmdrv.write('*END\n')

    def _write_profile(self, pressure, temperature, initial_temperature):
        """."""

        # Open the RFM atmosphere input file
        if not os.path.isdir(f'{self.wd}/atm'):
            os.makedirs(f'{self.wd}/atm')
        with open(f'{self.wd}/atm/layer.atm', 'w') as atm:

            # Write header lines
            atm.write(
                '! Model for atmospheric pressure volume mixing ratios of a '
                'volcanic plume\n'
                '! Designed for input to the RFM forward model\n'
                '  1  ! No.Levels in profiles\n'
                '*HGT [km]\n'
                '    1.0\n'
                '*PRE [mb]\n'
                f' {pressure:.3E}\n'
                '*TEM [K]\n'
                f'          {temperature:.5f}\n'
            )

            # Write molecule lines
            for name, vmr in self.vmr.items():
                scaled_vmr = vmr * initial_temperature / temperature
                atm.write(f'*{name} [ppmv]\n {scaled_vmr:.3E}\n')

            # Write the file end
            atm.write('*END\n')

    def _read_od_output(self, fname=None):
        """."""
        # Check if a specific file path is given, otherwise use the default
        # in the RFM working directory
        if fname is None:
            fname = f'{self.wd}/rfm.out'

        # Read in the main RFM output file
        try:
            with open(fname, 'r') as rfm_out:
                lines = rfm_out.readlines()
        except FileNotFoundError:
            logger.error(
                'RFM path output missing! The RFM did not run. '
                f'Please check RFM output log: {self.wd}/rfm.log'
            )
            raise

        # First three lines are the header, followed by the file info which
        # contains the number of points, start wavenumber, wavenumber step,
        # end wavenumber and description.
        # Split the header row by whitespace. Note the description can be split
        # if it contains any whitespace!
        file_info = lines[3].split()
        npts, wn_start, wn_step, wn_stop = [float(s) for s in file_info[:4]]

        # Compute the wavelength grid
        wn_grid = np.linspace(
            wn_start,
            wn_stop,
            int((wn_stop-wn_start) / wn_step) + 1
        )

        # This is followed by the data, which is split by whitespace across
        # multiple rows
        line_data = [line.strip().split() for line in lines[4:]]
        data = np.array(
            [float(item) for sublist in line_data for item in sublist]
        )

        # Remove the file to avoid reading the wrong one by mistake
        os.remove(fname)

        return wn_grid, data

    def _read_path_output(self, gas, fname=None):
        """."""
        # Check if a specific file path is given, otherwise use the default
        # in the RFM working directory
        if fname is None:
            fname = f'{self.wd}/rfm_pth.out'

        # Read in the file
        try:
            with open(fname, 'r') as path_out:
                lines = [line.strip() for line in path_out.readlines()]
        except FileNotFoundError:
            logger.error(
                'RFM path output missing! The RFM did not run. '
                f'Please check RFM output log: {self.wd}/rfm.log'
            )
            raise

        # The first two lines are the header info, followed by info on the
        # number of gases and number of segments
        ngas, nseg1, nseg2 = [int(s) for s in lines[2].split()[:3]]

        # The next line is the gas in question
        gas_name = lines[3]

        # The next lines give the header for the data followed by the data
        header_info = lines[4].split()
        path_data_arr = [float(s) for s in lines[5].split()]

        # Store the path data in a dictionary
        path_data = {}
        for i, key in enumerate(header_info):
            path_data[key] = path_data_arr[i]

        # Convert the path amount from kmol.cm-2 to molecules.cm-2
        path_amt = path_data['Amt[kmol/cm2]'] * 6.02205e26

        # Check the gas name is the one expected
        if gas_name.upper() != gas.upper():
            logger.exception(
                f'RFM output gas {gas_name.upper()} does not match requested '
                + f'{gas}! Please check RFM output log: {self.wd}/rfm.log'
            )

        # Remove the file to avoid reading the wrong one by mistake
        os.remove(fname)

        return path_amt
