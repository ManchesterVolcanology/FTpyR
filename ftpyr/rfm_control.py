import os
import yaml
import logging
import numpy as np
import xarray as xr
from subprocess import Popen, PIPE, CalledProcessError

logger = logging.getLogger(__name__)


class RFM(object):
    """Wrapper to call the Reference Forward Model (RFM).

    Uses a precompiled RFM executable and the HITRAN line database to calculate
    requested gas optical depth spectra in order to fit measured OP-FTIR
    spectra. Gas optical depths are calculated one by one for both the voclanic
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
                 pts_per_cm=25, vmr_file='databases/atm_layer.yml'):
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

            for gas in layer.gases.values():

                # Pull gas mixing ratio
                gas_vmr = self.vmr[gas.name]

                # Form cache filename
                wn_lo_limit = self.wn_start - self.model_padding
                wn_hi_limit = self.wn_stop + self.model_padding
                fname = str(
                    f'{gas.name}_'
                    f'{wn_lo_limit:.1f}_{wn_hi_limit:.1f}_'
                    f'{layer.pressure:.1f}_{layer.temperature:.1f}_'
                    f'{layer.path_length}_{self.pts_per_cm}_{gas_vmr:.2E}'
                ).replace('.', 'p') + '.nc'

                # Generate a flag to control if the calculation is run or not
                calc_od_flag = True

                # If the cache file exists, read it in. Otherwise run RFM
                cache_fname = f'{self.cache_folder}/{fname}'
                if os.path.isfile(cache_fname):
                    logger.info(f'Cache file found for {gas.name}')
                    with xr.open_dataarray(cache_fname) as da:

                        # Check that the precise temperature, pressure and window
                        # match the requested ones
                        attrs = da.attrs
                        checks = np.array(
                            [
                                gas.name == attrs['species'],
                                layer.temperature == attrs['temperature_k'],
                                layer.pressure == attrs['pressure_mb'],
                                layer.path_length == attrs['path_length'],
                                self.pts_per_cm == attrs['pts_per_cm'],
                                wn_lo_limit == attrs['wn_lo_limit'],
                                wn_hi_limit == attrs['wn_hi_limit'],
                                gas_vmr == attrs['gas_vmr']
                            ]
                        )

                        if checks.all():
                            optical_depth = da.to_numpy()
                            path_amt = da.attrs['path_amt']
                            calc_od_flag = False

                        else:
                            logger.info(
                                'Consistency check failed, '
                                're-calculating optical depth'
                            )

                if calc_od_flag:

                    # Write the atmosphere profile and driver table for this gas
                    self._write_profile(layer.pressure, layer.temperature)
                    self._write_driver_table(gas, layer)

                    logger.info(
                        f'Running RFM for {gas.name} '
                        f'at {layer.temperature:.1f} K, {layer.pressure:.1f} mb'
                        f' and path length {layer.path_length} m'
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
                    path_amt = self._read_path_output(gas.name)

                    # Form as a dataarray and save to the cache file
                    da = xr.DataArray(
                        data=optical_depth,
                        coords={'Wavenumber': wn_grid},
                        attrs={
                            'path_amt': path_amt,
                            'path_amt_units': 'molecules.cm-2',
                            'wavenumber_units': 'cm-1',
                            'temperature_k': layer.temperature,
                            'pressure_mb': layer.pressure,
                            'species': gas.name,
                            'path_length': layer.path_length,
                            'wn_lo_limit': wn_lo_limit,
                            'wn_hi_limit': wn_hi_limit,
                            'pts_per_cm': self.pts_per_cm,
                            'gas_vmr': gas_vmr
                        }
                    )
                    da.to_netcdf(cache_fname)

                # Set the optical depth and path_amt for the gas parameter
                layer.optical_depths[gas.name] = optical_depth
                layer.path_amounts[gas.name] = path_amt

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

            # Write gas name
            rfmdrv.write(f'*GAS\n{gas.name}\n')

            # Write path to atmosphere file
            rfmdrv.write(f'*ATM\n{atm_str}\n')

            # Write the pathlength or air mass factor
            rfmdrv.write(f'*TAN\n{tan_str}\n')

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

    def _write_profile(self, pressure, temperature):
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
            for name, mol in self.vmr.items():
                atm.write(f'*{name} [ppmv]\n {mol:.3E}\n')

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
