import os
import yaml
import logging
import numpy as np
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
    plume_path_length : float
        The path length through the volcanic plume layer in meters
    atmos_path_length : float, optional
        The physical path length through the atmosphere layer in meters. Not
        required if solar_flag is True. Default is None
    solar_flag : bool, optional
        If True, then calculation is perfromed using a full atmosphere.
        Requires obs_height to be defined. Default is False.
    obs_height : float, optional
        The altitude of the instrument in meters above sea level, used if
        solar_flag is True. Default is None

    Methods
    -------
    calc_optical_depths(gases, atmos_pres, atmos_temp, plume_pres, plume_temp)
        Calculate the gas optical depths
    """

    npts_per_cm = 100

    def __init__(self, exe_path, hitran_path, wn_start, wn_stop,
                 plume_path_length, atmos_path_length=None, solar_flag=False,
                 obs_height=None):
        """."""
        # Assign object variables =============================================
        self.exe_path = str(exe_path)
        self.wd = str(os.path.split(self.exe_path)[0])
        self.hitran_path = str(hitran_path)
        self.wn_start = float(wn_start)
        self.wn_stop = float(wn_stop)
        self.plume_path_length = plume_path_length
        self.solar_flag = bool(solar_flag)
        if obs_height is None:
            self.obs_height = None
        else:
            self.obs_height = float(obs_height)
        if atmos_path_length is None:
            self.atmos_path_length = None
        else:
            self.atmos_path_length = float(atmos_path_length)

        # Perform input checks ================================================
        # Check the RFM working directory isn't the current one
        if self.wd == '':
            self.wd = '.'

        # Check the the start wavenumber is smaller than the end
        if self.wn_start >= self.wn_stop:
            logger.error(
                'Start wavenumber must be lower than stop wavenumber!'
            )

        # If solar flag is on then the altitude must be specified
        if self.solar_flag and self.obs_height is None:
            logger.error('Observation height must be defined for solar mode!')

        # If not in solar mode then check an atmosphere path length is given
        if not self.solar_flag and self.atmos_path_length is None:
            logger.error('Atmosphere path length must be defined!')

        # Calculate the model x-grid ==========================================
        # This includes a 1 cm-1 padding on either side to allow shifts
        self.xgrid_npts = self.npts_per_cm*((wn_stop - wn_start) + 2) + 1
        self.xgrid = np.linspace(wn_start-1, wn_stop+1, self.xgrid_npts)

    def calc_optical_depths(self, params, plume_pres, plume_temp, atmos_pres,
                            atmos_temp):
        """Calculate gas optical depths using RFM.

        Parameters
        ----------
        params : parameters.Parameters object
            Contains the fit parameters used in the retrieval
        plume_pres : float
            The pressure of the plume layer in mb
        plume_temp : float
            The temperature of the plume layer in K
        atmos_pres : float
            The pressure of the atmos layer in mb
        atmos_temp : float
            The temperature of the atmos layer in K

        Returns
        -------
        """
        # Extract the gas parameters from the main Parameters
        gas_params = params.extract_gases()

        # Iterate through the desired gases
        for i, [name, gas] in enumerate(gas_params.items()):

            # Get the temperature and pressure, depending on whether plume or
            # atmos
            if gas.layer == 'plume':
                pressure, temperature = plume_pres, plume_temp
            elif gas.layer == 'atmos':
                pressure, temperature = atmos_pres, atmos_temp
            else:
                logger.error(f'Parameter {gas.name} layer ({gas.layer}) '
                             'not recognised!')

            # Write the atmosphere profile and driver table for this gas
            self._write_profile(pressure, temperature)
            self._write_driver_table(gas)

            # Execute RFM
            with Popen(self.exe_path, stdout=PIPE, bufsize=1,
                       universal_newlines=True, cwd=self.wd) as proc:
                for line in proc.stdout:
                    logger.debug(line)

            # Report error in case of a failure
            if proc.returncode != 0:
                logger.error('Error running RFM!')
                raise CalledProcessError(proc.returncode, proc.args)

            # Read in the RFM results
            wn_grid, od = self._read_od_output()
            path_amt = self._read_path_output(gas)

            # Set the optical depth and path_amt to the gas parameter
            params[gas.name].original_od = od
            params[gas.name].original_amt = path_amt

        return params

    def _write_driver_table(self, gas):
        """."""
        # Set up strings depending on the mode
        # If running plume gas then write the plume info
        if gas.layer == 'plume':
            head_str = 'Single layer atmosphere calculation\n'
            flag_str = 'HOM OPT CTM MIX PTH'
            atm_str = 'atm\\layer.atm'
            tan_str = f'{self.plume_path_length/1000}'

        # Otherwise write the atmosphere info, either for solar or layer mode
        else:
            if self.solar_flag:
                head_str = '50-layer Earth atmosphere calculation\n'
                flag_str = 'ZEN OBS OPT CTM MIX PTH'
                atm_str = 'atm\\mid_lat_summer.atm'
                tan_str = '1.0'
            else:
                head_str = 'Single layer atmosphere calculation\n'
                flag_str = 'HOM OPT CTM MIX PTH'
                atm_str = 'atm\\layer.atm'
                tan_str = f'{self.atmos_path_length/1000}'

        # Open the driver file
        with open(f'{self.wd}\\rfm.drv', 'w') as rfmdrv:
            # Write the title
            rfmdrv.write('! RFM driver table for transmittance calculation\n')

            # Write header line for info
            rfmdrv.write(f'*HDR\n{head_str}\n')

            # Write flags depending on retrieval mode to tell RFM how to run
            rfmdrv.write(f'*FLG\n{flag_str}\n')

            # Write grid details: start wavenumber, stop wavenumber and number
            # of points/cm-1. Note a 1 cm-1 padding is applied
            rfmdrv.write(f'*SPC\n{self.wn_start-1} {self.wn_stop+1} '
                         f'{self.npts_per_cm}\n')

            # Write gas name
            rfmdrv.write(f'*GAS\n{gas.species}\n')

            # Write path to atmosphere file
            rfmdrv.write(f'*ATM\n{atm_str}\n')

            # Write the pathlength or air mass factor
            rfmdrv.write(f'*TAN\n{tan_str}\n')

            # If solar, write the observer hight in km
            if self.solar_flag:
                rfmdrv.write(f'*OBS\n{self.obs_height/1000}\n')

            # Write HITRAN file location
            rfmdrv.write(f'*HIT\n{self.hitran_path}\n')

            # Write the file end
            rfmdrv.write('*END\n')

    def _write_profile(self, pressure, temperature):
        """."""
        # Read in the molecule data
        with open('databases/atm_layer.yml', 'r') as ymlfile:
            vmr = yaml.load(ymlfile, Loader=yaml.FullLoader)

        # Open the RFM atmosphere input file
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
            for name, mol in vmr.items():
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
        with open(fname, 'r') as rfm_out:
            lines = rfm_out.readlines()

        # First three lines are the header, followed by the file info which
        # contains the number of points, start wavenumber, wavenumber step,
        # end wavenumber and description.
        # Split the header row by whitespace. Note the description can be split
        # if it contains any whitespace!
        file_info = lines[3].split()
        npts, wn_start, wn_step, wn_stop = [float(s) for s in file_info[:4]]

        # Compute the wavelength grid
        wn_grid = np.linspace(wn_start, wn_stop, int(npts))

        # This is followed by the data, which is split by whitespace across
        # multiple rows
        line_data = [line.strip().split() for line in lines[4:]]
        data = np.array(
            [float(item) for sublist in line_data for item in sublist]
        )

        return wn_grid, data

    def _read_path_output(self, gas, fname=None):
        """."""
        # Check if a specific file path is given, otherwise use the default
        # in the RFM working directory
        if fname is None:
            fname = f'{self.wd}/rfm_pth.out'

        # Read in the file
        with open(fname, 'r') as path_out:
            lines = [line.strip() for line in path_out.readlines()]

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
        if gas_name.upper() != gas.species.upper():
            logger.error(
                f'RFM output gas {gas_name.upper()} does not match requested '
                + f'{gas.species}!'
            )

        return path_amt
