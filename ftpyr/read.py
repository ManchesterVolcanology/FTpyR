"""Read in FTIR spectra into an xarray DataArray.
"""
import os
import struct
import logging
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

# from brukeropusreader import read_file

logger = logging.getLogger(__name__)


def read_spc(filename):
    """Read in .spc file.

    Based on the spc_spectra library written by Rohan Isaac:
    See https://github.com/rohanisaac/spc for details

    Parameters
    ----------
    filename : str
        The file to read in

    Returns
    -------
    xarray DataArray
        Holds the spectrum and metadata
    """
    # Read in the binary file
    with open(filename, "rb") as r:
        content = r.read()

    # Create a dictionary to hold the metadata
    metadata = {'filename': filename}

    # File setup ==============================================================

    # Parse the header data
    header_size = 512
    hdata = struct.unpack(
        "<cccciddicccci9s9sh32s130s30siicchf48sfifc187s",
        content[:header_size]
    )

    # Generate a dict of the header data, converting dtypes if necessary
    info = {
        'tflg': hdata[0],
        'versn': hdata[1],
        'exper': ord(hdata[2]),
        'exp': ord(hdata[3]),
        'npts': int(hdata[4]),
        'first': float(hdata[5]),
        'last': float(hdata[6]),
        'nsub': hdata[7],
        'xtype': ord(hdata[8]),
        'ytype': ord(hdata[9]),
        'ztype': ord(hdata[10]),
        'post': hdata[11],
        'date': hdata[12],
        'res': hdata[13],
        'source': hdata[14],
        'peakpt': hdata[15],
        'spare': hdata[16],
        'cmnt': str(hdata[17]),
        'catxt': hdata[18],
        'logoff': int(hdata[19]),
        'mods': hdata[20],
        'procs': hdata[21],
        'level': hdata[22],
        'sampin': hdata[23],
        'factor': hdata[24],
        'method': hdata[25],
        'zinc': hdata[26],
        'wplanes': hdata[27],
        'winc': hdata[28],
        'wtype': hdata[29],
        'reserv': hdata[30]
    }

    # Generate data flags from the flag number
    flag_bits = [x == '1' for x in list('{0:08b}'.format(ord(info['tflg'])))]
    flags = {
        'sprec': flag_bits[0],
        'cgram': flag_bits[1],
        'multi': flag_bits[2],
        'randm': flag_bits[3],
        'ordrd': flag_bits[4],
        'alabs': flag_bits[5],
        'xyxys': flag_bits[6],
        'xvals': flag_bits[7]
    }

    # Replace null characters in comment with spaces
    metadata['comment'] = info['cmnt'].replace('\x00', ' ')

    # Get the file modification time if available as this goes to the second
    try:
        mtime = pd.Timestamp(os.path.getmtime(filename), unit='s')
        metadata['timestamp'] = str(mtime)

    # If it is corrupted, then use the timestamp in the header
    except OSError:
        d = info['date']
        year = d >> 20
        month = (d >> 16) % (2**4)
        day = (d >> 11) % (2**5)
        hour = (d >> 6) % (2**5)
        minute = d % (2**6)
        ts = datetime(year, month, day, hour, minute)
        metadata['timestamp'] = str(pd.Timestamp(ts))

    # Generate the x data from the header info
    xdata = np.linspace(info['first'], info['last'], num=info['npts'])

    # Log the file format for debugging
    if flags['xyxys']:
        # x values are given
        dat_fmt = '-xy'
    elif flags['xvals']:
        # only one subfile, which contains the x data
        dat_fmt = 'x-y'
    else:
        # no x values are given, but they can be generated
        dat_fmt = 'gx-y'

    logger.debug(f'{dat_fmt}({info["nsub"]})')

    if dat_fmt != 'gx-y':
        logger.error('Not able to read files with multiple spectra yet!')

    # Set the position within the file data
    file_pos = header_size

    # Read in the sub-header
    subheader = struct.unpack(
        '<cchfffiif4s',
        content[file_pos:file_pos+32]
    )
    subinfo = {
        'flgs': subheader[0],
        'exp': subheader[1],
        'indx': subheader[2],
        'time': subheader[3],
        'next': subheader[4],
        'nois': subheader[5],
        'npts': subheader[6],
        'scan': subheader[7],
        'wlevel': subheader[8],
        'resv': subheader[9]
    }

    # Add info and sub info to the metadata
    metadata['header_info'] = info
    metadata['subinfo'] = subinfo

    # Move the file position
    file_pos += 32

    # Get subfile size (4 bytes per point)
    data_size = (4 * info['npts'])

    # Parse the spectral info
    spec_data = content[file_pos:file_pos+data_size]
    ydata = np.array(
        [yi for yi in struct.iter_unpack('<f', spec_data)]
    ).flatten()

    # Progress the file position
    file_pos += 32 + data_size

    # Parse the log content, removing "\x00" and splitting by "\r\n"
    logs = content[file_pos:].replace(b'\x00', b'').split(b'\r\n')

    # Sort the log data, splitting values with an "=" into a dict and storing
    # the rest as a list
    log_dict = {}
    log_list = []
    for log in logs:
        if b'=' in log:
            key, val = log.split(b'=')[:2]
            log_dict[key] = val
        else:
            log_list.append(log)

    # Add to the metadata
    metadata['log_dict'] = log_dict
    metadata['log_list'] = log_list

    # Form the output DataArray
    spectrum = xr.DataArray(
        data=ydata,
        coords={'Wavenumber': xdata},
        attrs=metadata
    )

    return spectrum


def read_spectacle(filename):
    """."""
    # Read in the binary file
    with open(filename, "rb") as r:
        content = r.read()

    # Create a dictionary to hold the metadata
    metadata = {'filename': filename}

    # File setup ==============================================================

    # Parse the header data
    header_size = 512
    hdata = struct.unpack(
        "<6sddd88s17s175sl198s",
        content[:header_size]
    )

    # Unpack the desired variables
    wn_start = float(hdata[1])
    wn_stop = float(hdata[2])
    date_str = hdata[5].replace(b'\x00', b' ').decode("utf-8")
    npts = int(hdata[7])

    # Parse the timestamp
    ts = datetime.strptime(date_str, "%d/%m/%y %H:%M:%S")
    metadata['timestamp'] = str(pd.Timestamp(ts))

    # Get data size (4 bytes per point)
    data_size = (4 * npts)

    # Get the spectrum data
    file_pos = 512
    spec_data = content[file_pos:file_pos+data_size]
    ydata = np.array(
        [yi for yi in struct.iter_unpack('<i', spec_data)]
    ).flatten()

    # Calculate the wavenumber grid
    xdata = np.linspace(wn_start, wn_stop, num=npts)

    # Form the output DataArray
    spectrum = xr.DataArray(
        data=ydata,
        coords={'Wavenumber': xdata},
        attrs=metadata
    )

    return spectrum


def read_sbm(filename):
    """."""
    raise ValueError('Read function not implemented yet!')


def read_cerburus(filename):
    """."""
    raise ValueError('Read function not implemented yet!')


def read_bruker(filename):
    """."""
    # spec_data = read_file(filename)
    raise ValueError('Read function not implemented yet!')
    return spec_data


def read_sb(filename):
    """."""
    raise ValueError('Read function not implemented yet!')


def read_ifg(filename):
    """."""
    raise ValueError('Read function not implemented yet!')


# Define file types and read functions based on extension
filetype_dict = {
    'Midac Essential': read_sbm,
    'Midac TuHH': read_cerburus,
    'Bruker OPUS': read_bruker,
    'Midac AUTOQUANT': read_sb,
    'Midac Spectacle': read_spectacle,
    'Essential interferogram': read_ifg,
    'Essential spectrum': read_spc
}

extension_dict = {
    '.sbm': 'Midac Essential',
    '.hsi': 'Midac TuHH',
    '.0': 'Bruker OPUS',
    '.sb': 'Midac AUTOQUANT',
    '.rsb': 'Midac AUTOQUANT',
    '.irs': 'Midac Spectacle',
    '.ifg': 'Essential interferogram',
    '.spc': 'Essential spectrum'
}


def read_spectrum(filename, file_type=None):
    """."""
    # Ensure the filename used UNIX slashes
    filename = filename.replace('\\', '/')

    # If a file type is not give, infer from the extension
    if file_type is None:

        # Get the file extension
        file, extension = os.path.splitext(filename)

        # Assign the file type
        file_type = extension_dict[extension.lower()]

    # Assign the correct read function
    read_func = filetype_dict[file_type]

    # Read in the spectrum
    spectrum = read_func(filename)

    return spectrum
