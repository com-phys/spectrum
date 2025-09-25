import numpy as np
import struct
import os

def read_raw8(filename):
    """
    Reads Avantes AvaSoft8 binary spectroscopy files and extracts
    wavelength, sample, dark, and reference spectra.

    Supported file formats:
        - .RAW8 / .RWD8
        - .ABS8 / .TRM8
        - .RFL8 / .IRR8 / .RIR8

    Parameters
    ----------
    filename : str
        Full path to the binary file.

    Returns
    -------
    tuple of numpy.ndarray
        (wavelength, sample, dark, reference) arrays with shape (N,)
        where N is the number of used pixels.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is corrupted or unsupported.

    Notes
    -----
    - Automatically determines the number of used pixels by
      detecting discontinuities in wavelength spacing.
    - Assumes little-endian (<) binary format with 32-bit floats.
    - Typical wavelength discontinuity threshold is set to 0.001 nm.
    - Avantes files store multiple spectra sequentially after wavelength.

    Example
    -------
    >>> wl, sample, dark, ref = read_raw8("data/test.RAW8")
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(wl, sample, label="Sample")
    >>> plt.plot(wl, dark, label="Dark")
    >>> plt.plot(wl, ref, label="Reference")
    >>> plt.legend(); plt.show()
    """

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")

    try:
        with open(filename, "rb") as fid:
            # ---- Read total pixels ----
            fid.seek(91)
            total_pixels = struct.unpack("<H", fid.read(2))[0] + 1

            # ---- Read full wavelength array ----
            fid.seek(328)
            wl_temp = np.fromfile(fid, dtype="<f4", count=total_pixels)

            # ---- Detect used pixels ----
            diff = np.abs(np.diff(wl_temp, n=2))
            nonzero_idx = np.nonzero(diff > 0.001)[0]
            used_pixels = nonzero_idx[0] + 1 if nonzero_idx.size > 0 else total_pixels

            # ---- Read spectra ----
            fid.seek(328)
            wavelength = np.fromfile(fid, dtype="<f4", count=used_pixels)
            sample = np.fromfile(fid, dtype="<f4", count=used_pixels)
            dark = np.fromfile(fid, dtype="<f4", count=used_pixels)
            reference = np.fromfile(fid, dtype="<f4", count=used_pixels)

        return wavelength, sample, dark, reference

    except Exception as e:
        raise ValueError(f"Failed to read {filename}: {e}")
