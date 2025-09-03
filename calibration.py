from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, struct
from scipy.signal import savgol_filter

# -----------------------------
# Load reference lamp data
# -----------------------------
lamp_excel_path = "/Users/behnamazizi/Desktop/project/calibration_files/source_data/measurements/Reference/SLS201L_Spectrum.xlsx"

# Read columns C, D, E from Excel (skip first row)
# C: wavelength (nm), D: power intensity, E: black body intensity
data = pd.read_excel(lamp_excel_path, usecols=[2, 3, 4], skiprows=1)
data = np.array(data)
wl, I_p, I_b = data[:,0], data[:,1], data[:,2]

ref_len = len(wl)                     # Number of points in reference spectrum
ref_range = (wl[0], wl[-1])           # Wavelength range

# Interpolate reference data to 0.1 nm resolution for smoother comparison
wl_ref = np.arange(wl[0], wl[-1], 0.1)
I_p_ref = np.interp(wl_ref, wl, I_p)
I_b_ref = np.interp(wl_ref, wl, I_b)

# =============================
# Function: read_raw8
# =============================
def read_raw8(filename):
    """
    Reads Avantes AvaSoft8 binary files (.RAW8/.RWD8/.ABS8/.TRM8/.RFL8/.IRR8/.RIR8) 
    and extracts spectroscopy data: wavelength, sample, dark, reference.
    
    Args:
        filename (str): Path to the binary file
    
    Returns:
        tuple: (Wavelength, Sample, Dark, Reference)
    
    Note:
        - Handles file not found errors.
        - Determines number of used pixels automatically.
    """
    try:
        with open(filename, 'rb') as fid:
            # -----------------------------
            # Extract number of total pixels
            # -----------------------------
            fid.seek(91)
            total_pixels = struct.unpack('<H', fid.read(2))[0] + 1
            
            # -----------------------------
            # Determine number of actually used pixels
            # -----------------------------
            fid.seek(328)
            wl_temp = np.fromfile(fid, dtype='<f4', count=total_pixels)
            diff = np.abs(np.diff(wl_temp, n=2))
            nonzero_idx = np.nonzero(diff > 0.001)[0]
            used_pixels = nonzero_idx[0] + 1 if nonzero_idx.size > 0 else total_pixels
            
            # -----------------------------
            # Read actual data arrays
            # -----------------------------
            fid.seek(328)
            Wavelength = np.fromfile(fid, dtype='<f4', count=used_pixels)
            Sample = np.fromfile(fid, dtype='<f4', count=used_pixels)
            Dark = np.fromfile(fid, dtype='<f4', count=used_pixels)
            Reference = np.fromfile(fid, dtype='<f4', count=used_pixels)
            
            return Wavelength, Sample, Dark, Reference
            
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} does not exist")
    except Exception as e:
        raise ValueError(f"Unsupported file format: {str(e)}")

# =============================
# Function: raw_folder
# =============================
def raw_folder(folder_path, ext='.RAW8', pre=''):
    """
    Reads all raw files in a folder and returns wavelength and intensity arrays.
    
    Args:
        folder_path (str): Path to folder containing raw files
        ext (str): Extension of raw files (default: '.RAW8')
        pre (str): Prefix of files to include (default: '')
        
    Returns:
        dict: {
            'wl': wavelength array,
            'I': 2D array of all spectra,
            'I_av': averaged spectrum
        }
    
    Note:
        - Subtracts dark spectrum from sample
        - Computes average over multiple scans
    """
    wl = []
    I = []
    ext = ext.lower()
    
    files = os.listdir(folder_path)
    raw_files = [file for file in files if file.lower().endswith(ext) and file.startswith(pre)]
    
    if raw_files:
        print(f"Found {len(raw_files)} files with extension {ext} in {folder_path}")
    if not raw_files:
        raise FileNotFoundError(f"No files with extension {ext} found in {folder_path}")
    
    for i, file in enumerate(raw_files):
        landa, sample, dark, reference = read_raw8(os.path.join(folder_path, file))
        
        if i == 0:
            wl = landa
        
        Sample = sample - dark               # Subtract dark current
        # Sample = Sample + np.abs(np.min(Sample))   # Optional: remove negative values
        I.append(Sample)
    
    wl = np.array(wl)
    I = np.array(I)
    I_av = np.mean(I, axis=0)              # Average spectrum across all scans
    
    return {'wl': wl, 'I': I, 'I_av': I_av}

# =============================
# Function: read_txt
# =============================
def read_txt(path, smooth=False):
    """
    Reads text-based spectral data files.
    
    Args:
        path (str): Path to text file
        smooth (bool): Apply Savitzky-Golay smoothing if True (default: False)
        
    Returns:
        tuple: (wavelength array, normalized intensity array)
    
    Note:
        - Automatically detects "Processed Spectral Data" markers
        - Normalizes intensity to max = 1
    """
    target_i = ">>>>>Begin Processed Spectral Data<<<<<"
    target_f = ">>>>>End Processed Spectral Data<<<<<"
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if target_i in "".join(lines):
        i_start = next(i for i, line in enumerate(lines) if target_i in line) + 1
        i_end   = next(i for i, line in enumerate(lines) if target_f in line)
        data = np.loadtxt(lines[i_start:i_end], usecols=(0,1))
        wl, I = data[:,0], data[:,1]
    else:
        wl, I = np.loadtxt(path, usecols=(0,1), unpack=True)
    
    I = I / np.max(I)                      # Normalize intensity
    
    if smooth:
        I = savgol_filter(I, window_length=20, polyorder=3)   # Smooth spectrum
    
    return wl, I

# =============================
# Function: make_calib
# =============================
def make_calib(path=None, num=None, kind=None, source=None, prefix=None, smooth=False):
    """
    Main calibration function to generate calibration curve.
    
    Args:
        path (str): File/folder path containing measured spectrum
        num (str/int): 'single' or 'scan' to indicate single file or folder scan
        kind (str): Instrument type, e.g., "Avantes" or "NIRQuest"
        source (np.array): Reference spectrum as Nx2 array (wl, intensity)
        prefix (str): File prefix to select (optional)
        smooth (bool): Apply smoothing for text files (default: False)
    
    Returns:
        dict: {
            'wl_orginal', 'I_orginal', 'wl_ref', 'I_ref',
            'wl_common', 'I_interp', 'I_ref_interp',
            'I_diff', 'I_calib'
        }
    
    Note:
        - Aligns measured spectrum with reference spectrum
        - Interpolates to common wavelength grid
        - Computes calibration factor
    """
    if path is None:
        raise ValueError("Path to the file or folder containing data must be provided")
    if source is None:
        raise ValueError("Source of the data must be provided")
    else:
        wl_ref = source[:,0]
        I_ref = source[:,1]
    
    # -----------------------------
    # Read measured spectrum
    # -----------------------------
    if kind == "Avantes":
        if num in (None, 1, "single"):
            wl, I, dark, _ = read_raw8(path)
            I = I - dark
        elif num == "scan":
            data = raw_folder(path, ext='.RAW8', pre=prefix)
            wl, I = data['wl'], data['I_av']
        else:
            raise ValueError("Invalid value for num")
    
    elif kind == "NIRQuest":
        if num in (None, 1, "single"):
            wl, I = read_txt(path, smooth=smooth)
        elif num == "scan":
            wl_temp = []
            I_temp = []
            for file in os.listdir(path):
                if file.endswith(".txt"):
                    wl, I = read_txt(os.path.join(path,file), smooth=smooth)
                    wl_temp.append(wl)
                    I_temp.append(I)
            I = np.mean(I_temp, axis=0)
        else:
            raise ValueError("Invalid value for num")
    
    # -----------------------------
    # Align measured spectrum with reference
    # -----------------------------
    wl = np.array(wl)
    wl_ref = np.array(wl_ref)
    wl_min = max(wl.min(), wl_ref.min())
    wl_max = min(wl.max(), wl_ref.max())
    mask = (wl >= wl_min) & (wl <= wl_max)
    mask_ref = (wl_ref >= wl_min) & (wl_ref <= wl_max)

    wl = wl[mask]
    I = I[mask]
    I = I / np.max(I)                     # Normalize
    wl_ref = wl_ref[mask_ref]
    I_ref = I_ref[mask_ref] / np.max(I_ref)

    common_wl = np.linspace(wl_min, wl_max, max(len(wl), len(wl_ref)))
    I_interp = np.interp(common_wl, wl, I)
    I_ref_interp = np.interp(common_wl, wl_ref, I_ref)

    # -----------------------------
    # Compute calibration factor
    # -----------------------------
    return {
        "wl_orginal": wl,
        "I_orginal": I,
        "wl_ref": wl_ref,
        "I_ref": I_ref,
        "wl_common": common_wl,
        "I_interp": I_interp,
        "I_ref_interp": I_ref_interp,
        "I_diff": I_interp - I_ref_interp,
        "I_calib": np.divide(I_ref_interp, I_interp, out=np.zeros_like(I_interp), where=I_interp!=0)
    }

# -----------------------------
# Example usage: generate calibration
# -----------------------------
spec_path = "/Users/behnamazizi/Desktop/project/calibration_files/source_data/measurements/Blue_osceanoptic_QP400_1_VIS_NIR_EOS_27291_AVANTES_1605105U1"

spec_data = make_calib(spec_path, kind="Avantes", num="scan", source=np.column_stack((wl_ref, I_p_ref)),prefix = "1605105U1")

# -----------------------------
# Plot spectra and calibration
# -----------------------------
figure, (ax, bx, cx) = plt.subplots(3, 1, figsize=(9,6), sharex=True)

# Original spectra
ax.plot(spec_data["wl_orginal"], spec_data["I_orginal"]/np.max(spec_data["I_orginal"]), label="Measured", alpha=0.7)
ax.plot(spec_data["wl_ref"], spec_data["I_ref"], label="Reference", alpha=0.7)
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("Original Spectra")
ax.legend()
ax.grid(True)

# Interpolated comparison
bx.plot(spec_data["wl_common"], spec_data["I_interp"], label="Measured (interp)", alpha=0.7)
bx.plot(spec_data["wl_common"], spec_data["I_ref_interp"], label="Reference (interp)", alpha=0.7)
bx.set_ylabel("Intensity (a.u.)")
bx.set_title("Interpolated Spectra (Common Grid)")
bx.legend()
bx.grid(True)

# Calibration factor applied
cx.plot(spec_data["wl_common"], spec_data["I_interp"]*spec_data["I_calib"], label="Calibrated measured")
cx.plot(spec_data["wl_common"], spec_data["I_ref_interp"], label="Reference (interp)", ls='--')
cx.set_xlabel("Wavelength (nm)")
cx.set_ylabel("Intensity (a.u.)")
cx.set_title("Calibration Curve")
cx.legend()
cx.grid(True)
plt.show()

# -----------------------------
# Save calibration data
# -----------------------------
folder_name = os.path.basename(os.path.normpath(spec_path))  # Extract folder name
data_to_save = np.column_stack((spec_data["wl_common"], spec_data["I_calib"]))  # Columns: wl, calib
save_path = os.path.join(spec_path, f"{folder_name}_calibration.TXT")
np.savetxt(save_path, data_to_save, fmt="%.6f", header="Wavelength(nm)\tCalibration_Factor", delimiter="\t")

# Save plot as PNG
figure.savefig(save_path.replace(".TXT", ".png"), dpi=300)

print(f"Calibration data saved to: {save_path}")
