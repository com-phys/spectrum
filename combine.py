import numpy as np

def combine_spectra(wl_1, I_1, wl_2, I_2, 
                    correction_1=None, correction_2=None, resolution=None):
    """
    Combine two spectra into one continuous spectrum by applying corrections,
    interpolating to common resolution, scaling intensities in the overlap,
    and merging into a final spectrum. through the code wl means wavelength.
    
    Args:
        wl_1, I_1 (array): Wavelengths and intensities of first spectrum
        wl_2, I_2 (array): Wavelengths and intensities of second spectrum
        correction_1 (tuple or None): (wl_cor, I_cor) for spectrum 1 correction
        correction_2 (tuple or None): (wl_cor, I_cor) for spectrum 2 correction
        resolution (float): Step size in nm for interpolation (default=0.5)
    
    Returns:
        dict with:
            - "wl_1": wavelength grid of first spectrum
            - "I_1": intensity values of first spectrum
            - "wl_2": wavelength grid of second spectrum
            - "I_2": intensity values of second spectrum
            - "wl_final": merged wavelength grid
            - "I_final": merged intensity values
            - "coef": scaling coefficient applied to first spectrum
            - "overlap_range": (min_wl, max_wl) of overlap
    """
    if resolution is None:
        resolution = 0.5
    # -----------------------------
    # Interpolation helper
    # -----------------------------
    def interp_spectrum(wl, I, correction=None):
        wl_new = np.round(np.arange(wl.min(), wl.max(), resolution), 1) # create an array from wl_min to wl_max with resoultion 
        I_new = np.interp(wl_new, wl, I)
        if correction is not None:
            wl_cor, I_cor = correction
            I_cor_interp = np.interp(wl_new, wl_cor, I_cor)
            I_new *= I_cor_interp
        return wl_new, I_new
    
    # Interpolate both spectra
    wl_1_new, I_1_new = interp_spectrum(wl_1, I_1, correction_1)
    wl_2_new, I_2_new = interp_spectrum(wl_2, I_2, correction_2)
    
    # -----------------------------
    # Find overlap region
    # -----------------------------
    if wl_1_new[-1] < wl_2_new[0] or wl_2_new[-1] < wl_1_new[0]:
        raise ValueError("There is no overlap between the two spectra.")
    
    if wl_1_new[0] <= wl_2_new[0] <= wl_1_new[-1] <= wl_2_new[-1]:
        overlap = (wl_2_new[0], wl_1_new[-1])
        first_wl, first_I = wl_1_new, I_1_new
        second_wl, second_I = wl_2_new, I_2_new
    elif wl_2_new[0] <= wl_1_new[0] <= wl_2_new[-1] <= wl_1_new[-1]:
        overlap = (wl_1_new[0], wl_2_new[-1])
        first_wl, first_I = wl_2_new, I_2_new
        second_wl, second_I = wl_1_new, I_1_new
    else:
        # Case where one spectrum is fully inside the other
        overlap = (max(wl_1_new[0], wl_2_new[0]), min(wl_1_new[-1], wl_2_new[-1]))
        if wl_1_new[0] < wl_2_new[0]:
            first_wl, first_I = wl_1_new, I_1_new
            second_wl, second_I = wl_2_new, I_2_new
        else:
            first_wl, first_I = wl_2_new, I_2_new
            second_wl, second_I = wl_1_new, I_1_new
    
    # -----------------------------
    # Compute scaling coefficient
    # -----------------------------

    mask_first = (first_wl >= overlap[0]) & (first_wl <= overlap[1])        # mask to extract overlap region from first spectrum
    mask_second = (second_wl >= overlap[0]) & (second_wl <= overlap[1])     # mask to extract overlap region from second spectrum
    
    common_wl = np.arange(overlap[0], overlap[1]+resolution, resolution)    # overlap wavelength
    first_overlap = np.interp(common_wl, first_wl[mask_first], first_I[mask_first])                 # this is extra step just to ensure that we have the same grid for both spectrums in overlap region
    second_overlap = np.interp(common_wl, second_wl[mask_second], second_I[mask_second])
    
    coef = np.mean(second_overlap / first_overlap)
    first_I *= coef  # scale first spectrum
    
    # -----------------------------
    # Merge spectra
    # -----------------------------
    
    wl_final = np.union1d(first_wl, second_wl)      #   union of both wavelength grids
    wl_final = np.round(wl_final, 1)
    I_first_final = np.interp(wl_final, first_wl, first_I, left=np.nan, right=np.nan)
    I_second_final = np.interp(wl_final, second_wl, second_I, left=np.nan, right=np.nan)
    
    I_final = np.where(~np.isnan(I_first_final), I_first_final, I_second_final)
    # For overlap region, prefer scaled first spectrum
    overlap_mask = (wl_final >= overlap[0]) & (wl_final <= overlap[1])
    I_final[overlap_mask] = I_first_final[overlap_mask]
    
    print("Overlap range is: (", overlap[0], ", ", overlap[1], ")", "and overlap coef is: ", coef)
    print("Matching the spectrums is done")
    return {
        "wl_1":wl_1_new,
        "wl_2":wl_2_new,
        "I_1":I_1_new,
        "I_2":I_2_new,
        "wl_final": wl_final,
        "I_final": I_final,
        "coef": coef,
        "overlap_range": overlap
    }
