import pandas as pd
import numpy as np
from pyteomics import mzml

def extract_eic(mzml_file, mz, ppm = 10):
    """
    Extracts all the intensity values for a specific m/z value within a certain ppm range. 
    Parameters:
    mzml_file (str): Path to the mzML file
    mz (float): m/z value to extract
    ppm (float): parts per million range to extract
    Returns:
    eic (pd.DataFrame): DataFrame with the retention time and intensity values for the given m/z value
    """
    
    eic_df = []

    with mzml.read(mzml_file) as reader:
        for spectrum in reader:
            if spectrum.get('ms level') == 1:
                scan_info = spectrum['scanList']['scan'][0]
                rt = scan_info.get('scan start time', None)

                mz_array = spectrum.get('m/z array', [])
                intensity_array = spectrum.get('intensity array', [])

                intensity_value = 0
                for mz_val, intensity_val in zip(mz_array, intensity_array):
                    if abs(mz_val - mz) <= mz * ppm / 1e6:
                        intensity_value = intensity_val
                        break
                
                eic_df.append({
                    'RT': rt,
                    'Intensity': intensity_value
                })
    
    eic_df = pd.DataFrame(eic_df)
    return eic_df