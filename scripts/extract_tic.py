import pandas as pd
import numpy as np
from pyteomics import mzml

def extract_tic(mzml_file):
    """
    Extracts the total ion current (TIC) for each retention time.
    Parameters:
    mzml_file (str): Path to the mzML file
    Returns:
    tic (pd.DataFrame): DataFrame with the retention time and TIC values
    """

    tic_df = []

    with mzml.read(mzml_file) as reader:
        for spectrum in reader:
            if spectrum.get('ms level') == 1:
                scan_info = spectrum['scanList']['scan'][0]
                rt = scan_info.get('scan start time', None)
                tic = spectrum.get('total ion current', None)
                tic_df.append({
                    'RT': rt,
                    'TIC': tic
                })
    
    tic_df = pd.DataFrame(tic_df)
    return tic_df