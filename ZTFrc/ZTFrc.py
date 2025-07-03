# Standard libraries
import os, pickle, re, time, logging, operator,subprocess,warnings
# Numerical and data tools
import numpy as np
import pandas as pd
# Astronomy tools
from rocks import Rock
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.timeseries import LombScargle
from astroquery.gaia import Gaia
from astroquery.jplhorizons import Horizons
# Signal processing and fitting
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline,interp1d
from scipy.optimize import curve_fit

# Parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


#---------------------------------------------------
# Utils.py
#---------------------------------------------------
def get_files(directory, extension='.fits'):
    """
    Returns a sorted list of full paths to files in the given directory with a specific extension.

    Parameters:
        directory (str): Path to the directory containing the files.
        extension (str): File extension to search for (default: '.fits').
    
    Returns:
        list of str: Sorted list of full paths to matching files.
    """
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extension.lower())
    ])


def get_object_information_ssocard(target):
    """
    Retrieve published physical and color information from the SsoCard platform (Berthier et al. 2023),
    returning a human-readable dictionary with fields and references.
    """

    def safe_get(attr, default=None):
        try:
            return attr
        except (AttributeError, IndexError, TypeError):
            return default

    def is_valid(value):
        return value is not None and not np.isnan(value)

    def get_color_info(body, color_code):
        try:
            color_data = getattr(body.parameters.physical.color, color_code)
            return (
                color_data.color.value,
                color_data.color.error_,
                color_data.facility.value,
                color_data.id_filter_1.value,
                color_data.id_filter_2.value,
                color_data.bibref.shortbib[0] if hasattr(color_data.bibref, 'shortbib') else None
            )
        except AttributeError:
            return np.nan, np.nan, None, None, None, None

    def average_colors(color_list):
        valid = [(v, e, f, f1, f2, ref) for v, e, f, f1, f2, ref in color_list if is_valid(v) and is_valid(e)]
        if not valid:
            return np.nan, np.nan, None, None, None, []
        avg = round(np.mean([v for v, *_ in valid]), 4)
        err = round(np.sqrt(np.sum([e**2 for _, e, *_ in valid])), 4)
        refs = list(set(ref for _, _, _, _, _, ref_list in valid for ref in (ref_list or [])))
        return avg, err, valid[0][2], valid[0][3], valid[0][4], refs

    body = Rock(target)

    # Diameter
    diameter = {
        'value': safe_get(body.parameters.physical.diameter.value),
        'error': safe_get(body.parameters.physical.diameter.error_),
        'ref': safe_get(body.parameters.physical.diameter.bibref.shortbib),
        'doi': safe_get(body.parameters.physical.diameter.bibref.doi)
    }

    # H-magnitude
    absolute_magnitude = {
        'H': safe_get(body.parameters.physical.absolute_magnitude.value),
        'H_error': safe_get(body.parameters.physical.absolute_magnitude.error_),
        'ref': safe_get(body.parameters.physical.absolute_magnitude.bibref.shortbib),
        'doi': safe_get(body.parameters.physical.absolute_magnitude.bibref.doi)
    }

    # Color indices
    r_i, r_i_err, r_i_fac, r_i_f1, r_i_f2, r_i_ref = get_color_info(body, 'r_i')
    g_r, g_r_err, g_r_fac, g_r_f1, g_r_f2, g_r_ref = get_color_info(body, 'g_r')

    # Derive g-r if not present
    g_r_refs = g_r_ref or []
    color_sources = []

    if not is_valid(g_r):
        g_i, g_i_err, g_i_fac, g_i_f1, g_i_f2, g_i_ref = get_color_info(body, 'g_i')
        g_z, g_z_err, g_z_fac, g_z_f1, g_z_f2, g_z_ref = get_color_info(body, 'g_z')
        r_z, r_z_err, r_z_fac, r_z_f1, r_z_f2, r_z_ref = get_color_info(body, 'r_z')
        v_r, v_r_err, v_r_fac, v_r_f1, v_r_f2, v_r_ref = get_color_info(body, 'v_r')
        v_g, v_g_err, v_g_fac, v_g_f1, v_g_f2, v_g_ref = get_color_info(body, 'v_g')

        if is_valid(g_i) and is_valid(r_i) and g_i_fac == r_i_fac:
            gr = round(g_i - r_i, 4)
            gr_err = round(np.sqrt(g_i_err**2 + r_i_err**2), 4)
            refs = list(set(filter(None, [g_i_ref, r_i_ref])))
            color_sources.append((gr, gr_err, g_i_fac, g_i_f1[:-2], g_i_f2[:-2], refs))

        if is_valid(g_z) and is_valid(r_z) and g_z_fac == r_z_fac:
            gr = round(g_z - r_z, 4)
            gr_err = round(np.sqrt(g_z_err**2 + r_z_err**2), 4)
            refs = list(set(filter(None, [g_z_ref, r_z_ref])))
            color_sources.append((gr, gr_err, g_z_fac, g_z_f1[:-2], g_z_f2[:-2], refs))

        if is_valid(v_r) and is_valid(v_g) and v_r_fac == v_g_fac:
            gr = round(v_r - v_g, 4)
            gr_err = round(np.sqrt(v_r_err**2 + v_g_err**2), 4)
            refs = list(set(filter(None, [v_r_ref, v_g_ref])))
            color_sources.append((gr, gr_err, v_r_fac, v_r_f1[:-2], v_g_f2[:-2], refs))

        g_r, g_r_err, g_r_fac, g_r_f1, g_r_f2, g_r_refs = average_colors(color_sources)

    # Spin period
    spin = {
        'period': safe_get(body.parameters.physical.spin.period.value),
        'error': safe_get(body.parameters.physical.spin.period.error_),
        'ref': safe_get(body.parameters.physical.spin.bibref.shortbib),
        'doi': safe_get(body.parameters.physical.spin.bibref.doi)
    }

    # Final result dictionary
    result = {
        'target': target,
        'diameter_km': diameter,
        'absolute_magnitude': absolute_magnitude,
        'color_indices': {
            'r-i': {
                'value': r_i if is_valid(r_i) else 0,
                'error': r_i_err if is_valid(r_i_err) else 0,
                'facility': r_i_fac,
                'filters': [r_i_f1, r_i_f2],
                'ref': r_i_ref
            },
            'g-r': {
                'value': g_r if is_valid(g_r) else 0,
                'error': g_r_err if is_valid(g_r_err) else 0,
                'facility': g_r_fac,
                'filters': [g_r_f1, g_r_f2],
                'refs': g_r_refs
            }
        },
        'spin_period_hr': spin
    }

    return result


def filter_table(table, column, min_value=None, max_value=None,
                          allowed_values=None, save=False, save_path='filtered_table.csv'):
    """
    Filter a table (Astropy or Pandas) by column values using numeric thresholds and/or allowed categorical values.
    
    Parameters:
        table (astropy.table.Table or pandas.DataFrame): The table to filter.
        column (str): Name of the column to apply filters on.
        min_value (float, optional): Minimum value (inclusive) for numeric filtering. Default is None.
        max_value (float, optional): Maximum value (inclusive) for numeric filtering. Default is None.
        allowed_values (list, optional): List of exact values to allow for filtering (e.g., strings or categories). Default is None.
        save (bool): Whether to save the filtered DataFrame as a CSV. Default is False.
        save_path (str): Output path for the filtered CSV. Used only if save=True.
    
    Returns:
        astropy.table.Table or pandas.DataFrame:
            The filtered table is returned in the same format as the input.
    """
    is_astropy = isinstance(table, Table)
    if not isinstance(table, pd.DataFrame):
        table = table.to_pandas()

    condition = pd.Series(True, index=table.index)

    if allowed_values is not None:
        condition &= table[column].isin(allowed_values)

    if min_value is not None:
        condition &= table[column] >= min_value

    if max_value is not None:
        condition &= table[column] <= max_value

    filtered = table[condition]

    if save:
        filtered.to_csv(save_path, index=False)
        logging.info(f"Saved {len(filtered)} rows to {save_path}")
    else:
        logging.info(f"Filtered {len(filtered)} rows (not saved)")
    # Returns an Astropy Table when the input was in this format, otherwise returns a pandas df. 
    return Table.from_pandas(filtered) if is_astropy else filtered

def split_table_by_time_gaps(table, jd_column='jd_lt_corr', gap_threshold=90):
    """
    Split a table into sub-tables wherever the time gap exceeds the threshold.

    Parameters
    ----------
    table : astropy.table.Table
        Input table containing a time column (Julian Dates).
    jd_column : str, optional
        Name of the column containing Julian Dates (default is 'jd_lt_corr').
    gap_threshold : float
        Minimum time gap (in days) to trigger a new segment (default is 90 days).

    Returns
    -------
    list_of_tables : list of astropy.table.Table
        List of sub-tables split by time gaps.
    """
    # Sort by JD if not already sorted
    table = table.copy()
    table.sort(jd_column)

    # Compute time differences
    jd_values = table[jd_column]
    time_diffs = np.diff(jd_values)

    # Find indices where time gap exceeds the threshold
    split_indices = np.where(time_diffs > gap_threshold)[0] + 1  # +1 to get start of next group

    # Split the table
    sub_tables = []
    start_idx = 0
    for idx in split_indices:
        sub_tables.append(table[start_idx:idx])
        start_idx = idx
    sub_tables.append(table[start_idx:])  # add the last chunk

    return sub_tables



def general_sine(t, A, f, phi, C):
    """General sine function: A * sin(2π f t + φ) + C"""
    return A * np.sin(2 * np.pi * f * t + phi) + C

def fit_weighted_sine(df, time_col='orbit_av_time', mag_col='amp', mag_err_col='amp_err',
                      time_err_col='orbit_av_time_err', color_col='n_points', title='Weighted Sine Fit'):
    """
    Fit a weighted sinusoidal model to amplitude-vs-time data and return an interactive plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing time, amplitude, and error columns.
    time_col : str
        Column name for time (e.g., 'orbit_av_time').
    mag_col : str
        Column name for measured amplitude or magnitude (e.g., 'amp').
    mag_err_col : str
        Column name for y-axis errors (e.g., 'amp_err').
    time_err_col : str
        Column name for x-axis errors (optional).
    color_col : str
        Column name to color the scatter plot (e.g., 'n_points').
    title : str
        Title to use for the figure.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plot showing the data points with error bars and the best-fit sine curve.
    popt : tuple
        Best-fit parameters (A, f, phi, C).
    """

    # Extract and normalize data
    t = df[time_col].to_numpy(dtype=float)
    y = df[mag_col].to_numpy(dtype=float)
    yerr = df[mag_err_col].to_numpy(dtype=float)
    t_norm = t - np.mean(t)

    # Initial parameter estimates
    A0 = (max(y) - min(y)) / 2
    f0 = 1 / (t.max() - t.min())
    phi0 = 0
    C0 = np.mean(y)

    # Curve fitting with weights
    popt, _ = curve_fit(
        general_sine,
        t_norm,
        y,
        p0=[A0, f0, phi0, C0],
        sigma=yerr,
        absolute_sigma=True
    )

    A_fit, f_fit, phi_fit, C_fit = popt

    # Evaluate model
    t_fit = np.linspace(min(t_norm), max(t_norm), 500)
    y_fit = general_sine(t_fit, A_fit, f_fit, phi_fit, C_fit)
    t_fit_plot = t_fit + np.mean(t)

    # Create plot
    fig = px.scatter(
        df,
        x=time_col,
        y=mag_col,
        color=color_col,
        title=title
    )
    fig.update_traces(
        error_x=dict(type='data', array=df[time_err_col]),
        error_y=dict(type='data', array=df[mag_err_col])
    )
    fig.add_trace(go.Scatter(
        x=t_fit_plot,
        y=y_fit,
        mode='lines',
        name='Weighted Sine Fit',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
    legend=dict(
        x=0,             # left side
        y=0.9,             # bottom
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.6)',  # optional background for readability
        bordercolor='gray',
        borderwidth=1
        )
    )
    fig.show()
    # return fig, popt


##---------------------------------------------------------------------------------------------------------------------------
#           Downloader
#---------------------------------------------------------------------------------------------------------------------------

class Downloader:
    """
    Class for downloading ZTF FITS photometric measurements from IRSA/IPAC database using metadata from provided CSV file.
    """

    def __init__(self, csv_file: str, path_to_save_fits: str, max_workers: int = 5, verbose: bool = False):
        """
        Parameters:
        -----------
        csv_file : str
            Path to the CSV file containing metadata (filefracday, field, ccdid, etc.)
        path_to_save_fits : str
            Directory to save downloaded FITS files.
        max_workers : int
            Number of parallel threads for downloading.
        verbose : bool
            Whether to print per-file download status messages.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.save_path = path_to_save_fits
        self.max_workers = max_workers
        self.verbose = verbose
        os.makedirs(self.save_path, exist_ok=True)

        # Configure logger
        logging.basicConfig(level=logging.INFO,  # always allow INFO and above
                    format='%(asctime)s - %(levelname)s - %(message)s')

        # logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
        #                     format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _build_filename(self, row):
        field = str(row['field']).zfill(6)
        ccdid = str(row['ccdid']).zfill(2)
        filefracday = str(row['filefracday'])
        filtercode = str(row['filtercode'])
        imgtypecode = str(row['imgtypecode'])
        qid = str(row['qid'])
        return f"ztf_{filefracday}_{field}_{filtercode}_c{ccdid}_{imgtypecode}_q{qid}_psfcat.fits"

    def _build_url(self, filefracday, filename):
        return f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{filefracday[:4]}/{filefracday[4:6]}{filefracday[6:8]}/{filefracday[8:]}/{filename}"

    def _download_file(self, idx, row):
        filename = self._build_filename(row)
        full_path = os.path.join(self.save_path, filename)
    
        # Build URL and download command
        filefracday = str(row['filefracday'])
        url = self._build_url(filefracday, filename)
        cmd = f"wget -q -O {full_path} {url}"
    
        # Run download
        result = subprocess.run(cmd, shell=True)
    
        # Check file validity
        if result.returncode == 0 and os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            return f"✓ Downloaded: {filename}" if self.verbose else None
        else:
            if os.path.exists(full_path):
                os.remove(full_path)
            if self.verbose:
                self.logger.warning(f"[{idx}] Download failed or file is empty: {filename}")
            return f"✗ Failed: {filename}" if self.verbose else None



    def download_fits(self):
        """Downloads missing FITS files in parallel."""
        self.logger.info("Starting download of FITS files...")
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for idx, row in self.csv_file.iterrows():
                futures.append(executor.submit(self._download_file, idx, row))

            for future in as_completed(futures):
                msg = future.result()
                if msg:  # Print only if message exists (based on verbosity)
                    print(msg)

        self.logger.info("Download complete.")


# --------------------------------------------------
# Body class
# --------------------------------------------------
class Body:
    """
    Represents a solar system object and manages the retrieval and interpolation
    of its ephemerides using JPL Horizons via the astroquery interface.
    
    Supports local caching, MPC-style name formatting, and scipy cubic spline interpolation
    of ephemeris values including light-time corrected Julian Dates.
    """

    def __init__(self, name, ephem_start_date, ephem_end_date, step='4h', location='I41', id_type='smallbody', use_cache=True):
        """
        Initializes a Body object and loads ephemerides from cache or queries JPL Horizons if needed.
        
        Parameters:
            name (str): Provisional designation or name of the solar system body (e.g., '2002 MS4', 'Quaoar').
            ephem_start_date (str): Start date of the ephemeris (e.g., '2020-01-01').
            ephem_end_date (str): End date of the ephemeris (e.g., '2020-12-31').
            step (str): Time resolution for the ephemeris (e.g., '4h').
            location (str): Observer location code (e.g., 'I41' for ZTF location).
            id_type (str): Horizons ID type (usually 'smallbody' for asteroids).
            use_cache (bool): If True, will attempt to load ephemeris from disk if already cached.
        """

        self.name = name
        self.ephem_start_date = ephem_start_date
        self.ephem_end_date = ephem_end_date
        self.step = step
        self.location = location
        self.id_type = id_type
        self.ephemeris = None
        
        if use_cache and self._ephemeris_cache_exists():
            self.ephemeris = self._load_ephemeris_cache()
        else:
            self.ephemeris = self.fetch_ephemerides()
            self._save_ephemeris_cache()
            
        self.strip_units_from_ephemeris()
        
    def _ephemeris_cache_path(self):
        """
        Generates a file path for storing or loading cached ephemerides.
        
        Returns:
            str: Path to the ephemeris cache file.
        """
        cache_name = f"ephem_{self.name.replace(' ', '_')}_{self.ephem_start_date}_{self.ephem_end_date}.pkl"
        return os.path.join(".ephem_cache", cache_name)

    def _ephemeris_cache_exists(self):
        """
        Checks whether a cached ephemeris file already exists on disk.
        
        Returns:
            bool: True if cache file is found, False otherwise.
        """
        return os.path.exists(self._ephemeris_cache_path())

    def _save_ephemeris_cache(self):
        """
        Saves the currently loaded ephemeris to a local cache file.
        Creates the cache directory if it does not exist.
        """
        os.makedirs(".ephem_cache", exist_ok=True)
        with open(self._ephemeris_cache_path(), "wb") as f:
            pickle.dump(self.ephemeris, f)

    def _load_ephemeris_cache(self):
        """
        Loads a previously cached ephemeris from disk.
        
        Returns:
            astropy.table.Table: The ephemeris table previously saved.
        """
        with open(self._ephemeris_cache_path(), "rb") as f:
            return pickle.load(f)

    def _format_target(self, target):
        """
        Formats the target name to insert a space between the year and letter code
        in MPC-style provisional designations (e.g., '2002MS4' → '2002 MS4').
        
        Parameters:
            target (str): The name or designation of the object.
        
        Returns:
            str: The formatted name compatible with JPL Horizons.
        """
        return f"{target[:4]} {target[4:]}" if re.match(r'^\d{4}[A-Za-z]', target) else target

    def fetch_ephemerides(self):
        """
        Queries JPL Horizons via astroquery to retrieve the ephemerides for the object.
        
        Returns:
            astropy.table.Table: Ephemeris table with columns such as RA, DEC, alpha, r, delta, V, lighttime.
        """
        target = self._format_target(self.name)
        obj = Horizons(id=target, 
                       location=self.location, 
                       epochs={'start': self.ephem_start_date, 'stop': self.ephem_end_date, 'step': self.step},
                       id_type=self.id_type
                      )
        return obj.ephemerides(extra_precision=True)
        
    def strip_units_from_ephemeris(self):
        """
        Removes units from all numeric columns in the ephemeris table in-place.
        This ensures all columns contain plain floats or strings.
        """
        for col in self.ephemeris.colnames:
            if hasattr(self.ephemeris[col], 'unit'):
                self.ephemeris[col] = self.ephemeris[col].value


    def get_ephemeris_at(self, epoch, require_keys=None, lighttime=None):
        """
        Interpolates the ephemeris to a specific epoch using scipy cubic splines.
        Returns physical quantities (RA, DEC, etc.) with correct units.

        If light-time correction is not available in the Horizons table, the user may supply a fallback value.
    
        Parameters:
            epoch (str or astropy.time.Time): The epoch at which to evaluate the ephemeris.
            require_keys (list, optional): List of keys (e.g., 'RA', 'alpha') that must be present in the ephemeris.
                                           If missing, raises ValueError.
            lighttime (float, Quantity, or array-like, optional): One-way light-time (in minutes).
                Used only if 'lighttime' is not present in the ephemeris.
        
        Returns:
            dict: A dictionary with interpolated ephemeris values (floats) 
                  and 'JD_LT_corr' as an astropy Time object.
        """
        jd = Time(epoch).jd
        eph = self.ephemeris
        available_keys = set(eph.colnames)
        
        if require_keys:
            missing = [k for k in require_keys if k not in available_keys]
            if missing:
                raise ValueError(f"The following required keys are missing from the ephemeris: {missing}")
        
        default_keys = ['RA', 'DEC', 'alpha', 'delta', 'r', 'lighttime', 'V']
        keys = [k for k in default_keys if k in available_keys]
        ephemeris_at_epoch = {}
        
        for key in keys:
            ephemeris_at_epoch[key] = CubicSpline(eph['datetime_jd'], eph[key])(jd)
            # ephemeris_at_epoch[key] = float(value)  # Return plain float

        # Handle lighttime correction
        if 'lighttime' in ephemeris_at_epoch:
            lighttime_day = ephemeris_at_epoch['lighttime']/1440
            ephemeris_at_epoch['JD_LT_corr'] = jd - lighttime_day
        elif lighttime is not None:
            try:
                lt = u.Quantity(lighttime, u.min).to(u.day).value
                logging.warning("Lighttime not available in ephemeris; using user-supplied lighttime for correction.")
                ephemeris_at_epoch['JD_LT_corr'] = Time(jd - lt, format='jd')
            except Exception as e:
                logging.error(f"Failed to interpret user-supplied lighttime: {e}. Falling back to uncorrected JD.")
                ephemeris_at_epoch['JD_LT_corr'] = jd
        
        else:
            logging.warning(
                "One-way light travel time ('lighttime') not available in ephemeris and no user value provided. "
                "Using uncorrected JD. Consider supplying a float or array via `lighttime=`."
            )
            ephemeris_at_epoch['JD_LT_corr'] = jd
    
        return ephemeris_at_epoch
    
    def __repr__(self):
        """
        Returns a human-readable string representation of the Body object.
        
        Returns:
            str: Summary of object name and ephemeris date range.
        """
        return f"<Body: {self.name} from {self.ephem_start_date} to {self.ephem_end_date}>"
# --------------------------------------------------
# Measurements class
# --------------------------------------------------
class Measurements:
    """
    Handles photometric measurements from FITS files. Performs quality filtering,
    calibrates instrumental magnitudes using color terms, and selects the closest
    PSF sources to expected positions based on RA/DEC and V magnitude.
    """
    def __init__(self, fits_path, colors=None, color_errs=None):
        """
        Initializes the Measurements object by loading and calibrating a FITS photometry table.
        
        Parameters:
            fits_path (str): Path to the input FITS file.
            colors (dict): Dictionary mapping photometric color keys (e.g., 'gr') to values.
            color_errs (dict): Dictionary mapping photometric color keys to their associated uncertainties.
        """
        self.colors = colors or {}
        self.color_errs = color_errs or {}
        self.filter_code = None
        self.table = None
        self.mag_cal_table = None
        self.header = None
        
        self.table, self.header = self.load_fits(fits_path)
        self.mag_cal_table = self.filter_photometry(self.table)
        self.calibrate_magnitudes()

    def _get_color_and_error(self, pcolor_key):
        """
        Retrieves the user-provided color index and its uncertainty for the given photometric color key.
        
        Parameters:
            pcolor_key (str): Color key as found in the FITS header (e.g., 'gr', 'g-r').
        
        Returns:
            tuple: (color value, color uncertainty)
        
        Raises:
            ValueError: If the required color key is not found in either the color or error dictionary.
        """
        key = pcolor_key.lower().replace(' ', '')
        if key not in self.colors or key not in self.color_errs:
            raise ValueError(f"Color or color error not provided for '{key}'.")
        return self.colors[key], self.color_errs[key]

    def load_fits(self, path,verbose=False):
        """
        Loads the photometric catalog and header from a FITS file.
        
        Parameters:
            path (str): Path to the FITS file.
        
        Returns:
            tuple: (Astropy Table of sources, FITS header object)
        """
        with fits.open(path) as hdul:
            data = Table(hdul[1].data)
            header = hdul[0].header
            if verbose:
                logging.info(f"Loaded {len(data)} sources from {path}")
        return data, header

    def filter_photometry(self, data,verbose=False):
        """
        Filters raw photometric sources based on quality flags and parameters.
        
        Criteria:
            - flags == 0
            - -50 < sharp < 50
            - chi < 5
        
        Parameters:
            data (astropy.table.Table): The raw input photometry table.
        
        Returns:
            astropy.table.Table: Filtered photometry table.
        """
        mask = (
            (data['flags'] == 0) &
            (data['sharp'] > -50) &
            (data['sharp'] < 50) &
            (data['chi'] < 5)
        )
        filtered = data[mask]
        if verbose:
            logging.info(f"Filtered to {len(filtered)} sources based on flags, sharpness, and chi.")
        return filtered

    def calibrate_magnitudes(self,verbose=False):
        """
        Calibrates instrumental magnitudes using zero point, color term, and user-provided color values.
        
        This function uses header keywords from the FITS file and computes the final magnitudes in the PAN-STARSS 1 photometric system
        and their uncertainties using standard error propagation equations.
        
        Adds the following columns to `self.mag_cal_table`:
            - mag_cal
            - mag_cal_err
            - mjd
            - filter_code
        
        Logs:
            - Success or failure, including missing header keys.
        """
        try:
            mag_zero = self.header['MAGZP']
            mag_zero_sigma = self.header['MAGZPUNC']
            pcolor = self.header['PCOLOR']
            pcolor_key = pcolor.lower().replace(' ', '')
            epoch = Time(self.header['OBSMJD'], format='mjd')
            cff = self.header['CLRCOEFF']
            cff_err = self.header['CLRCOUNC']
            filter_code = self.header['FILTERID']

            mag_inst = self.mag_cal_table['mag']
            mag_inst_err = self.mag_cal_table['sigmag']
            color, color_err = self._get_color_and_error(pcolor_key)

            self.mag_cal_table['mag_cal'] = mag_inst + mag_zero + (cff * color)
            err_inst = mag_inst_err**2
            err_zp = mag_zero_sigma**2
            err_color_term = (color * cff_err)**2
            err_color_val = (cff * color_err)**2
            self.mag_cal_table['mag_cal_err'] = np.sqrt(err_inst + err_zp + err_color_term + err_color_val)
            self.mag_cal_table['mjd'] = np.full(len(self.mag_cal_table), epoch.mjd)
            self.mag_cal_table['filter_code'] = np.full(len(self.mag_cal_table), int(filter_code))
            if verbose:
                logging.info("Photometric calibration applied successfully.")
            
        except KeyError as e:
            logging.error(f"Calibration failed: missing FITS header keyword '{e.args[0]}'.")
        except Exception as e:
            logging.error(f"Unexpected error during calibration: {e}")

    def _get_separations(self, ra_expected, dec_expected):
        """
        Internal helper method to compute angular separations (in milliarcseconds) between
        each catalog source and an expected RA/DEC position.
        
        Parameters:
            ra_expected (float or Quantity): Right Ascension of the expected target.
            dec_expected (float or Quantity): Declination of the expected target.
        
        Returns:
            tuple:
                - Quantity array of separations in milliarcseconds.
                - SkyCoord of the target position.
        """
        ra_expected = ra_expected if isinstance(ra_expected, u.Quantity) else ra_expected * u.deg
        dec_expected = dec_expected if isinstance(dec_expected, u.Quantity) else dec_expected * u.deg
    
        catalog_coords = SkyCoord(ra=self.mag_cal_table['ra'] * u.deg,
                                  dec=self.mag_cal_table['dec'] * u.deg)
        target_coord = SkyCoord(ra=ra_expected, dec=dec_expected)
        separations = catalog_coords.separation(target_coord).to_value(u.mas)  # <- now it's a float array
        return separations, target_coord


    def select_closest_sources(self, ra_expected, dec_expected, n=10):
        """
        Selects the N closest catalog sources to the expected RA/DEC position.
        
        Parameters:
            ra_expected (float or Quantity): Expected RA of the target (deg or Quantity).
            dec_expected (float or Quantity): Expected DEC of the target (deg or Quantity).
            n (int): Number of closest sources to return (default: 10).
        
        Returns:
            astropy.table.Table: Subset of the magnitude calibrated table with the closest sources,
                                 including a new column named 'separation (mas)'.
        """
        if self.mag_cal_table is None:
            logging.warning("No calibrated table available. Run calibrate_magnitudes() first.")
            return None    
        separations, _ = self._get_separations(ra_expected, dec_expected)
        self.mag_cal_table['separation (mas)'] = separations
        closest_idx = np.argsort(separations)[:n]
        return self.mag_cal_table[closest_idx]
        
    def select_best_source_within_vrange(self, ra_expected, dec_expected, v_expected, v_tol, n=10,verbose=False):
        """
        Selects the closest source (by separation) among the N nearest that also falls within
        a specified V-magnitude range of an expected value.
        
        Parameters:
            ra_expected (float or Quantity): Expected RA of the target.
            dec_expected (float or Quantity): Expected DEC of the target.
            v_expected (float): Expected apparent magnitude of the target.
            v_tol (float): Tolerance for acceptable magnitude difference from v_expected.
            n (int, optional): Number of closest sources to return when calling select_closest_sources method (default: 10).
        
        Returns:
            astropy.table.Row or None: The best matching source, or None if no match is found.
        """
        if self.mag_cal_table is None or 'mag_cal' not in self.mag_cal_table.colnames:
            logging.error("Calibrated magnitudes not available. Cannot select best source.")
            return None
        # Reuse top-N closest sources
        closest_sources = self.select_closest_sources(ra_expected, dec_expected, n=n)
        if closest_sources is None:
            return None
        
        filtered = [row for row in closest_sources if abs(row['mag_cal'] - v_expected) <= v_tol]

        if not filtered:
            if verbose:
                self.logger.info("No source within the expected V magnitude range.")
            return None

        # Now recompute separation for filtered list only
        catalog_coords = SkyCoord(ra=[row['ra'] for row in filtered] * u.deg,
                                  dec=[row['dec'] for row in filtered] * u.deg)
        _, target_coord = self._get_separations(ra_expected, dec_expected)
        separations = catalog_coords.separation(target_coord)
        closest_idx = np.argmin(separations)
        return filtered[closest_idx]

# --------------------------------------------------
# MultiPhotometryAnalysis class
# --------------------------------------------------
class MultiPhotometryAnalysis:
    def __init__(self, fits_files, body, colors=None, color_errs=None, v_tol=0.5):
        """
        Initialize the class with FITS files, the body for ephemerides, colors, error values, and v_tol.
        """
        self.fits_files = fits_files
        self.body = body
        self.colors = colors or {}
        self.color_errs = color_errs or {}
        self.v_tol = v_tol
        self.reduced_table = None
        self.filter_code = None

        # Set up logging configuration here
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        # Suppress astroquery INFO logs
        logging.getLogger("astroquery").setLevel(logging.WARNING)
    def __repr__(self):
        if self.reduced_table:
            filters = set(self.reduced_table['filter_code'])
            return f"<MultiPhotometryAnalysis with {len(self.fits_files)} FITS files, filters: {filters}>"
        return f"<MultiPhotometryAnalysis with {len(self.fits_files)} FITS files>"

    def summary(self):
        if self.reduced_table is None:
            print("No reduced table available.")
            return
        print(f"Total reduced rows: {len(self.reduced_table)}")
        for f in set(self.reduced_table['filter_code']):
            n = sum(self.reduced_table['filter_code'] == f)
            print(f" - {f} filter: {n} rows")

    def _log(self, message, level="info", verbose=True):
        if verbose:
            getattr(self.logger, level)(message)

    def _get_source_entry(self, fname):
        meas = Measurements(fits_path=fname, colors=self.colors, color_errs=self.color_errs)
        mjd = meas.header['OBSMJD']
        eph = self.body.get_ephemeris_at(Time(mjd, format='mjd'), require_keys=['RA', 'DEC', 'alpha', 'delta', 'r', 'lighttime', 'V'])
        row = meas.select_best_source_within_vrange(eph['RA'], eph['DEC'], eph['V'], self.v_tol)
        if not row:
            return None

        reduced_mag = row['mag_cal'] - 5 * np.log10(eph['r'] * eph['delta'])
        return {
            'jd_lt_corr': eph['JD_LT_corr'],
            'ra': eph['RA'],
            'dec': eph['DEC'],
            'ref_mag': eph['V'],
            'phase_angle': eph['alpha'],
            'mag_cal': row['mag_cal'],
            'mag_cal_err': row['mag_cal_err'],
            'reduced_mag': reduced_mag,
            'filter_code': row['filter_code'],
            'separation (mas)': row['separation (mas)'],
            'filename': os.path.basename(fname),
        }
        
    def run_analysis(self, check_close_stars=False, separation_lim=4000, verbose=False):
        """
        Process a list of FITS photometric files and produce a table of reduced magnitudes.
        
        """
        self._log("Starting analysis...", verbose=verbose)
        results = []

        for fname in self.fits_files:
            try:
                entry = self._get_source_entry(fname)
                if entry:
                    results.append(entry)
            except Exception as e:
                self._log(f"Error processing {fname}: {e}", level="error", verbose=verbose)
                continue

        self.reduced_table = Table(rows=results, 
            names=['jd_lt_corr', 'ra', 'dec', 'ref_mag', 'phase_angle', 'mag_cal', 
                   'mag_cal_err', 'reduced_mag', 'filter_code', 'separation (mas)', 'filename'])

        filter_map = {1: 'g', 2: 'r', 3: 'i'}
        self.reduced_table['filter_code'] = [filter_map.get(code, 'unknown') for code in self.reduced_table['filter_code']]

        grouped = self.reduced_table.group_by('filter_code')
        filtered_rows = []

        for group in grouped.groups:
            median_mag_cal_err = np.median(group['mag_cal_err'])
            threshold1 = 5 * median_mag_cal_err
            threshold2 = separation_lim
            valid_rows = group[(group['mag_cal_err'] <= threshold1) & (group['separation (mas)'] <= threshold2)]
            filtered_rows.append(valid_rows)

        filtered_table = Table(np.concatenate(filtered_rows))
        self.reduced_table = filtered_table
        print(f"Filtered table has {len(self.reduced_table)} rows after filtering.")

        if check_close_stars:
            output_filename = str(self.fits_files[0].split(os.sep)[0] + '/gaia_closeby_stars.fits')
            self._log("Checking for close by stars...", verbose=verbose)
            secondary_table = self.query_gaia_per_source_bp(self.reduced_table, output_filename=output_filename)
            self.reduced_table = self.close_by_stars_filter(self.reduced_table, secondary_table)

        self._log("Analysis complete.", verbose=verbose)

    def save_table(self, path='reduced_table.csv'):
        """
        Saves the reduced photometry table to a CSV file.

        Parameters:
            path (str): Output path to save the table.
        """
        if self.reduced_table is None:
            self._log("No reduced table to save.", level="warning")
            return
        self.reduced_table.write(path, format='csv', overwrite=True)
        self._log(f"Saved reduced table to {path}")


    def query_gaia_per_source_bp(self, astropy_table, output_filename, save_every=100,
                                 search_radius_arcsec=8, n_threads=5,verbose=False):
        """
        Queries Gaia DR3 for nearby stars around each target RA/DEC to identify contamination risks.
        
        Resumes automatically if a partial results file exists. Saves final table locally as a .fits file.
        
        Parameters:
            astropy_table (Table): Input table with 'ra' and 'dec' columns.
            output_filename (str): Path to save final Gaia result table.
            save_every (int): Save progress every N queries (default: 100).
            search_radius_arcsec (float): Cone search radius in arcseconds (default: 8").
            n_threads (int): Number of parallel threads for querying (default: 5, max recommended: 10).
            verbose=False
        
        Returns:
            Table or None: Gaia sources near input positions, or None if no matches.
        """

        ########################################################################################################
        # Check if the final output file already exists
        if os.path.exists(output_filename):
            self._log(f"Found existing Gaia stars final file: {output_filename}\t Skipping Gaia query and loading existing results...", verbose=True)
            results = Table.read(output_filename)
            return results
        else:
            # Set the partial file name
            partial_filename = output_filename.replace('.fits', '') + '_partial.fits'
            # Check if the partial file exists
            if os.path.exists(partial_filename):
                self._log(f"Found existing partial file: {partial_filename}. Resuming...", verbose=True)
                # Read the existing results from the partial file
                results = Table.read(partial_filename)
                done_indices = set(results['index'])
            else:
                # If no partial file, create an empty results table
                results = Table(names=('index', 'input_ra', 'input_dec', 'source_id', 'ra', 'dec'),
                                dtype=('i8', 'f8', 'f8', 'i8', 'f8', 'f8'))
                done_indices = set()
            ########################################################################################################
            # Convert search radius from arcseconds to degrees (SkyCoord uses degrees)
            search_radius = search_radius_arcsec * u.arcsec
        
            def query_one(index, ra, dec):
                coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
                query = f"""
                    SELECT source_id, ra, dec
                    FROM gaiadr3.gaia_source
                    WHERE CONTAINS(POINT('ICRS', ra, dec),
                                   CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {search_radius.to(u.deg).value})) = 1
                """
                job = Gaia.launch_job_async(query, dump_to_file=False)
                r = job.get_results()
                if len(r) > 0:
                    r['index'] = index
                    r['input_ra'] = ra
                    r['input_dec'] = dec
                    return r
                else:
                    return None
                
            tasks = []
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                for idx, row in enumerate(astropy_table):
                    # Skip rows that have already been processed (using done_indices set)
                    if idx in done_indices:
                        self._log(f"Skipping index {idx}, already processed.",verbose=verbose)
                        continue
                    ra, dec = row['ra'], row['dec']
                    tasks.append(executor.submit(query_one, idx, ra, dec))
    
                futures_completed = 0
                for future in as_completed(tasks):
                    result = future.result()
                    futures_completed += 1
                    if result is not None:
                        results = vstack([results, result])
    
                    # Save partial results every 'save_every' queries
                    if futures_completed % save_every == 0:
                        self._log(f"Saving partial results after {futures_completed} queries...",verbose=verbose)
                        results.write(partial_filename, overwrite=True)
    
            self._log(f"Gaia closeby stars file has {len(results)} total rows.",verbose=verbose)
            results.write(output_filename, overwrite=True)
    
            # Remove the partial file after final save
            if os.path.exists(partial_filename):
                os.remove(partial_filename)
    
            logging.info(f"Done. Results saved to {output_filename}")
        # Return None if no results were found
        return results if len(results) > 0 else None
    
    def close_by_stars_filter(self, main_table, secondary_table, max_sep_arcsec=4,verbose=False):
        """
        Filters out rows in the main photometry table that have a Gaia DR3 source closer than a threshold.
        
        Uses Gaia DR3 matches from `query_gaia_per_source_bp`.
        
        Parameters:
            main_table (Table): Table of reduced photometry.
            secondary_table (Table): Table of close Gaia stars.
            max_sep_arcsec (float): Maximum separation allowed before a row is rejected (default is 4").
            verbose = False
        
        Returns:
            Table: Filtered table with nearby contaminants removed.
        """
        separations = []
        # Loop through each row in the table and calculate the distance of the expected position to the star.
        for row in secondary_table:
            # Create SkyCoord objects for both coordinate pairs
            coord1 = SkyCoord(ra=row['input_ra']*u.deg, dec=row['input_dec']*u.deg)
            coord2 = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg)
            separation = coord1.separation(coord2).to(u.arcsec).value
            separations.append(round(separation, 2))
        
        secondary_table['distance (arcsec)'] = separations
        self._log(f"Removing rows where a Gaia star is ≤ {max_sep_arcsec}\" from the asteroid's expected position.", verbose=True)
        
        rows_to_remove = []
        for i, _ in enumerate(main_table):
            matches = secondary_table[secondary_table['index'] == i]
            if len(matches) > 0:
                min_distance = np.min(matches['distance (arcsec)'])
                if min_distance <= max_sep_arcsec:
                    rows_to_remove.append(i)

        
        # for i, row in enumerate(main_table):
        #     if i in secondary_table['index']:
        #         distance = secondary_table[secondary_table['index'] == i]['distance (arcsec)'][0]
        #         if distance <= max_sep_arcsec:
        #             to_remove.append(i)

        # Remove the marked rows from the main table
        main_table.remove_rows(rows_to_remove)
        return main_table
        
    
    
    
    def fit_phase_curve(self, model='linear', filter_code='r', H=None, G=None, B=None, phase_bin=0.5, mag_col='auto',table=None):
        """
        Fits a phase curve to the reduced or detrended magnitudes and optionally corrects the magnitudes 
        by subtracting the model from the original data. The method can apply both a model fit 
        to the original data and a model fit to binned data (using phase angle bins of 0.5 deg). It also 
        provides the option to correct magnitudes based on a user-defined fit model.
    
        Parameters:
        -----------
        model : str, optional, default='linear'
            The type of model to use for fitting. Can be one of the following:
            - 'linear': A linear model fit, H + G * alpha.
            - 'shevchenko': A Shevchenko model fit, H + G * alpha + B / (1 + alpha).
        filter_code : str, optional, default='r'
            The filter code to select data from `reduced_table`. The phase curve is fitted 
            only to data with this filter code.
        H, G, B : float or None
            Phase curve parameters. If None, fit will optimize them.
        phase_bin: float, optional, default=0.5
            The bin size, in degrees, for the binned phase curve model.
        mag_col : str, default='auto'
            Column name to fit. If 'auto', uses 'mag_corr_by_rot' if available, otherwise 'reduced_mag'.
        table : astropy.table.Table or None
            Input table. If not provided, uses self.reduced_table.
    
        Returns:
        --------
        tuple or astropy.table.Table
            - If H, G, and B are not provided, the function returns:
                - popt: Optimized parameters from the fit to the original data.
                - fit_label: String describing the fit parameters for the original data.
                - popt2: Optimized parameters from the fit to the binned data.
                - fit_label_binned: String describing the fit parameters for the binned data.
            
            - If H, G, and B are provided, the function returns:
                - filter_data: The updated `reduced_table` for the chosen filter with a new column, 'phase_corrected_mag', 
                  that contains the magnitude corrections (original data - binned model fit).
    
        Raises:
        -------
        RuntimeError
            If `reduced_table` is None, an analysis has to be run first.
            
        ValueError
            If an unsupported model type is provided (other than 'linear' or 'shevchenko').
    
        Notes:
        ------
        - The function first filters the `reduced_table` by the given filter code (`filter_code`).
        - If `H`, `G`, and `B` are provided, it performs a fit **only** on the binned data, 
          and subtracts the binned model from the original reduced magnitudes.
        - If `H`, `G`, and `B` are not provided, the function performs fits both on the 
          original data and the binned data and returns the fit results for both.
        - The binned fit uses a median value per 0.5 degree bin of phase angle.
        """
        if table:
            self.reduced_table = table
        
        if self.reduced_table is None:
            raise RuntimeError("Run analysis before fitting phase curve or pass table explicitly.")            
        
        # Detect best column to use if mag_col='auto'
        if mag_col == 'auto':
            if 'mag_corr_by_rot' in self.reduced_table.colnames:
                mag_col = 'mag_corr_by_rot'
            elif 'reduced_mag' in self.reduced_table.colnames:
                mag_col = 'reduced_mag'
            else:
                raise ValueError("No valid magnitude column found ('mag_corr_by_rot' or 'reduced_mag').")
        # Filter the data by the provided filter code
        msk = (self.reduced_table['filter_code'] == filter_code)
        filter_data = self.reduced_table[msk]    

        if len(filter_data) == 0:
            raise ValueError(f"No data found for filter '{filter_code}'.")

        if mag_col not in self.reduced_table.colnames:
            raise ValueError(f"Column '{mag_col}' not found in self.table.")
        
        x = filter_data['phase_angle']
        y = filter_data[mag_col]
        #########################
        # Binned data
        phase_bins = np.arange(min(x), max(x) + phase_bin, phase_bin)
        median_mag_per_bin = []
        phase_bin_centers = []
        
        for bin_start in phase_bins:
            bin_end = bin_start + phase_bin
            bin_mask = (x >= bin_start) & (x < bin_end)
            if np.any(bin_mask):  # Only calculate the median if there are points in this bin
                median_mag = np.median(y[bin_mask])
                phase_bin_centers.append(np.mean([bin_start, bin_end]))
                median_mag_per_bin.append(median_mag)
        
        # Define the fitting model based on the selected type ('linear' or 'shevchenko')
        if model == 'linear':
            def model_function(alpha, H, G):
                return H + G * alpha
        elif model == 'shevchenko':
            def model_function(alpha, H, G, B):
                return H + G * alpha + B / (1 + alpha)
        else:
            raise ValueError("Unsupported model type. Choose between 'linear' or 'shevchenko'.")
        
        if H is None and G is None and B is None:
            # Fit both original and binned data
            popt, _ = curve_fit(model_function, x, y)
            fit_label = f"Original: H = {popt[0]:.2f}, G={popt[1]:.2f}" if model == 'linear' else f"Fit: H = {popt[0]:.2f}, G = {popt[1]:.2f}, B = {popt[2]:.2f}"
            
            popt2, _ = curve_fit(model_function, phase_bin_centers, median_mag_per_bin)
            fit_label_binned = f"Binned: H = {popt2[0]:.2f}, G = {popt2[1]:.2f}" if model == 'linear' else f"Bin Fit: H = {popt2[0]:.2f}, G = {popt2[1]:.2f}, B={popt2[2]:.2f}"
    
            return popt, fit_label, popt2, fit_label_binned
        else:
            # Use user-defined parameters and subtract model
            popt2 = [H,G] if model == 'linear' else [H,G,B] 
            fit_label_binned = f"Original data linear fit: H = {popt2[0]:.2f}, G = {popt2[1]:.2f}" if model == 'linear' else f"Original data shev fit: H={popt2[0]:.2f}, G={popt2[1]:.2f}, B={popt2[2]:.2f}"
            #####################
            # Subtract the model from the original reduced magnitude (using binned fit)
            model_mag = model_function(x, *popt2)  # Apply the binned fit to the original data
            filter_data['phase_corrected_mag'] = y - model_mag
            
            # Optionally, print the result for the user
            print(f"Fit result for {filter_code} filter using original data and {model} model:")
            print(fit_label_binned)
            
            # Return the modified table with the 'phase_corrected_mag' column
            return filter_data


#---------------------------------------------------------------------------------------------------------------------------
#                                             RotationalAnalysis
#---------------------------------------------------------------------------------------------------------------------------
class RotationalAnalysis:
    def __init__(self, table, P_ref: float = 8, P_ref_delta: float = None, 
                 n_term: int = 1, f_min: float = 0.5, f_max: float = 12, P_dec: int = 5):
        """
        Class to perform Lomb-Scargle analysis on rotational data.
        
        Parameters
        ----------
        table: astropy.table.Table
            The table containing the data to analyze.
        
        save_path: str
            Path to save the plots.
        
        P_ref: float, optional
            The reference period for the periodogram plot, default is 8 hours.
        
        P_ref_delta: float, optional
            The delta from the reference period to adjust frequency limits. If not provided, 
            `f_min` and `f_max` will be used directly.
        
        n_term: int, optional
            The number of terms to be used in Lomb-Scargle fit, default is 1.
        
        f_min: float, optional
            The minimum frequency (cycles/day) to be used in the search, default is 0.5.
        
        f_max: float, optional
            The maximum frequency (cycles/day) to be used in the search, default is 12.0.
        
        P_dec: int, optional
            The number of decimals to display in the plot, default is 5.
        """
        self.table = table
        # self.path = save_path
        self.P_ref = P_ref
        self.P_ref_delta = P_ref_delta
        self.n_term = n_term
        self.P_dec = P_dec

        # Set f_min and f_max based on P_ref and P_ref_delta
        if P_ref_delta is not None:
            self.f_min = 24 / (P_ref - P_ref_delta)
            self.f_max = 24 / (P_ref + P_ref_delta)
        else:
            self.f_min = f_min
            self.f_max = f_max

    def lomb_scargle_fit(self, asteroid: str, filter_code: str, shev: str = 'No',legend = False, 
                         samples_per_peak=None, peaks = False,pn=4,use_magerr=True,
                         save_path=None,table=None,delta_t=None,center_data=True,
                         y_max=None,n_shuffle=None, use_window_function=False):
        """
        Performs a Lomb-Scargle periodogram analysis and generates a periodogram plot using the `Plots` class.
        
        Parameters:
            asteroid (str): Name of the asteroid to annotate the plot.
            filter_code (str): Filter code used for the dataset (e.g., 'r').
            shev (str): If 'yes', appends 'shev' to the plot filename. Default is 'No'.
            legend (bool): Whether to display a legend in the periodogram plot.
            peaks (bool): Whether to highlight peaks in the periodogram.
            pn (int): Number of peaks to annotate, if `peaks=True`.
            use_magerr (bool): Whether to include magnitude errors in the LS fit.
            save_path (str): Output path for saving the periodogram plot.
            table (astropy.table.Table): If provided, overrides self.table for analysis.
            center_data: bool, optional (default is True).
        
        Returns:
            int: Number of frequencies evaluated in the periodogram.
        """

        if table is not None:
            self.table = table
        
        JD = self.table['jd_lt_corr']
        mag = self.table['phase_corrected_mag']
        mag_err = self.table['mag_cal_err'] if use_magerr else None
    
        JD = np.asarray(JD).astype(float)
        mag = np.asarray(mag).astype(float)
        if mag_err is not None:
            mag_err = np.asarray(mag_err).astype(float)

        ls = LombScargle(JD, mag, mag_err if use_magerr else None, nterms=self.n_term, center_data=center_data)
        autopower_kwargs = dict(minimum_frequency=self.f_min, maximum_frequency=self.f_max)
        if samples_per_peak:
            autopower_kwargs['samples_per_peak'] = samples_per_peak

        frequency, power = ls.autopower(**autopower_kwargs)
        self.best_freq = frequency[np.argmax(power)]
            
        # Optional: compute false alarm level
        false_alarm_shuffle = None
        if n_shuffle:
            false_alarm_shuffle = self.compute_false_alarm_from_shuffle(
                JD=JD, 
                mags=mag, 
                mag_err=mag_err if use_magerr else None, 
                n_shuffle=n_shuffle,
                samples_per_peak=samples_per_peak, 
                center_data=center_data)    

        # Optional: compute window function overlay
        window_overlay = None
        if use_window_function:
            freq_win, power_win = self.compute_window_periodogram(
                JD=JD, nterms=1, samples_per_peak=samples_per_peak, center_data=center_data)
            window_overlay = (freq_win, power_win)

        # Generate periodogram plot
        plots = Plots(table=self.table, ast_name=asteroid, sep_limit=4000, fontsize=18)
        if save_path is None:
            save_path = f'{asteroid}_periodogram_{round(24 / self.best_freq, 0)}.jpg'
        
        plots.periodogram(
            asteroid=asteroid,
            filter_code=filter_code,
            frequencia=frequency,
            potencia=power,
            best_freq=self.best_freq,
            legend=legend,
            peaks=peaks,
            pn=pn,
            n_term=self.n_term,
            P_ref=self.P_ref,
            delta_t=delta_t,
            y_max=y_max,
            save_path=save_path,
            false_alarm_shuffle=false_alarm_shuffle,
            window_overlay=window_overlay
        )
        if use_window_function:
            result = [frequency,power,freq_win, power_win]
        else:
            result = [frequency,power]
        return result

    def compute_window_periodogram(self, JD, nterms=1, samples_per_peak=None, center_data=True):
        """
        Computes Lomb-Scargle periodogram of the window function by setting all magnitudes to 1.
        """
        mags = np.ones_like(JD)
        ls = LombScargle(JD, mags, nterms=nterms, center_data=center_data)
        if samples_per_peak is not None:
            frequency, power = ls.autopower(minimum_frequency=self.f_min, maximum_frequency=self.f_max,
                                            samples_per_peak=samples_per_peak)
            power = power/np.max(power)
        else:
            frequency, power = ls.autopower(minimum_frequency=self.f_min, maximum_frequency=self.f_max)
            power = power/np.max(power)
        return frequency, power
    
    def compute_false_alarm_from_shuffle(self, JD, mags, mag_err=None, n_shuffle=1000, samples_per_peak=None, center_data=True):
        """
        Estimate a false alarm level using randomized magnitude shuffling.
        
        Parameters
        ----------
        JD : np.ndarray
            Time values (Julian Date).
        mags : np.ndarray
            Magnitude values.
        mag_err : np.ndarray or None
            Magnitude errors (optional).
        n_shuffle : int
            Number of shuffling iterations.
        samples_per_peak : int or None
            Passed to LombScargle.autopower().
        center_data : bool
            Whether to center the data in LS.
        
        Returns
        -------
        float
            The 99% false alarm power level.
        """
        peak_powers = []
        for _ in range(n_shuffle):
            shuffled_mags = np.random.permutation(mags)
            if mag_err is not None:
                ls = LombScargle(JD, shuffled_mags, mag_err, nterms=self.n_term, center_data=center_data)
            else:
                ls = LombScargle(JD, shuffled_mags, nterms=self.n_term, center_data=center_data)
            
            if samples_per_peak is not None:
                _, power = ls.autopower(minimum_frequency=self.f_min, maximum_frequency=self.f_max,
                                        samples_per_peak=samples_per_peak)
            else:
                _, power = ls.autopower(minimum_frequency=self.f_min, maximum_frequency=self.f_max)

            peak_powers.append(np.max(power))
        return np.percentile(peak_powers, 99)



    def compute_phased_model(self, JD: np.ndarray, mag: np.ndarray, best_freq: float, epoch0=None):
        """
        Compute the phased lightcurve model without time centering.
        This uses the previously determined best_freq.
        
        Parameters
        ----------
        JD : np.ndarray
            Julian Dates of observations.
        mag : np.ndarray
            Corrected magnitudes to model.
        best_freq : float
            Frequency in cycles/day.
        epoch0 : float or None
            JD to use as phase zero. If None, JD.mean() is used.
        
        Returns
        -------
        list : [phased_model_x, phased_model_y, P, amplitude, std, best_freq]
        """
        # Force JD to plain float array
        if hasattr(JD, 'value'):
            JD = JD.value
        JD = np.asarray(JD)

        # Use phase-zero epoch
        t0 = epoch0 if epoch0 is not None else JD.mean()
    
        # Fit the model using uncentered time
        ls = LombScargle(JD, mag, nterms=self.n_term, center_data=False)
        
        t_fit = np.linspace(JD.min(), JD.max(), len(JD))
        y_fit = ls.model(t_fit, best_freq)
    
        phase_model = ((t_fit - t0) * best_freq) % 1
        sorted_idx = np.argsort(phase_model)
        phase_model_sorted = phase_model[sorted_idx]
        y_fit_sorted = y_fit[sorted_idx]
    
        amp = y_fit.max() - y_fit.min()
        std = np.std(mag - ls.model(JD, best_freq), ddof=1)
        P = np.round(24 / best_freq, decimals=self.P_dec)
    
        self.best_freq = best_freq
        self.amplitude = amp
        self.std_model_residuals = std
        self.epoch0_used = t0
        self.ls_model = ls
        return [phase_model_sorted, y_fit_sorted, P, amp, std, best_freq]




    def get_detrended_magnitudes(self, table=None, jd_column='jd_lt_corr', mag_column='reduced_mag'):
        """
        Compute magnitudes corrected for rotational modulation using the stored Lomb-Scargle model.
    
        Stores the result as a new column 'mag_corr_by_rot' and returns the updated table.
        Also stores the updated table in `self.corrected_table` for downstream use.
    
        Parameters
        ----------
        table : astropy.table.Table, optional
            Table containing the photometric data. If None, uses self.table.
        jd_column : str
            Name of the column in the table containing corrected Julian Dates.
        mag_column : str
            Name of the column containing the reduced magnitudes to detrend.
    
        Returns
        -------
        astropy.table.Table
            Table with 'jd_used', 'mag_original', 'mag_corr_by_rot' columns.
        """
        if self.ls_model is None or self.best_freq is None:
            raise ValueError("Lomb-Scargle model or best frequency not available.")
    
        if table is None:
            if self.table is None:
                raise ValueError("No table provided and self.table is None.")
            table = self.table
    
        if jd_column not in table.colnames or mag_column not in table.colnames:
            raise ValueError(f"Required column(s) '{jd_column}' or '{mag_column}' not found in table.")
    
        jd = table[jd_column]
        if hasattr(jd, 'unit'):
            jd = jd.value  # Convert to raw float array if Quantity
        mag = table[mag_column]
        if hasattr(mag, 'unit'):
            mag = mag.value  # Convert to raw float array if Quantity
    
        # Evaluate the Lomb-Scargle model and subtract it
        rotation_model = self.ls_model.model(jd, self.best_freq)
        corrected_mag = mag - rotation_model
    
        # Insert columns into the table (safely, replacing if they exist)
        table['jd_used'] = jd
        table['mag_original'] = mag
        table['mag_corr_by_rot'] = corrected_mag        
    
        # Store and return
        # self.corrected_table = table
        return table

    @staticmethod
    def per_orbit_analysis(asteroid: str, table, nterms: list, P_ref: float,
                           f_min: float, f_max: float, save_path: str,
                           delta_t: float, samples_per_peak: int,
                           n_shuffle: int = None, use_window_function: bool = False):
        """
        Perform per-orbit rotational analysis on a list of orbit-specific photometric tables.
        """
        rot = {}
        tab = {}
        folded_models = {}
        frequency = {}
        power = {}
        freq_win = {}      
        power_win = {}     
    
        plots = Plots(table=table, ast_name=asteroid, sep_limit=4000, fontsize=18)
    
        for i, (orbit_table, n) in enumerate(zip(table, nterms), start=1):
            key = f"orbit{i}"
            filename1 = f"{save_path}_orbit{i}_corrected_by_phase.jpg"
            filename2 = f"{save_path}_orbit{i}_individual_periodogram.jpg"
            filename3 = f"{save_path}_orbit{i}_lightcurve_folded.jpg"
    
            # Scatter of corrected magnitudes vs time
            plots.general(
                x_column='jd_lt_corr',
                y_column='phase_corrected_mag',
                show_yerr=True,
                show_color=True,
                color_column='filter_code',
                table=orbit_table,
                save_path=filename1
            )
    
            # Lomb-Scargle fit
            rot[key] = RotationalAnalysis(
                table=orbit_table,
                P_ref=P_ref,
                f_min=f_min,
                f_max=f_max,
                n_term=n
            )
    
            if use_window_function:
                f, p, f_win_, p_win_ = rot[key].lomb_scargle_fit(
                    asteroid=asteroid,
                    filter_code=None,
                    use_magerr=True,
                    save_path=filename2,
                    delta_t=delta_t,
                    legend=True,
                    samples_per_peak=samples_per_peak,
                    use_window_function=use_window_function,
                    n_shuffle=n_shuffle
                )
                freq_win[key] = f_win_
                power_win[key] = p_win_
            else:
                f, p = rot[key].lomb_scargle_fit(
                    asteroid=asteroid,
                    filter_code=None,
                    use_magerr=True,
                    save_path=filename2,
                    delta_t=delta_t,
                    legend=True,
                    samples_per_peak=samples_per_peak,
                    use_window_function=use_window_function,
                    n_shuffle=n_shuffle
                )
    
            frequency[key] = f
            power[key] = p
    
            # Model generation
            model_params = rot[key].compute_phased_model(
                JD=np.array(orbit_table['jd_lt_corr']),
                mag=np.array(orbit_table['phase_corrected_mag']),
                best_freq=rot[key].best_freq
            )
    
            # Folded lightcurve plot + return DataFrame with data + model
            df = plots.folded_lightcurve(
                asteroid=asteroid,
                table=orbit_table,
                filter_codes=np.array(orbit_table['filter_code']),
                JD=np.array(orbit_table['jd_lt_corr']),
                corrected_mag=np.array(orbit_table['phase_corrected_mag']),
                mag_err=np.array(orbit_table['mag_cal_err']),
                best_freq=rot[key].best_freq,
                model_params=model_params,
                save_path=filename3,
                shev=None,
                show_fig=False
            )
    
            folded_models[key] = df
            tab[key] = rot[key].get_detrended_magnitudes()
    
        # Return results
        if use_window_function:
            return rot, tab, folded_models, frequency, power, freq_win, power_win
        else:
            return rot, tab, folded_models, frequency, power




##---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   PLOTS
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Plots:
    def __init__(self,table,ast_name:str,filter_code=None,sep_limit:int=4000,fontsize:int=18, n_term=1, P_ref=8):
        """
        Initializes the Plots class for plotting asteroid photometric and rotational analysis results.
        
        Parameters
        ----------
        table : astropy.table.Table
            The input data table to plot.
        ast_name : str
            The asteroid's name (used in plot titles).
        filter_code : str or None
            Optional filter to apply before plotting.
        sep_limit : int
            Separation threshold (in mas) to filter sources. Default is 4000.
        fontsize : int
            Base font size for plot labels.
        n_term : int
            Number of terms used in the Lomb-Scargle fit (affects periodogram).
        P_ref : float
            Reference rotation period in hours (used for vertical line in periodogram).
        """
        self.table = table
        self.ast_name = ast_name
        self.filter_code = filter_code
        self.sep_limit = sep_limit
        self.fontsize = fontsize
        self.n_term = n_term
        self.P_ref = P_ref

    def format_axis_title(self, column_name):
        """
        Formats a column name by replacing underscores with spaces and capitalizing each word.
    
        Example:
            'mag_corr_by_rot' → 'Mag Corr By Rot'
        """
        return column_name.replace('_', ' ').title()

    def filter_table_by(self, column_name, value, operation):
        """
        Filters the internal table using a comparison operation on a specified column.
        
        Parameters
        ----------
        column_name : str
            The column to filter.
        value : numeric
            Value to compare with.
        operation : str
            One of: '==', '!=', '<', '>', '<=', '>='.
        
        Returns
        -------
        astropy.table.Table
            Filtered table.
        """
        ops = {
            '==': operator.eq,
            '<': operator.lt,
            '>': operator.gt,
            '<=': operator.le,
            '>=': operator.ge,
            '!=': operator.ne
        }

        # Check if operation is valid
        if operation not in ops:
            raise ValueError(f"Operation '{operation}' not supported")

        # Apply the mask using the operator function
        msk = ops[operation](self.table[column_name], value)

        # Filter and return the table
        return self.table[msk]


    def general(self,x_column:str='jd_lt_corr',y_column:str='reduced_mag', table=None, 
                filter_code=None, show_yerr=False, err_column:str='mag_cal_err', 
                show_color=False, color_column ='separation (mas)',save_path=None):
        """
        Creates a general scatter plot with optional error bars and color coding.
        
        Parameters
        ----------
        x_column : str
            Column to use on the x-axis.
        y_column : str
            Column to use on the y-axis.
        table : astropy.table.Table or None
            Optional external table to use instead of self.table.
        filter_code : str or None
            Filter code to select specific data.
        show_yerr : bool
            Whether to plot vertical error bars.
        err_column : str
            Column for y-axis errors (default: 'mag_cal_err').
        show_color : bool
            Whether to color-code by a data column.
        color_column : str
            Column used for color coding.
        save_path : str or None
            File path to save the figure.
        
        Returns
        -------
        plotly.graph_objects.Figure
        """
        if table is not None:
            filter_data = table
        else:
            filter_data = self.filter_table_by('separation (mas)',4000,'<')
            
        if filter_code is not None:
            filter_data = filter_data[filter_data['filter_code'] == filter_code]
        # Validate columns
        for col, condition in [(x_column, True), (y_column, True),(err_column, show_yerr), (color_column, show_color)]:
            if condition and col not in filter_data.colnames:
                raise ValueError(f"Column '{col}' not found in the filtered data.")
        
        if isinstance(filter_data[x_column][0], Time):
            filter_data[x_column] = filter_data[x_column].jd
            
        x = filter_data[x_column]
        y = filter_data[y_column]
        # Optional parameters
        y_err = filter_data[err_column] if show_yerr else None
        color = color_column if show_color else None

        # color = filter_data[color_column] if show_color else None

        fig = px.scatter(
            filter_data,
            x=x,
            y=y,
            color=color
        )

        # Add error bars if yerr is true
        if show_yerr:
            fig.update_traces(error_y=dict(type='data', array=y_err))

        # Format title and axis labels
        title_text = f"{self.ast_name} - {filter_code if filter_code else 'all'} - {len(filter_data)} points"
        layout_params = {
            'title': {'text': title_text, 'y': 0.95, 'x': 0.5},
            'font': dict(family="serif", size=self.fontsize, color="Black"),
            'xaxis_title': self.format_axis_title(x_column),
            'yaxis_title': self.format_axis_title(y_column),
        }
        fig.update_layout(layout_params)
        fig.update_yaxes(automargin=True, autorange="reversed")

        if save_path:
            fig.write_image(save_path, width=1500, height=500)
            print(f"Figure saved to {save_path}")
        return fig

    def phase_curve(self,popt,fit_label,popt2,fit_label_binned,x_column='phase_angle',
                    y_column='reduced_mag',model='linear',filter_code=None,
                    save_path=None,show_yerr=False,show_color=False,show_fig = False,table=None):
        """
        Plots a phase curve with overlaid linear or Shevchenko model fits.
        
        Parameters
        ----------
        popt : list
            Parameters from fit to original data.
        fit_label : str
            Label for the original model fit.
        popt2 : list
            Parameters from binned data fit.
        fit_label_binned : str
            Label for the binned model fit.
        x_column : str
            Column representing phase angle.
        y_column : str
            Magnitude column to use.
        model : str
            'linear' or 'shevchenko'.
        filter_code : str or None
            Filter code to select data.
        save_path : str or None
            Optional output file path.
        show_yerr : bool
            Whether to display error bars.
        show_color : bool
            Whether to color code points.
        show_fig : bool 
            Whether to display the figure in jupyter or not.
        table : astropy.table.Table or None
            Optional table override.
        
        Returns
        -------
        None or plotly.graph_objects.Figure
        """

        # Use provided table or fallback to filtered self.table
        if table is not None:
            filter_data = table
        else:
            filter_data = self.filter_table_by('separation (mas)', 4000, '<')
    
        # # Apply filter_code if specified
        # if filter_code is not None:
        #     filter_data = filter_data[filter_data['filter_code'] == filter_code]
    
        # Generate base plot with selected columns and options
        fig = self.general(
            x_column=x_column,
            y_column=y_column,
            table=filter_data,
            filter_code=filter_code,  # Already filtered above
            show_yerr=show_yerr,
            show_color=show_color,
            save_path=None  # We'll handle saving at the end
        )
    
        # Proceed with model overlays
        try:
            phase = filter_data[x_column]
            alpha_fit = np.linspace(phase.min(), phase.max(), 200)
    
            if model == 'linear':
                mag_fit = popt[0] + popt[1] * alpha_fit
                mag_fit_binned = popt2[0] + popt2[1] * alpha_fit
            else:
                mag_fit = popt[0] + popt[1] * alpha_fit + popt[2] / (1 + alpha_fit)
                mag_fit_binned = popt2[0] + popt2[1] * alpha_fit + popt2[2] / (1 + alpha_fit)
    
            # Add model fits
            fig.add_scatter(
                x=alpha_fit,
                y=mag_fit,
                mode='lines',
                name=fit_label,
                line=dict(dash='dot', color='green')
            )
    
            fig.add_scatter(
                x=alpha_fit,
                y=mag_fit_binned,
                mode='lines',
                name=fit_label_binned,
                line=dict(dash='solid', color='red')
            )
    
            fig.update_layout(
                legend=dict(
                    x=0.05,
                    y=0.05,
                    xanchor='left',
                    yanchor='bottom'
                )
            )
    
            if save_path:
                fig.write_image(save_path, width=1500, height=500)
                print(f"Figure saved to {save_path}")
            if show_fig:
                fig.show()
            # return fig
    
        except Exception as e:
            print(f"Failed to fit phase curve: {e}")
            return None

    def periodogram(self,asteroid: str, frequencia: np.ndarray, potencia: np.ndarray, 
                    best_freq: float,legend:bool,peaks:bool,pn:int=4,save_path=None,
                    filter_code = None, n_term=1, P_ref=8, delta_t=None,y_max=None,
                    false_alarm_shuffle=None, window_overlay=None):
        """
        Generate and save the periodogram plot, optionally overlaying a scaled window function
        and a shuffle-based false alarm level.
        
        Parameters
        ----------
        asteroid: str
            The name of the asteroid.
        
        filter_code: str, default = None
            The filter code.
        
        frequencia: np.ndarray
            Array of frequencies.
        
        potencia: np.ndarray
            Array of power values.
        
        best_freq: float
            The frequency with the maximum power.
        
        year: str
            The year of data acquisition.
        """
        FSS = 15
        fig_width, fig_height = 9, 4

        if y_max is None:
            y_max = 1.0  # Default y-limit if not explicitly set


        plt.figure(figsize=(fig_width + 1, fig_height))
        # Plot real LS periodogram
        plt.plot(frequencia, potencia, color='c', label='Data')
        # Plot window function, scaled to match real periodogram peak
        if window_overlay is not None:
            freq_win, power_win = window_overlay
            plt.plot(freq_win, power_win, color='gray', linestyle=':', linewidth=1.5, label='Obs window effects')
        
        if filter_code:
            plt.title(f'{asteroid} - {filter_code} - {n_term} terms', fontsize=FSS)
        else:
            plt.title(f'{asteroid} - all - {n_term} terms', fontsize=FSS)
        plt.xlabel('Frequency (cycles/day)', fontsize=FSS)
        plt.ylabel('Power LS', fontsize=FSS)
        plt.axvline(24 / P_ref, color='k', linestyle='-', lw=2, label=f'Publ = {P_ref} h')
        if delta_t:
            plt.axvline(24/((24/best_freq)+delta_t), color='r', linestyle=':', lw=2, label=f'Limits = {delta_t} h')
            plt.axvline(24/((24/best_freq)-delta_t), color='r', linestyle=':', lw=2)
        plt.plot(best_freq, potencia.max(), 'ro', label=f'Peak = {24 / best_freq:.6f} h')
        
        if false_alarm_shuffle:
            plt.axhline(false_alarm_shuffle, color='orange', linestyle='--', lw=2, label='99% false alarm (shuffle)')

        if int(n_term) == 1:
            # Provide a list of probabilities (e.g., [0.001]) for false alarm level
            corrected_mag = self.table['phase_corrected_mag']
            JD = self.table['jd_lt_corr']
            if 'mag_cal_err' in self.table.columns:
                mag_err = self.table['mag_cal_err']
            else:
                mag_err = None
            ls = LombScargle(JD, corrected_mag, mag_err, nterms=n_term)

        # Highlight top N peaks
        if peaks:
            from scipy.signal import find_peaks
            peak_indices, _ = find_peaks(potencia)
            peak_heights = potencia[peak_indices]
            top_indices = peak_indices[np.argsort(peak_heights)[-pn:][::-1]]
            for i, idx in enumerate(top_indices):
                x, y = frequencia[idx], potencia[idx]
                plt.text(x + (fig_width / 300), y - (fig_height / 300), str(i + 1),
                         fontsize=FSS - 2, color='black', ha='center', va='bottom')
                print(f"{i+1}: Period = {24/x:.6f} hours, Power = {y:.6f}")
        
        if legend == True:
            plt.legend(fontsize=FSS - 5, loc='upper left', bbox_to_anchor=(1, 1))
            
        plt.xticks(fontsize=FSS)
        plt.yticks(fontsize=FSS)
        plt.ylim(0,y_max)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='jpg')
            plt.close()
        plt.show()

    def folded_lightcurve(self, model_params, asteroid: str, filter_codes: np.ndarray, 
                      JD: np.ndarray, corrected_mag: np.ndarray, mag_err: np.ndarray, 
                      best_freq: float, shev: str = None, save_path=None, table=None, 
                      epoch_zero=None,show_fig=False):
        """
        Generate and save the phased lightcurve plot using Plotly Express.
        
        Parameters
        ----------
        model_params : list
            Output of compute_phased_model() → [phased_x, model_y, P, amp, std, best_freq]
        asteroid : str
            Asteroid name to label the plot.
        filter_codes : np.ndarray
            Filter code for each observation.
        JD : np.ndarray
            Julian Dates of observations.
        corrected_mag : np.ndarray
            Corrected magnitudes (e.g., phase_corrected_mag).
        mag_err : np.ndarray
            Associated uncertainties.
        best_freq : float
            Frequency from LS fit (cycles/day).
        shev : str
            Tag to label Shevchenko vs linear. Default None.
        save_path : str or None
            Path to save the figure.
        table : astropy.table.Table or None
            Optional table to get N.
        epoch_zero : float or None
            JD to use as phase zero. If None, JD.mean() is used.
        show_fig: bool
            Whether show the figure in jupyter or not (default is False)
        Returns
        -------
        pandas.DataFrame
            DataFrame with sorted folded data and model points.
        """
        # Unpack model
        phase_model, y_model, P, amp, std, best_freq = model_params
    
        # Define phase origin
        epoch0 = epoch_zero if epoch_zero is not None else JD.mean()
        phase_data = ((JD - epoch0) * best_freq) % 1
    
        # DataFrame for data points
        df = pd.DataFrame({
            'JD': JD,
            'Rotational phase': phase_data,
            'Magnitude': corrected_mag,
            'Magnitude_err': mag_err,
            'filter_code': filter_codes
        })
    
        fig = go.Figure()
    
        # Color map
        unique_filters = df['filter_code'].unique()
        color_map = {f: px.colors.qualitative.Set1[i % 9] for i, f in enumerate(unique_filters)}
    
        # Add scatter points by filter
        for filt in unique_filters:
            filt_df = df[df['filter_code'] == filt]
            fig.add_trace(go.Scatter(
                x=filt_df['Rotational phase'],
                y=filt_df['Magnitude'],
                mode='markers',
                name=filt,
                error_y=dict(array=filt_df['Magnitude_err'], visible=True),
                marker=dict(color=color_map[filt], size=6),
            ))
    
        # Add model last (on top)
        fig.add_trace(go.Scatter(
            x=phase_model,
            y=y_model,
            mode='lines',
            line=dict(color='black', width=2),
            name='LS best model',
        ))
    
        # Title & Layout
        N = len(table) if table else len(self.table)
        title = f"{asteroid} - {N} points - P = {P} hours"
        fig.update_layout(
            title=title,
            xaxis_title='Rotational Phase',
            yaxis_title='Magnitude',
            width=900,
            height=450,
            font=dict(size=18),
            legend_title_text='Filter'
        )
        fig.update_yaxes(autorange='reversed')
    
        # Annotations
        fig.add_annotation(
            text=f'std = {std:.2f} mag',
            x=0.7, y=0.1, xref='paper', yref='paper',
            showarrow=False, font=dict(size=16)
        )
        fig.add_annotation(
            text=f'amp = {amp:.2f} mag',
            x=0.7, y=0.2, xref='paper', yref='paper',
            showarrow=False, font=dict(size=16)
        )
    
        # Save
        if save_path:
            fig.write_image(save_path, width=1500, height=500)
        if show_fig:
            fig.show()
    
        # Output
        df = df.sort_values(by='Rotational phase')
        df['model_phase'] = phase_model
        df['model_mag'] = y_model
        return df

        
    @staticmethod
    def plot_folded_lightcurve_with_residuals(df, save_path=None):
        """
        Plots the phased light curve and residuals using matplotlib.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Must contain 'Rotational phase', 'Magnitude', 'Magnitude_err', 'model_mag', and 'residuals'.
        save_path : str or None
            If provided, saves the figure to this path.
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9,4), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]}, constrained_layout=True)
    
        # --- Upper panel: lightcurve + model
        ax1.errorbar(df['Rotational phase'], df['Magnitude'], yerr=df['Magnitude_err'],
                     fmt='o', markersize=4, label='Data', alpha=0.6)
        ax1.plot(df['model_phase'], df['model_mag'], color='black', lw=2, label='Model')
        ax1.set_ylabel("Magnitude")
        ax1.invert_yaxis()
        ax1.legend()
    
        # --- Lower panel: residuals
        ax2.axhline(0, color='gray', lw=1, linestyle='--')
        ax2.plot(df['Rotational phase'], df['residuals'], 'o', markersize=3, color='darkred')
        ax2.set_ylabel("Residuals")
        ax2.set_xlabel("Rotational Phase")
        ax2.invert_yaxis()
        if save_path:
            plt.savefig(save_path, dpi=300)

    @staticmethod
    def plot_all_folded_orbits(folded_models, save_path, align_to='min', align_phase=0.5,fontsize=24):
        """
        Plot all phased data and models together from folded_models.
    
        Parameters
        ----------
        folded_models : dict
            Dictionary where each value is a DataFrame with columns:
            ['Rotational phase', 'Magnitude', 'Magnitude_err', 'filter_code',
             'model_phase', 'model_mag']
        align_to : {'min', 'max'} or None
            If 'min' or 'max', align all orbits based on model min/max.
            If None, no phase alignment applied.
        align_phase : float
            Phase to center the alignment (default = 0.5)
        """
        fig = go.Figure()
        palette = px.colors.qualitative.Set1[1:]  # or 'Dark24', 'Bold', etc
    
    
        for i, (orbit_name, df) in enumerate(folded_models.items()):
            color = palette[i % len(palette)]
            phase_data = df['Rotational phase'].to_numpy()
            model_phase = df['model_phase'].to_numpy()
            model_mag = df['model_mag'].to_numpy()
    
            # Optional alignment of model and data
            if align_to == 'min':
                shift = align_phase - model_phase[np.argmin(model_mag)]
            elif align_to == 'max':
                shift = align_phase - model_phase[np.argmax(model_mag)]
            else:
                shift = 0.0
    
            # Apply phase offset and wrap to [0,1]
            aligned_data_phase = (phase_data + shift) % 1
            aligned_model_phase = (model_phase + shift) % 1
    
            # Plot data
            fig.add_trace(go.Scatter(
                x=aligned_data_phase,
                y=df['Magnitude'],
                error_y=dict(type='data', array=df['Magnitude_err']),
                mode='markers',
                name=orbit_name,  # This will be the legend entry
                marker=dict(color=color, size=6),
                legendgroup=orbit_name,
                showlegend=True
            ))
    
            # Plot model
            # Sort model phase/mag together
            sorted_idx = np.argsort(aligned_model_phase)
            x_sorted = aligned_model_phase[sorted_idx]
            y_sorted = model_mag[sorted_idx]
            
            # Detect wrap point (where phase drops)
            jumps = np.where(np.diff(x_sorted) < -0.5)[0]  # large negative jump
            segments = np.split(np.column_stack([x_sorted, y_sorted]), jumps + 1)
            
            # Plot each segment as a separate line
            for j, seg in enumerate(segments):
                fig.add_trace(go.Scatter(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    mode='lines',
                    name=None,
                    line=dict(color=color, width=2),
                    legendgroup=orbit_name,
                    showlegend=False  # This hides the model line from legend
                ))
    
    
    
        fig.update_layout(
            # title="All Orbits: Folded Lightcurves and Model Fits",
            font=dict(family="serif", size=fontsize,color="black"),  # Global default font size

            xaxis_title="Rotational phase",       
            yaxis_title="Corrected magnitude",            
            xaxis=dict(
                range=[0, 1],
                title_font=dict(size=24, color="black"),
                tickfont=dict(size=20, color="black")
            ),
            yaxis=dict(
                autorange='reversed',
                title_font=dict(size=24, color="black"),
                tickfont=dict(size=20, color="black")
            ),
           
            legend=dict(
                font=dict(size=20, color="black"),
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255,255,255,0)',
                bordercolor='black',
                borderwidth=1
            ),

            
            margin=dict(l=50, r=50, t=20, b=50),  # Left, Right, Top, Bottom
            
            width=1200,
            height=500)
        fig.write_image(save_path+"_folded_curves.png", width=1200, height=500, scale=3)
        fig.show()


    
    @staticmethod
    def plot_all_periodograms(frequency,power,best_frequency,error,save_path,fontsize=24):
        '''
        Plots all periodograms into a single plot, useful when doing yearly analysis. 
    
        Parameters
        ----------
    
        frequency: dict
        A dictionary with keys being eacho orbit and values a list of frequencies.
    
        power: dict
        A dictionary with keys being eacho orbit and values a list of frequencies.
    
        best_frequency: float
        The reference frequency were to center the error bars (cycles/day).
    
        error: float
        Error bar in hours, to plot in the periodogram.
    
        save_path: str
        The path to save the file.
    
        Returns
        -------
    
        A single plot with all periodograms from the yearly analyses.
        '''
        # Use Set1 palette, skipping red
        palette = px.colors.qualitative.Set1[1:]
        
        # Create the figure
        fig = go.Figure()
        
        # Add each periodogram trace
        for i, key in enumerate(frequency):
            fig.add_trace(go.Scatter(
                x=frequency[key],
                y=power[key],
                mode='lines',
                name=key,
                line=dict(color=palette[i % len(palette)], width=2)
            ))
        
        # Add vertical lines at ±0.0042 offset from best_freq of orbit4
        P_center = 24 / best_frequency
        P_plus = 24 / (P_center + error)
        P_minus = 24 / (P_center - error)
        
        fig.add_vline(x=P_plus, line_dash="dot", line_color="gray", line_width=2)
        fig.add_vline(x=P_minus, line_dash="dot", line_color="gray", line_width=2)
        
        # Update layout
        fig.update_layout(
            margin=dict(l=50, r=50, t=20, b=50),  # Left, Right, Top, Bottom
            xaxis_title="Frequency (cycles/day)",
            yaxis_title="Power LS",
            font=dict(family="serif", size=fontsize,color="black"),  # Global default font size
        
            xaxis=dict(
                title_font=dict(size=24),
                tickfont=dict(size=20)
            ),
            yaxis=dict(
                title_font=dict(size=24),
                tickfont=dict(size=20)
            ),
            legend=dict(
                font=dict(size=20),
                bgcolor='rgba(255,255,255,0)',
                bordercolor='black',
                borderwidth=1
            ),
            width=1200,
            height=500
        )  
        fig.write_image(save_path+"_periodogram.png", width=1200, height=500, scale=3)
        fig.show()