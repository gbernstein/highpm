from functools import partial
from multiprocessing import Pool
import numpy as np
import tqdm

# =============================================================================
# Multithreading Functions
# =============================================================================


def multithreader(func, lol, cat, config, fitting=False):
    """Executes a function in parallel across multiple processes using a pool.

    Parameters
    ----------
    func : callable
        The function to apply to each element of `lol`. Must accept an element
        from `lol` as its first argument and `cat` as a keyword argument.
    lol : list
        List of elements to process in parallel.
    cat : any
        Additional argument to pass to `func` as a keyword argument.
    config : dict
        Configuration dictionary containing parameters for parallel execution,
        such as the number of cores and chunk size.
    fitting : bool, optional
        If True, the function is assumed to be a fitting function that requires
        additional parameters. If False, it is assumed to be a general function
        that only requires `cat` as a keyword argument.
    Returns
    -------
    ls_out : list
        List of results returned by applying `func` to each element in `lol`.

    Notes
    -----
    Uses `multiprocessing.Pool` for parallel execution and `tqdm` for progress
    display. The function is partially applied with the `cat` argument.
    """
    config_reqs = ["cores", "chunksize"]
    if np.any([key not in config for key in config_reqs]):
        raise ValueError(f"Missing required config keys: {config_reqs}")

    if fitting:
        partial_func = partial(func, cat=cat)
    else:
        partial_func = partial(func, cat=cat, config=config)

    with Pool(processes=config["cores"]) as pool:
        ls_out = list(
            tqdm.tqdm(
                pool.imap_unordered(partial_func, lol, chunksize=config["chunksize"]),
                total=len(lol),
            )
        )
    return ls_out


def multi_fit5d(fitter, detections_groups, cat, config):
    """Applies a 5D fitter to groups of detections using multithreading.

    Parameters
    ----------
    fitter : callable
        A function or callable object that performs fitting on a group of
        detections.
    detections_groups : iterable
        An iterable of detection groups to be processed by the fitter.
    cat : object
        Catalog or additional data required by the fitter.
    config : dict
        Configuration dictionary containing fitting parameters, including:
        - 'cores': int
            Number of CPU cores to use for multithreading.
        - 'chunksize': int
            Number of detection groups to process per thread chunk.
        - 'fitting': dict
            Dictionary containing fitting parameters:
            - 'time_sep': float
                Time separation threshold for fitting.
            - 'chisqClip': float
                Chi-squared clipping threshold for fitting.
            - 'parallax_prior': float
                Parallax prior value for fitting.
            - 'color_prior': float
                Color prior value for fitting.
            - 'colorFrac': float
                Color fraction for fitting.
            - 'pm_prior': float
                Proper motion prior value for fitting.
            - 'additional_error': bool
                whether or not to add additional error to be added to the fitting
                process.
    Returns
    -------
    np.ndarray
        An array of fit results, with failed fits (None) removed. The array has
        dtype=object.
    """
    config_reqs = [
        "cores",
        "chunksize",
        "time_sep",
        "chisqClip",
        "parallax_prior",
        "color_prior",
        "colorFrac",
        "pm_prior",
        "additional_error",
    ]
    if np.any(
        [key not in config and key not in config["fitting"] for key in config_reqs]
    ):
        raise ValueError(f"Missing required config keys: {config_reqs}")

    partial_fitter = partial(
        fitter,
        time_sep=config["fitting"]["time_sep"],
        chisqClip=config["fitting"]["chisqClip"],
        parallax_prior=config["fitting"]["parallax_prior"],
        color_prior=config["fitting"]["color_prior"],
        colorFrac=config["fitting"]["colorFrac"],
        pm_prior=config["fitting"]["pm_prior"],
        additional_error=config["fitting"]["additional_error"],
    )

    # Multithreaded application of a 5D fitter to groups of detections
    pm_list = multithreader(
        partial_fitter, detections_groups, cat, config, fitting=True
    )

    # Remove fits that return None
    clean_pm_list = [i for i in pm_list if i is not None]

    # Convert list to array and discard list
    clean_pm_arr = np.array(clean_pm_list, dtype=object)

    return clean_pm_arr