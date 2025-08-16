from ._version import __version__, __version_info__

# Lightweight imports that are safe and often used
from .cat_reader import clean_cat, read_cat_data, read_cat_header
from .utils import arborist, detections_for_removal, filter_list, new_posvel

# Lazy attribute access for heavier modules or those that pull many deps.
# This avoids importing submodules unless actually accessed by users.


def __getattr__(name):
    if name in {"fast_movers"}:
        from .fast import fast_movers as _fast_movers

        return _fast_movers
    if name in {"output_fits", "output_fits_mask"}:
        from .fits_writer import output_fits as _output_fits
        from .fits_writer import output_fits_mask as _output_fits_mask

        return {"output_fits": _output_fits, "output_fits_mask": _output_fits_mask}[
            name
        ]
    if name in {"find_friend", "friends_of_friends"}:
        from .friends_of_friends import find_friend as _find_friend
        from .friends_of_friends import friends_of_friends as _friends_of_friends

        return {"find_friend": _find_friend, "friends_of_friends": _friends_of_friends}[
            name
        ]
    if name in {"gnomonic_plate2sky"}:
        from .gnomonic_plate2sky import gnomonic_plate2sky as _gnomonic_plate2sky

        return _gnomonic_plate2sky
    if name in {"new_modest_fitter", "new_modest_mover"}:
        from .modest import new_modest_fitter as _new_modest_fitter
        from .modest import new_modest_mover as _new_modest_mover

        return {
            "new_modest_fitter": _new_modest_fitter,
            "new_modest_mover": _new_modest_mover,
        }[name]
    if name in {"multi_fit5d", "multithreader"}:
        from .multithreader import multi_fit5d as _multi_fit5d
        from .multithreader import multithreader as _multithreader

        return {"multi_fit5d": _multi_fit5d, "multithreader": _multithreader}[name]
    if name in {"err2cov", "fit5d", "singleFit"}:
        from .pmfit import err2cov as _err2cov
        from .pmfit import fit5d as _fit5d
        from .pmfit import singleFit as _singleFit

        return {"err2cov": _err2cov, "fit5d": _fit5d, "singleFit": _singleFit}[name]
    raise AttributeError(name)
