import argparse
import os
import re
import sys
from glob import glob

import numpy as np

DEFAULT_PATTERN = "gpr_*_*.fits"  # same convention as dev/exposuresToDo.py
DEFAULT_REGEX = r"(\d{7})(?=_[griz]\.fits$)"  # 7-digit expnum before _<band>.fits, same as dev/exposuresToDo.py


def scan_expnums(directory, pattern, regex):
    expnums = set()
    for f in glob(os.path.join(directory, pattern)):
        m = re.search(regex, os.path.basename(f))
        if m:
            expnums.add(int(m.group(1)))
    return expnums


def exposure_pixels(ra, dec, nside, radius_deg, hp):
    """List of healpixel arrays overlapping each exposure's footprint disc."""
    vecs = hp.ang2vec(ra, dec, lonlat=True)
    rad = np.radians(radius_deg)
    return [hp.query_disc(nside, v, rad, inclusive=True) for v in vecs]


def per_healpix_fraction(pixel_lists, found_mask, nside):
    """Vectorized per-pixel (total, found, fraction) via bincount.

    pixel_lists[i] = healpixels overlapping exposure i; an exposure counts
    toward every pixel it overlaps, not just the one containing its center.
    """
    npix = 12 * nside**2
    counts = np.fromiter((len(p) for p in pixel_lists), dtype=int, count=len(pixel_lists))
    all_pixels = np.concatenate(pixel_lists) if len(pixel_lists) else np.array([], dtype=int)
    all_found = np.repeat(np.asarray(found_mask), counts)
    total = np.bincount(all_pixels, minlength=npix)
    found = np.bincount(all_pixels[all_found], minlength=npix)
    fraction = np.divide(found, total, out=np.zeros_like(found, dtype=float), where=total > 0)
    pixels = np.nonzero(total)[0]
    return pixels, total[pixels], found[pixels], fraction[pixels]


def _self_test():
    pixel_lists = [np.array([1, 2]), np.array([1]), np.array([2, 3])]
    found_mask = np.array([True, True, False])
    pixels, total, found, fraction = per_healpix_fraction(pixel_lists, found_mask, nside=32)
    assert list(pixels) == [1, 2, 3]
    assert list(total) == [2, 2, 1]
    assert list(found) == [2, 1, 0]
    np.testing.assert_allclose(fraction, [1.0, 1 / 2, 0.0])
    print("self-test ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Per nside-32 healpixel, report what fraction of DELVE exposures "
            "(from the pointing table) are present in a directory of exposure files."
        )
    )
    parser.add_argument("--pointing-file", help="Path to delveExposures.hdf5 (astropy Table).")
    parser.add_argument("--dir", help="Directory to scan for exposures.")
    parser.add_argument(
        "--dir2",
        help="Optional second directory; only exposures found in both --dir and --dir2 count.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern for filenames, same convention as dev/exposuresToDo.py (default: '{DEFAULT_PATTERN}').",
    )
    parser.add_argument(
        "--regex",
        default=DEFAULT_REGEX,
        help=f"Regex whose first group is the expnum (default: '{DEFAULT_REGEX}').",
    )
    parser.add_argument("--nside", type=int, default=32, help="Healpix nside (default 32).")
    parser.add_argument(
        "--radius",
        type=float,
        default=1.1,
        help="Exposure footprint disc radius in degrees, ~DES focal plane radius (default 1.1).",
    )
    parser.add_argument(
        "--min-exposures",
        type=int,
        default=1,
        help="Disregard healpixels with fewer than this many table exposures (default 1).",
    )
    parser.add_argument("--output-file", help="Optional path to save per-healpixel counts as CSV.")
    parser.add_argument("--plot-file", help="Optional path to save a Mollweide map colored by fraction found.")
    parser.add_argument("--self-test", action="store_true", help="Run a self-check and exit.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        sys.exit(0)

    if not args.pointing_file or not args.dir:
        parser.error("--pointing-file and --dir are required")

    import healpy as hp
    from astropy.table import Table

    cat = Table.read(args.pointing_file, path="__astropy_table__")
    ra, dec = cat["pole"][:, 0], cat["pole"][:, 1]
    pixel_lists = exposure_pixels(ra, dec, args.nside, args.radius, hp)
    table_expnum = np.asarray(cat["expnum"])

    found_expnums = scan_expnums(args.dir, args.pattern, args.regex)
    if args.dir2:
        dir2_expnums = scan_expnums(args.dir2, args.pattern, args.regex)
        found_expnums &= dir2_expnums
        print(f"{len(found_expnums)} exposures found in both {args.dir} and {args.dir2}.")
    else:
        print(f"{len(found_expnums)} exposures found in {args.dir}.")
    found_mask = np.isin(table_expnum, list(found_expnums))
    print(f"{found_mask.sum()} matched the pointing table.")

    pixels, total, found, fraction = per_healpix_fraction(pixel_lists, found_mask, args.nside)
    keep = total >= args.min_exposures
    pixels, total, found, fraction = pixels[keep], total[keep], found[keep], fraction[keep]
    rows = [row for row in zip(pixels, total, found, fraction) if row[2] > 0]
    for p, t, f, frac in sorted(rows, key=lambda x: (x[3], x[1])):
        print(f"healpix {p:6d}: {f:5d}/{t:5d} ({frac:.1%})")

    if args.output_file:
        out = np.column_stack([pixels, total, found, fraction])
        np.savetxt(args.output_file, out, header="healpix total found fraction", fmt=["%d", "%d", "%d", "%.4f"])
        print(f"Saved to {args.output_file}")

    if args.plot_file:
        import matplotlib.pyplot as plt

        m = np.full(12 * args.nside**2, hp.UNSEEN)
        m[pixels] = fraction
        hp.mollview(m, min=0, max=1, unit="fraction found", title="DELVE exposure coverage fraction", cmap="viridis")
        plt.savefig(args.plot_file)
        print(f"Saved plot to {args.plot_file}")
