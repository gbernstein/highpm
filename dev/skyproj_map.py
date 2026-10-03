"""Sky map of a per-position quantity in the McBryde-Thomas projection and
frame of Rubin RTN-011 Figure 7 (skyproj, lon_0=0, RA increasing left).

Input is a table (anything astropy reads; whitespace .txt with a '#' header
line works) with either --ra/--dec columns (binned into healpixels: counts,
or the per-pixel mean of --value) or a --pixel column of healpix RING
indices plus a --value column.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import skyproj
from astropy.table import Table

EXTENT = (-180, 180, -75, 40)  # reproduces RTN-011 Fig. 7's frame


def plot_skymap(ra=None, dec=None, values=None, pixels=None, nside=32, label="", out="skymap.png",
                extent=EXTENT, norm="linear", vmin=None, vmax=None, cmap="viridis", title=None,
                cbar_ticks=None, cbar_ticklabels=None, dpi=300, xsize=1000):
    """Draw raw (ra, dec[, values]) points binned at nside (counts, or mean of
    values per pixel), or sparse (pixels, values) healpix RING pixels, and save to out.
    xsize is the raster width skyproj resamples the map onto; raise it for high nside."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sp = skyproj.McBrydeSkyproj(ax=ax, extent=list(extent))
    kw = dict(nside=nside, xsize=xsize, zoom=False, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap)
    if pixels is not None:
        sp.draw_hpxpix(pixels=np.asarray(pixels), values=np.asarray(values, dtype="f8"), **kw)
    else:
        sp.draw_hpxbin(np.asarray(ra), np.asarray(dec), C=values, **kw)
    # skyproj default pad=0 butts the colorbar against the projection edge
    cbar = sp.draw_colorbar(label=label, ticks=cbar_ticks, pad=0.04, shrink=0.6)
    if cbar_ticklabels is not None:
        cbar.set_ticklabels(cbar_ticklabels)
    sp.ax.set_xlabel("Right Ascension", fontsize=14)
    sp.ax.set_ylabel("Declination", fontsize=14)
    if title:
        sp.ax.set_title(title, pad=25)  # clear the top RA tick labels
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _read_table(path):
    if path.endswith((".txt", ".dat")):
        return Table.read(path, format="ascii.commented_header")
    return Table.read(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", help="Input table (FITS, HDF5, CSV, or .txt with a '#' column-name header)")
    parser.add_argument("--ra", help="RA column (deg); binned with --dec")
    parser.add_argument("--dec", help="Dec column (deg)")
    parser.add_argument("--pixel", help="Healpix RING pixel column (instead of --ra/--dec)")
    parser.add_argument("--value", help="Value column (required with --pixel; with --ra/--dec, plot its per-pixel mean)")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--extent", type=float, nargs=4, default=EXTENT, metavar=("LON0", "LON1", "LAT0", "LAT1"))
    parser.add_argument("--norm", choices=["linear", "log"], default="linear")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--label", default=None, help="Colorbar label (default: value column, or 'count')")
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--xsize", type=int, default=1000, help="Raster width of the drawn map (raise for high nside)")
    parser.add_argument("-o", "--out", default="skymap.png")
    args = parser.parse_args()

    if (args.pixel is None) == (args.ra is None or args.dec is None):
        parser.error("give either --pixel or both --ra and --dec")
    if args.pixel is not None and args.value is None:
        parser.error("--pixel requires --value")

    tab = _read_table(args.table)
    values = None if args.value is None else np.asarray(tab[args.value], dtype="f8")
    label = args.label if args.label is not None else (args.value or "count")
    plot_skymap(
        ra=None if args.ra is None else tab[args.ra], dec=None if args.dec is None else tab[args.dec],
        values=values, pixels=None if args.pixel is None else tab[args.pixel], nside=args.nside,
        label=label, out=args.out, extent=args.extent, norm=args.norm, vmin=args.vmin, vmax=args.vmax,
        cmap=args.cmap, title=args.title, dpi=args.dpi, xsize=args.xsize,
    )
