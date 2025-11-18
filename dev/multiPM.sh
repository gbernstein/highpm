for HEALPIX in 10791 10792 10898 10899 10900 10901 11001 11002 11003 \
       11004 11100 11101 11102 11103 11104 11196 11197 11198 \
       11199 11288 11289 11290 11376 
do
    echo "Processing healpix $HEALPIX"
    python gitrepos/highpm/scripts/PM.py \
    gitrepos/highpm/config/config.yaml \
    /cluster_scratch/users/vwetzell/RetII/RetIIDetections/cleaned_detections_*${HEALPIX}.fits \
    --output-name /cluster_scratch/users/vwetzell/RetII/PM/RetII_hp${HEALPIX}_movers.fits 
done