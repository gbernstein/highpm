for HEALPIX in 10791 10792 10898 10899 10900 10901 11001 11002 11003 \
       11004 11100 11101 11102 11103 11104 11196 11197 11198 \
       11199 11288 11289 11290 11376 
do
    echo "Processing healpix $HEALPIX"
    python ~/completeExposureClip.py \
    ./healpixDetections/cleaned_detections_hp${HEALPIX}.fits \
    ./RetIIDetections/cleaned_detections_hp${HEALPIX}.fits \
    53.94929166666667 \
    -54.04661111111111 \
    3.9
done