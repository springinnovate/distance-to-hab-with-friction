#!/bin/bash
python setup.py install
time=60
for mask in https://storage.googleapis.com/critical-natural-capital-ecoshards/optimization_results/A_90_md5_79f5e0d5d5029d90e8f10d5932da93ff.tif \
    https://storage.googleapis.com/critical-natural-capital-ecoshards/optimization_results/C_90_md5_bdf604015a7b1c7c78845ad716d568ef.tif \
    https://storage.googleapis.com/critical-natural-capital-ecoshards/habmasks/masked_all_nathab_wstreams_esa2015_nodata_WARPED_near_md5_d801fffb0e3fbfd8d7ffb508f18ebb7c.tif
do
    python mask_travel_coverage.py --max_travel_time $time --mask $mask "$@"
done
