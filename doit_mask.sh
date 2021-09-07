#!/bin/bash
python setup.py install
for time in 60
do
    python mask_travel_coverage.py --max_travel_time $time --mask https://storage.googleapis.com/critical-natural-capital-ecoshards/optimization_results/A_90_md5_79f5e0d5d5029d90e8f10d5932da93ff.tif "$@"
done
