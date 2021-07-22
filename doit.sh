#!/bin/bash
python setup.py install
for time in 60 360
do
    python ppl_travel_coverage.py --pixel_size_m 2000 --population_key lspop_2017_URCA_rural --max_travel_time $time
    python ppl_travel_coverage.py --pixel_size_m 2000 --population_key lspop_2017_URCA_urban --max_travel_time $time
done
