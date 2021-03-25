#!/bin/bash
#example country_only.sh --countries russia --pixel_size_m 2000 --population_key lspop_2017_URCA_rural --max_travel_time 60
python setup.py install
python ppl_travel_coverage.py $@
