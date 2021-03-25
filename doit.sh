#!/bin/bash
python setup.py install
for time in 60 120 180 240 300 360 420 480 540 600
do
    python ppl_travel_coverage.py lspop_2017_URCA_rural $time
    python ppl_travel_coverage.py lspop_2017_URCA_urban $time
done
