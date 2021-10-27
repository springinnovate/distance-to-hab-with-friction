#!/bin/bash
python setup.py install
for time in 60
do
    python ppl_travel_coverage.py --population_key population_2019 --max_travel_time $time "$@"
done
