#!/bin/bash
python setup.py install
python ppl_travel_coverage.py lspop_2017_URCA_rural 60 --countries germany france
python ppl_travel_coverage.py lspop_2017_URCA_urban 60 --countries germany france
python ppl_travel_coverage.py lspop_2017_URCA_rural 120 --countries germany france
python ppl_travel_coverage.py lspop_2017_URCA_urban 120 --countries germany france
