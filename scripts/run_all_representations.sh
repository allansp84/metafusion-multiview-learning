#!/bin/bash

echo 'PWD:' `pwd`

# ./scripts/run_livdet17_combined.meta_classification.sh &
./scripts/run_livdet17_clarkson.meta_classification.sh
./scripts/run_livdet17_iiitwvu.meta_classification.sh
./scripts/run_livdet17_nd.meta_classification.sh
./scripts/run_livdet17_warsaw.meta_classification.sh

# ./scripts/run_livdet17_iiitwvu.bsif_representation.sh > logs/run_livdet17_iiitwvu.bsif_representation.log
# ./scripts/run_livdet17_combined.rawimage.sh > logs/run_livdet17_combined.rawimage.log
# ./scripts/run_livdet17_combined.bsif_representation.sh > logs/run_livdet17_combined.bsif_representation.log

