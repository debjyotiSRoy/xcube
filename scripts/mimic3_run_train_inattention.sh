#!/bin/bash

output_file="inattention_mimic3_2.txt"

# for diff_inattn in 72 64 60 56 48 40 32 24 16 8 1; do
for diff_inattn in 16 8 1; do
  command="./launches/launch_22525_mimic3 --fname mimic3_clas_22525_plant_inattention --plant --lin_sgdr_lr0 1e-1 --l2r_sgdr_lr0 1e-2 --diff_inattn $diff_inattn --no_running_decoder " 
  echo "Running: $command"
  $command | tee -a "$output_file"
done