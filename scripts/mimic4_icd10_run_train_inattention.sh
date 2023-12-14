#!/bin/bash

output_file="inattention.txt"

for diff_inattn in 72 64 60 56 48 40 32 24 16 8 1; do
  command="./launches/launch_full_mimic4_icd10_inattention --fname mimic4_icd10_clas_full_plant_inattention --plant --diff_inattn $diff_inattn"
  echo "Running: $command"
  $command | tee -a "$output_file"
done