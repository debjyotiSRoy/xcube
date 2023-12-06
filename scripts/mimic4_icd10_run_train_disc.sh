#!/bin/bash

output_file="disc_finetune.txt"

command="./launches/launch_full_mimic4_icd10_statefuldecoder --fname mimic4_icd10_clas_full_statefuldecoder --plant --diff_inattn 8 --l2r_sgdr_lr0 1e-1"
echo "Running: $command"
$command | tee -a "$output_file"

command="./launches/launch_full_mimic4_icd10_statefuldecoder --fname mimic4_icd10_clas_full_statefuldecoder --plant --diff_inattn 8 --l2r_sgdr_lr0 3e-1"
echo "Running: $command"
$command | tee -a "$output_file"
