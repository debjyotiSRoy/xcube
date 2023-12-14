#!/bin/bash

output_file="disc_finetune_mimic3.txt"

command="./launches/launch_complete_mimic3 --fname mimic3_clas_complete_plant_flattune --plant --lin_sgdr_lr0 1e-1 --l2r_sgdr_lr0 1e-1 --diff_inattn 24 --no_running_decoder"
echo "Running: $command"
$command | tee -a "$output_file"

command="./launches/launch_complete_mimic3 --fname mimic3_clas_complete_plant_disctune --plant --lin_sgdr_lr0 1e-1 --l2r_sgdr_lr0 1e-3 --diff_inattn 24 --no_running_decoder"
echo "Running: $command"
$command | tee -a "$output_file"
