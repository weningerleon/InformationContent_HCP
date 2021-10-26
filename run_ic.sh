#!/bin/bash

# Wechsel in das Arbeitsverzeichnis, falls es nicht im Jobfile definiert wird

cd /home/staff/weninger/tmp/hcp_fmri/

# Ausführen des eigentlichen Programms
/work/scratch/weninger/conda/envs/hcp_fmri_env/bin/python -u run_ic_taskspecific.py