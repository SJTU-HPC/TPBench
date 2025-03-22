#!/bin/bash

bash ./scripts/run_blas1.sh
bash ./scripts/run_roofline.sh
bash ./scripts/run_comp_comm.sh
bash ./scripts/run_blocked_stencil.sh
# bash ./scripts/run_rtriad.sh
# bash ./scripts/run_striad.sh

source ./scripts/machine_config.sh
python ./scripts/data_process/process_data.py ./result/${machine}
python ./scripts/data_process/csv_to_xlsx.py ./result/${machine}

