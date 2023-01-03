#!/bin/bash
# the following must be performed with root privilege
# nvidia mps docs 5.1.1.1.
export CUDA_VISIBLE_DEVICES="0"
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS    # ** section 2.3.1.2 of doc pdf
                                        # When using MPS it is recommended to use EXCLUSIVE_PROCESS mode
                                        # -i = --id, -c = --compute-mode
nvidia-cuda-mps-control -d