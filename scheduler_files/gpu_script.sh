#!/bin/bash

out_file=$1
sleep_period=$2

rm -f $out_file

while true
do
    # date +"%T.%3N" >> $out_file
    nvidia-smi --query --display=MEMORY,PIDS | grep -E 'Process ID|Used GPU Memory' >> $out_file
    sleep $sleep_period
done