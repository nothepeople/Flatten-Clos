#!/bin/bash
# rm -rf ../deadlock

exps=(ecmp up-down edst)

for exp in ${exps[@]};
do
    output_dir="../deadlock/output/$exp/"
    traffic_dir="../deadlock/traffic/"
    topo_dir="../deadlock/topo/"
    trace_dir="../deadlock/trace/"
    config_dir="../deadlock/config/$exp/"
    mkdir -p $config_dir $output_dir $traffic_dir $topo_dir $trace_dir
done


# echo "Generating necessary files: e.g. CONFIG, TOPO, TRAFFIC..."
# python traffic.py
# python fc.py 
# python conf.py 