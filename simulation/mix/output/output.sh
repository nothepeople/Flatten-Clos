#!/bin/bash
# rm -rf fat-tree/ fc/ fc-ecmp/ fc-random/ Xpander/ Xpander-ecmp/ edst/ edst-random/ edst-lp/ fc-lp/

CCs=(dcqcn hpcc)
patterns=(cs-model)
retransmits=(pfc irn)

distributions=(web storage)
client_racks=(4 8 12 16 20 24 28 32)
server_racks=(4 8 12 16 20 24 28 32)

prefixes=(fc-ecmp fc-uniform fc-weighted fc-lp edst-uniform edst-weighted edst-lp)
sws=(64)

fractions=(0.04 0.10 0.20)
hot_traffics=(0.25 0.50 0.75)
# default bw is 1Gps and workload is 70
# dir for uniform mesh and fat-tree topology
for cc in ${CCs[@]};
do
    for retransmit in ${retransmits[@]};
    do
        for dis in ${distributions[@]};
        do
            for client in ${client_racks[@]};
            do
                for server in ${server_racks[@]};
                do
                    dir="fat-tree/$cc/cs-model/${retransmit}/$dis/C${client}-S${server}"
                    # echo $dir
                    mkdir -p $dir
                done
            done
        done
    done
done

## clos skew traffic dir
for cc in ${CCs[@]};
do
    for retransmit in ${retransmits[@]};
    do
        for dis in ${distributions[@]};
        do
            for fraction in ${fractions[@]};
            do
                for hot in ${hot_traffics[@]};
                do
                    mkdir -p "fat-tree/$cc/skew/${retransmit}/$dis/$fraction/$hot" 
                done
            done
        done
    done
done

for prefix in ${prefixes[@]};
do
    for cc in ${CCs[@]};
    do
        for sw in ${sws[@]};
        do
            for retransmit in ${retransmits[@]};
            do
                for dis in ${distributions[@]};
                do
                    for client in ${client_racks[@]};
                    do
                        for server in ${server_racks[@]};
                        do
                            dir="$prefix/$cc/cs-model/$sw/${retransmit}/$dis/C${client}-S${server}"
                            # echo $dir
                            mkdir -p $dir
                        done
                    done
                done
            done
        done
    done
done

for prefix in ${prefixes[@]};
do
    for cc in ${CCs[@]};
    do
        for sw in ${sws[@]};
        do
            for retransmit in ${retransmits[@]};
            do
                for dis in ${distributions[@]};
                do
                    for fraction in ${fractions[@]};
                    do
                        for hot in ${hot_traffics[@]};
                        do
                            mkdir -p "$prefix/$cc/skew/$sw/${retransmit}/$dis/$fraction/$hot" 
                        done
                    done
                done
            done
        done
    done
done
