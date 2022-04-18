#!/bin/bash
distributions=(web storage)
client_racks=(4 8 12 16 20 24 28 32)
server_racks=(4 8 12 16 20 24 28 32)
prefixes=(fat-tree fc-ecmp fc-uniform fc-weighted fc-lp edst-uniform edst-weighted edst-lp)
echo "rm previous dirs..."


for prefix in ${prefixes[@]};
do
    rm -rf $prefix
done



for prefix in ${prefixes[@]};
do
    for dis in ${distributions[@]};
    do
        for client in ${client_racks[@]};
        do
            for server in ${server_racks[@]};
            do
                dir="$prefix/cs-model/${dis}/C${client}-S${server}"
                mkdir -p $dir
            done
        done
    done
done

fractions=(0.04 0.10 0.20)
hot_traffics=(0.25 0.50 0.75)
for prefix in ${prefixes[@]};
do
    for dis in ${distributions[@]};
    do
        for fraction in ${fractions[@]};
        do
            for hot in ${hot_traffics[@]};
            do
                dir="$prefix/skew/${dis}/$fraction/$hot/"
                mkdir -p $dir
            done
        done
    done
done
echo "mkdir done"


