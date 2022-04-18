#!/bin/bash
prefixes=(fc-ecmp fc-uniform fc-weighted fc-lp edst-uniform edst-weighted edst-lp)
sws=(32 64)


echo "remove topo..."
for prefix in ${prefixes[@]};
do
  rm -rf $prefix
done

echo "mkdir ..."
pods=(4 8 16)
topoes=(fat-tree)
oversubscription_ratio=(1.0 1.1 1.3 1.6 2.0)
bws=(1G 40G)

mkdir -p fat-tree/

for prefix in ${prefixes[@]};
do
  for sw in ${sws[@]};
  do
    dir="$prefix/$sw/"
    mkdir -p $dir
  done
done
