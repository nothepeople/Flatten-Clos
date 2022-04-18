pods=(4 8 16)
patterns=(uniform non-uniform)
bws=(1G 40G)
workloads=(30 70)
rm -rf tm
for pod in ${pods[@]};
do
  for pattern in ${patterns[@]};
  do
    for bw in ${bws[@]};
    do
      for workload in ${workloads[@]};
      do
        dir="tm/${pod}pod/$pattern/$bw/$workload/"
        # echo $dir
        mkdir -p $dir
      done
    done
  done
done
