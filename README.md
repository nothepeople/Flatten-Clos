The anonymous repository is our topology and customized routing on the paper, Flattened Clos: A High-performance Deadlock-free Expander Graph for RoCEv2 Networks.

Our simulation is extended from HPCC (https://github.com/alibaba-edu/High-Precision-Congestion-Control).

The details of our extended code on HPCC are described at [Our Extended Code](#our-extended-code)

# Getting Started
Please refer to HPCC. https://github.com/alibaba-edu/High-Precision-Congestion-Control
## HPCC simulation
[Project page of HPCC](https://hpcc-group.github.io/) includes latest news of HPCC and extensive evaluation results using this simulator.

This is the simulator for [HPCC: High Precision Congestion Control (SIGCOMM' 2019)](https://rmiao.github.io/publications/hpcc-li.pdf). It also includes the implementation of DCQCN, TIMELY, DCTCP, PFC, ECN and Broadcom shared buffer switch.

We have update this simulator to support HPCC-PINT, which reduces the INT header overhead to 1 to 2 byte. This improves the long flow completion time. See [PINT: Probabilistic In-band Network Telemetry (SIGCOMM' 2020)](https://liyuliang001.github.io/publications/pint.pdf).

## NS-3 simulation
The ns-3 simulation is under `simulation/`. Refer to the README.md under it for more details.

## Traffic generator
The traffic generator is under `traffic_gen/`. Refer to the README.md under it for more details.

# Our Extended Code

## Added List
```
simulation/deadlock.sh
simulation/easy-kill.py
simulation/gen-conf-files.sh
simulation/gen-dir.sh
simulation/ksp/
simulation/mix/experiment/
simulation/mix/failure-exps/
simulation/mix/output/
simulation/mix/topo-generator/
simulation/mix/traffic-gen/
simulation/out/
simulation/run-cs-exp.py
simulation/run-deadlock-exp.py
simulation/run-large-exp.py
simulation/run-skew-exp.py
simulation/scratch/edst-all-in-one.cc
simulation/scratch/edst.cc
simulation/scratch/fat-tree.cc
simulation/scratch/fc.cc
simulation/scratch/rrg-lp.cc
simulation/scratch/rrg-uniform.cc
simulation/scratch/rrg-weighted.cc
simulation/src/network/utils/sack-block-set.cc
simulation/src/network/utils/sack-block-set.h
simulation/src/network/utils/sack-block.cc
simulation/src/network/utils/sack-block.h
simulation/src/network/utils/sack-header.cc
simulation/src/network/utils/sack-header.h
simulation/src/point-to-point/model/longer-path-tag.cc
simulation/src/point-to-point/model/longer-path-tag.h
simulation/src/point-to-point/model/rdma-flow.cc
simulation/src/point-to-point/model/rdma-flow.h
```

## Modified List
```
simulation/src/applications/helper/rdma-client-helper.cc
simulation/src/applications/helper/rdma-client-helper.h
simulation/src/applications/model/rdma-client.cc
simulation/src/applications/model/rdma-client.h
simulation/src/internet/helper/internet-stack-helper.cc
simulation/src/network/model/buffer.h
simulation/src/network/model/node.cc
simulation/src/network/model/node.h
simulation/src/network/model/packet.cc
simulation/src/network/utils/custom-header.cc
simulation/src/network/utils/custom-header.h
simulation/src/network/utils/int-header.cc
simulation/src/network/utils/int-header.h
simulation/src/network/wscript
simulation/src/point-to-point/helper/qbb-helper.cc
simulation/src/point-to-point/helper/qbb-helper.h
simulation/src/point-to-point/model/cn-header.cc
simulation/src/point-to-point/model/cn-header.h
simulation/src/point-to-point/model/qbb-header.h
simulation/src/point-to-point/model/qbb-net-device.cc
simulation/src/point-to-point/model/qbb-net-device.h
simulation/src/point-to-point/model/rdma-driver.cc
simulation/src/point-to-point/model/rdma-driver.h
simulation/src/point-to-point/model/rdma-hw.cc
simulation/src/point-to-point/model/rdma-hw.h
simulation/src/point-to-point/model/rdma-queue-pair.cc
simulation/src/point-to-point/model/rdma-queue-pair.h
simulation/src/point-to-point/model/switch-mmu.cc
simulation/src/point-to-point/model/switch-mmu.h
simulation/src/point-to-point/model/switch-node.cc
simulation/src/point-to-point/model/switch-node.h
simulation/src/point-to-point/wscript
```