#
## fct.tx
Eg output:
```
0b000301 0b000101 10000 100 2000000 2000000000 502008 171840
```
Sip:0b000301, Dip:0b000301, Sport: 10000, Dport:100, size:2000000Byte, start_time(ns): 2000000000, FCT(ns):502008, standalone_fct(ns):171840。standalone FC。
## flow.txlin
Eg Line
```
2 1 3 100 200000000 2
```
src_node: 2, dst_node: 1, priority_group: 3, dst_port: 100, size(Packet Count): 200000000, start_time(s):2packe1000byte200 000 000 * 1000 = 200 000 000 000(bytes) = 200Gbytes.

## topology.tx
Eg line
```
0 1 100Gbps 0.001ms 0
```
src_node:0, dst_node:1, link_speed:100Gbps, link_delay: 0.001ms, error rate:0。

## fat.txt
* 20 ToR switches.
* 20 Aggregation Switches.
* 16 Core switches.
* Each ToR connect 16 hosts.

## gen topology
Generate topology file.Po
```
python topology_gen.py -f ./pod_250/pod_250.txt -n 500 -t 25 -a 25
```

## gen trace
generate trace file
```
python trace_gen.py -f pod_500/trace.txt -t 550
```

## Ex(Now Doing and TODO)
1.。
2.。

## Traffic Aware Topology
### Topology Computation

### Traffic Generation
