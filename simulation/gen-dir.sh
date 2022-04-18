#!/bin/bash
echo "generate flow dir"
cd mix/experiment/flow/
./flow.sh
cd ../../../
echo "done"

echo "generate topo dir"
cd mix/experiment/topo/
./topo.sh
cd ../../../
echo "done"

echo "generate trace dir"
cd mix/experiment/trace
./trace.sh
cd ../../../
echo "done"

echo "Generate related output dir"
cd mix/output/
./output.sh
cd ../../

echo "Generate config dirs"
cd mix/experiment/config/
./conf.sh
cd ../../../
echo "done"

echo "generate out dirs"
cd out
./out.sh
echo "done"

