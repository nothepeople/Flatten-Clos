#!/bin/bash
echo "generate flow files..."
cd mix/traffic-gen/
./tm.sh
python2 skew-traffic-gen.py
python2 cs-traffic-gen.py
cd ../../
echo "done"

echo "generate topo files..."
cd mix/topo-generator
python2 clos.py
# python2 Xpander.py
python2 fc.py
cd ../../
echo "done"

echo "generate config files..."
cd mix/experiment/config
python2 fat-conf.py

python2 fc-ecmp-conf.py
python2 fc-lp-conf.py
python2 fc-uniform-conf.py
python2 fc-weighted-conf.py

python2 edst-uniform-conf.py
python2 edst-weighted-conf.py
python2 edst-lp-conf.py 
cd ../../
echo "done"