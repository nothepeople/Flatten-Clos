#coding=utf-8
import numpy as np
import random
import math
import heapq
import time as tt
from optparse import OptionParser
from custom_rand import CustomRand

# random.seed(1000)
class Flow:
	def __init__(self, src, dst, size, t):
		self.src, self.dst, self.size, self.t = src, dst, size, t
	def __str__(self):
		return "%d %d 3 100 %d %.9f" % (self.src, self.dst, self.size, self.t)
	def __lt__(self, other):
		return self.t < other.t

def translate_bandwidth(b):
	if b == None:
		return None
	if type(b)!=str:
		return None
	if b[-1] == 'G':
		return float(b[:-1])*1e9
	if b[-1] == 'M':
		return float(b[:-1])*1e6
	if b[-1] == 'K':
		return float(b[:-1])*1e3
	return float(b)

def poisson(lam):
	return -math.log(1-random.random())*lam

def traffic_gen(rack_switches=32,
                hosts_every_rack=8, 
                hot_rack_fraction=0.04,
                concentrated_traffic_fraction=0.77,
                mlu=0.7, 
                duration=1, 
                bw=1,
                fat_ofile="hot-fat.txt", 

								fc_ecmp_ofile="hot-fc-ecmp.txt",
                fc_uniform_ofile="hot-fc-uniform.txt",
                fc_weighted_ofile="hot-fc-weighted.txt",
                fc_lp_ofile="hot-fc-lp.txt",

                edst_uniform_ofile="hot-edst-uniform.txt",
                edst_weighted_ofile="hot-edst-weighted.txt",
                edst_lp_ofile="hot-edst-lp.txt",
								cdf_file="distribution/web.txt"):
  port = 80
  base_t = 2000000000
  # nhost = hosts
  bandwidth = float(bw*1e9)
  time = duration * 1e9
  mlu = 1

  R_hot = int(math.ceil(rack_switches * hot_rack_fraction))
  R_cold = rack_switches - R_hot
  p_comm_hot_racks = concentrated_traffic_fraction * concentrated_traffic_fraction / (R_hot * R_hot) 
  p_comm_cold_racks = (1 - concentrated_traffic_fraction) * (1 - concentrated_traffic_fraction) / (R_cold * R_cold)
  p_comm_hot_cold_racks = (1 - concentrated_traffic_fraction) * concentrated_traffic_fraction / (R_cold * R_hot)

  # print('hot_rack_fractions', hot_rack_fraction)
  # print('hot_concentrated_traffic', concentrated_traffic_fraction)
  # print('mlu', mlu)
  # print('duration', duration)
  # print('R_hot', R_hot)
  # print('R_cold', R_cold)
  # print('p_comm_hot', p_comm_hot_racks)
  # print('p_comm_cold', p_comm_cold_racks)
  # print('p_comm_hot_cold', p_comm_hot_cold_racks)


  demands = np.zeros((rack_switches, rack_switches))
  
  hot_rack_ids = []
  for _ in range(R_hot):
    hot = random.randint(0, rack_switches-1)
    while hot in hot_rack_ids:
      hot = random.randint(0, rack_switches-1)
    hot_rack_ids.append(hot)
  # print('hot racks:', hot_rack_ids)
  cold_rack_ids = []
  for cold in range(rack_switches):
    if cold in hot_rack_ids: continue
    cold_rack_ids.append(cold)


  
  fileName = cdf_file
  file = open(fileName,"r")
  lines = file.readlines()
  # read the cdf, save in cdf as [[x_i, cdf_i] ...]
  cdf = []
  for line in lines:
    x,y = map(float, line.strip().split(' '))
    cdf.append([x,y])
  
  # create a custom uniform generator, which takes a cdf, and generate number according to the cdf
  customRand = CustomRand()
  if not customRand.setCdf(cdf):
    print "Error: Not valid cdf"
    sys.exit(0)
  
  avg = customRand.getAvg()
  avg_inter_arrival = 1/(bandwidth*mlu/8./avg)*1000000000
  flows = []

  print(p_comm_hot_racks)
  print(p_comm_cold_racks)
  print(p_comm_hot_cold_racks)
  print(avg_inter_arrival)
  print(avg_inter_arrival / p_comm_hot_racks)
  print(avg_inter_arrival / p_comm_cold_racks)
  print(avg_inter_arrival / p_comm_hot_cold_racks)
  
  for i in range(rack_switches):
    for j in range(rack_switches):
      if i in hot_rack_ids and j in hot_rack_ids:
        demands[i][j] = p_comm_hot_racks
      elif i in cold_rack_ids and j in cold_rack_ids:
        demands[i][j] = p_comm_cold_racks
      else:
        demands[i][j] = p_comm_hot_cold_racks


  print('sum demands:', demands.sum())
  for i in range(rack_switches):
    for j in range(rack_switches):
      # if i==j: continue
      host_list = [(base_t + int(poisson(avg_inter_arrival)), i * hosts_every_rack + k) for k in range(hosts_every_rack)] 
      while len(host_list) > 0:
        t,src = host_list[0]
        inter_t = int(poisson(avg_inter_arrival))
        new_tuple = (src, t + inter_t)
        
        dst = j * hosts_every_rack + random.randint(0,hosts_every_rack-1) 
        while dst == src:
          dst = j * hosts_every_rack + random.randint(0,hosts_every_rack-1) 
        
        if (t + inter_t > time + base_t):
          heapq.heappop(host_list)
        else:
          p_comm = random.random()
          if p_comm > demands[i][j]: 
            heapq.heapreplace(host_list, (t + inter_t, src))
            continue
          size = int(customRand.rand())
          if size <= 0:
            size = 1
          flows.append( Flow(src, dst, size, t * 1e-9) )
          heapq.heapreplace(host_list, (t + inter_t, src))

  fat_out = open(fat_ofile, "w")

  
  fc_ecmp_out = open(fc_ecmp_ofile, "w")
  fc_uniform_out = open(fc_uniform_ofile, "w")
  fc_weighted_out = open(fc_weighted_ofile, "w")
  fc_lp_out = open(fc_lp_ofile, "w")


  edst_uniform_out = open(edst_uniform_ofile, "w")
  edst_weighted_out = open(edst_weighted_ofile, "w")
  edst_lp_out = open(edst_lp_ofile, "w")


  fat_out.write("%d\n" % len(flows) )


  fc_ecmp_out.write("%d\n" % len(flows) )
  fc_uniform_out.write("%d\n" % len(flows) )
  fc_weighted_out.write("%d\n" % len(flows) )
  fc_lp_out.write("%d\n" % len(flows) )

 
  edst_uniform_out.write("%d\n" % len(flows) )
  edst_weighted_out.write("%d\n" % len(flows) )
  edst_lp_out.write("%d\n" % len(flows) )
  
  print('(%.2f, %.2f)skewed flows: %d'% (hot_fraction, concentrated_traffic_fraction, len(flows)))
  heapq.heapify(flows)
  while len(flows) > 0:
    fat_out.write("%s 0\n" % (flows[0]))
    

    fc_ecmp_out.write("%s 0\n" % (flows[0]))
    fc_uniform_out.write("%s 0\n" % (flows[0]))
    fc_weighted_out.write("%s 0\n" % (flows[0]))
    fc_lp_out.write("%s 0\n" % (flows[0]))
    

    edst_uniform_out.write("%s 0\n" % (flows[0]))
    edst_weighted_out.write("%s 0\n" % (flows[0]))
    edst_lp_out.write("%s 0\n" % (flows[0]))
    heapq.heappop(flows)

if __name__ == "__main__":
  fat_fmt = "../experiment/flow/fat-tree/skew/%s/%.2f/%.2f/flow.txt"

  
  fc_ecmp_fmt = "../experiment/flow/fc-ecmp/skew/%s/%.2f/%.2f/flow.txt"
  fc_uniform_fmt = "../experiment/flow/fc-uniform/skew/%s/%.2f/%.2f/flow.txt"
  fc_weighted_fmt = "../experiment/flow/fc-weighted/skew/%s/%.2f/%.2f/flow.txt"
  fc_lp_fmt = "../experiment/flow/fc-lp/skew/%s/%.2f/%.2f/flow.txt"

  edst_uniform_fmt = "../experiment/flow/edst-uniform/skew/%s/%.2f/%.2f/flow.txt"
  edst_weighted_fmt = "../experiment/flow/edst-weighted/skew/%s/%.2f/%.2f/flow.txt"
  edst_lp_fmt = "../experiment/flow/edst-lp/skew/%s/%.2f/%.2f//flow.txt"

  distributions = ["web"]
  # hot_fractions=[0.04, 0.10, 0.20]
  # hot_trafics=f[0.25, 0.50, 0.75]
  hot_fractions=[0.04]
  hot_traffics=[0.25,0.50,0.75]
  for distribution in distributions:
    for hot_fraction in hot_fractions:
      for hot_traffic in hot_traffics:
        traffic_gen(
          rack_switches=64,
          hosts_every_rack=4,
          hot_rack_fraction=hot_fraction,
          concentrated_traffic_fraction=hot_traffic,
          duration=1,
          fat_ofile=fat_fmt % (distribution, hot_fraction, hot_traffic),
          
          fc_ecmp_ofile=fc_ecmp_fmt % (distribution, hot_fraction, hot_traffic),
          fc_uniform_ofile=fc_uniform_fmt % (distribution, hot_fraction, hot_traffic),
          fc_weighted_ofile=fc_weighted_fmt % (distribution, hot_fraction, hot_traffic),
          fc_lp_ofile=fc_lp_fmt % (distribution, hot_fraction, hot_traffic),

         
          edst_uniform_ofile=edst_uniform_fmt % (distribution, hot_fraction, hot_traffic),
          edst_weighted_ofile=edst_weighted_fmt % (distribution, hot_fraction, hot_traffic),
          edst_lp_ofile=edst_lp_fmt % (distribution, hot_fraction, hot_traffic),
          cdf_file="distribution/%s.txt" % (distribution)
        )
