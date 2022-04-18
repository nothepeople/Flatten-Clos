import sys
import random
import math
import heapq
from custom_rand import CustomRand

# import deadlock-detection


class Flow:
	def __init__(self, src, dst, size, t):
		self.src, self.dst, self.size, self.t = src, dst, size, t
	def __str__(self):
		return "%d %d 3 100 %d %.9f"%(self.src, self.dst, self.size, self.t)
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

def cyclic_traffic(loop_paths, switches, to_hosts):
  """
  background traffic
  """
  flows = []

  for path in loop_paths:
    src = path[0]
    dst = path[-1]
    print("Traffic: {} -> {} ".format(src, dst) )
    for h in range( to_hosts ):
      src_host = src * to_hosts + h
      dst_host = dst * to_hosts + h
      size = 128000
      t = 2070000000
      flows.append( Flow(src_host, dst_host, size, t * 1e-9) )
  return flows


def background_traffic_gen(hosts, bw, duration, load, cdf_file):
	port = 80
	base_t = 2000000000
	nhost = hosts
	bandwidth = translate_bandwidth(bw)
	time = float(duration) * 1e9

	fileName = cdf_file
	file = open(fileName,"r")
	lines = file.readlines()
	# read the cdf, save in cdf as [[x_i, cdf_i] ...]
	cdf = []
	for line in lines:
		x,y = map(float, line.strip().split(' '))
		cdf.append([x,y])

	# create a custom random generator, which takes a cdf, and generate number according to the cdf
	customRand = CustomRand()
	if not customRand.setCdf(cdf):
		print "Error: Not valid cdf"
		sys.exit(0)

	# generate flows
	avg = customRand.getAvg()
	avg_inter_arrival = 1/(bandwidth*load/8./avg)*1000000000
	n_flow_estimate = int(time / avg_inter_arrival * nhost)
	n_flow = 0
	# ofile.write("%d \n"%n_flow_estimate)
	host_list = [(base_t + int(poisson(avg_inter_arrival)), i) for i in range(nhost)]
	heapq.heapify(host_list)
	generated = 0
	flows = []
	while len(host_list) > 0:
		t,src = host_list[0]
		inter_t = int(poisson(avg_inter_arrival))
		new_tuple = (src, t + inter_t)
		dst = random.randint(0, nhost-1)
		while (dst == src):
			dst = random.randint(0, nhost-1)
		if (t + inter_t > time + base_t):
			heapq.heappop(host_list)
		else:
			size = int(customRand.rand())
			if size <= 0:
				size = 1
			n_flow += 1
			flows.append( Flow(src, dst, size, t * 1e-9) )
			heapq.heapreplace(host_list, (t + inter_t, src))
			generated += size

	return flows
	
def deadlock_evaluation_traffic(loop_paths, switches, to_hosts, ports, hosts, bw, duration, load, cdf_file, output):
  
  flows = []

  cyclic_flows = cyclic_traffic(loop_paths, switches, to_hosts)
  flows.extend( cyclic_flows )
  background_flows = background_traffic_gen(switches * to_hosts, bw, duration, load, cdf_file)
  flows.extend( background_flows )
  print("flows: {}".format( len(flows) ) )
  ofile = open(output, mode="w")
  ofile.write("%d\n" % ( len(flows ) ) )
  heapq.heapify(flows)
  while len(flows) > 0:
	  ofile.write("%s 0\n" % (flows[0]))
	  heapq.heappop(flows)


if __name__ == "__main__":
	loop_paths = [
      [35, 15, 41, 4], 
      [4, 16, 0], 
      [0, 2, 6], 
      [16, 0, 2, 56], 
      [6, 43, 15], 
      [15, 41, 4], 
      [8, 0, 2, 6], 
      [15, 41, 4, 3], 
      [41, 4, 16], 
      [43, 15, 41], 
      [16, 0, 2], 
      [61, 2, 6, 43]
	]
	random.seed(2022)
	deadlock_evaluation_traffic(loop_paths, 
		switches=64, 
		to_hosts=24, 
		ports=36,
		hosts=64 * 24,
		bw="1G",
		duration=0.1,
		load=0.3,
		cdf_file="web.txt",
		output="../deadlock/traffic/flow.txt"
	)