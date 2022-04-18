import sys
import random
import math
import heapq
from optparse import OptionParser
from custom_rand import CustomRand

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

def traffic_gen(hosts, bw, duration, load, 
								fat_ofile, 
								Xpander_ofile, 
								Xpander_ecmp_ofile, 
								rrg_ofile,
								rrg_ecmp_ofile,
								fat_incast_ofile,
								Xpander_incast_ofile,
								Xpander_ecmp_incast_ofile,
								rrg_incast_ofile,
								rrg_ecmp_incast_ofile,
								cdf_file="distribution/web.txt"):
	port = 80
	base_t = 2000000000
	nhost = hosts
	bandwidth = float(bw*1e9)
	time = duration * 1e9

	if cdf_file == "distribution/storage.txt":
		time = 200000000

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
	incast_flows = []
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
			incast_flows.append( Flow(src, dst, size, t * 1e-9) )

			heapq.heapreplace(host_list, (t + inter_t, src))
			generated += size

	fat_out = open(fat_ofile, "w")
	Xpander_out = open(Xpander_ofile, "w")
	Xpander_ecmp_out = open(Xpander_ecmp_ofile, "w")
	rrg_out = open(rrg_ofile, "w")
	rrg_ecmp_out = open(rrg_ecmp_ofile, "w")

	fat_out.write("%d\n" % len(flows) )
	Xpander_out.write("%d\n" % len(flows) )
	Xpander_ecmp_out.write("%d\n" % len(flows) )
	rrg_out.write("%d\n" % len(flows) )
	rrg_ecmp_out.write("%d\n" % len(flows) )

	heapq.heapify(flows)
	while len(flows) > 0:
		fat_out.write("%s 0\n" % (flows[0]))
		Xpander_out.write("%s 1\n" % (flows[0]))
		Xpander_ecmp_out.write("%s 0\n" % (flows[0]))
		rrg_out.write("%s 0\n" % (flows[0]))
		rrg_ecmp_out.write("%s 0\n" % (flows[0]))
		heapq.heappop(flows)
	
	### Mix incast traffic
	""" add multi to one traffic """
	bytes_to_sent = 200000
	if bw == 1:
		for i in range(2):
			unlucky = random.randint(0, nhost-1)
			print("unlucky: ", unlucky)
			incast_gen_t =  2000000000 + random.randint(100000000, 200000000)
			for _ in range(30):
				src = random.randint(0, nhost-1)
				while src == unlucky:
					src = random.randint(0, nhost-1)
				incast_flows.append( Flow(src, unlucky, bytes_to_sent, incast_gen_t * 1e-9) )
		
	fat_incast_out = open(fat_incast_ofile, "w")
	Xpander_incast_out = open(Xpander_incast_ofile, "w")
	Xpander_ecmp_incast_out = open(Xpander_ecmp_incast_ofile, "w")
	rrg_incast_out = open(rrg_incast_ofile, "w")
	rrg_ecmp_incast_out = open(rrg_ecmp_incast_ofile, "w")

	fat_incast_out.write("%d\n" % len(incast_flows) )
	Xpander_incast_out.write("%d\n" % len(incast_flows) )
	Xpander_ecmp_incast_out.write("%d\n" % len(incast_flows) )
	rrg_incast_out.write("%d\n" % len(incast_flows) )
	rrg_ecmp_incast_out.write("%d\n" % len(incast_flows) )

	heapq.heapify(incast_flows)
	while len(incast_flows) > 0:
		fat_incast_out.write("%s 0\n" % (incast_flows[0]))
		Xpander_incast_out.write("%s 1\n" % (incast_flows[0]))
		Xpander_ecmp_incast_out.write("%s 0\n" % (incast_flows[0]))
		rrg_incast_out.write("%s 0\n" % (incast_flows[0]))
		rrg_ecmp_incast_out.write("%s 0\n" % (incast_flows[0]))
		heapq.heappop(incast_flows)



if __name__ == "__main__":
	workloads = [30, 50, 70]
	distributions = ["web", "storage"]
	fat_fmt = "../experiment/flow/fat-tree/common/%s/%d/flow.txt"
	Xpander_fmt = "../experiment/flow/Xpander/common/%s/%d/flow.txt"
	Xpander_ecmp_fmt = "../experiment/flow/Xpander-ecmp/common/%s/%d/flow.txt"
	rrg_fmt = "../experiment/flow/rrg/common/%s/%d/flow.txt"
	rrg_ecmp_fmt = "../experiment/flow/rrg-ecmp/common/%s/%d/flow.txt"
	"""incast ofile"""
	fat_incast_fmt = "../experiment/flow/fat-tree/incast/%s/%d/flow.txt"
	Xpander_incast_fmt = "../experiment/flow/Xpander/incast/%s/%d/flow.txt"
	Xpander_ecmp_incast_fmt = "../experiment/flow/Xpander-ecmp/incast/%s/%d/flow.txt"
	rrg_incast_fmt = "../experiment/flow/rrg/incast/%s/%d/flow.txt"
	rrg_ecmp_incast_fmt = "../experiment/flow/rrg-ecmp/incast/%s/%d/flow.txt"
	for dis in distributions:
		for workload in workloads:
			traffic_gen(
				hosts=256,
				bw=1,
				duration=0.2,
				load=float(workload) / 100,
				fat_ofile=(fat_fmt % (dis, workload)),
				Xpander_ofile=(Xpander_fmt % (dis, workload)),
				Xpander_ecmp_ofile=(Xpander_ecmp_fmt % (dis, workload)),
				rrg_ofile=(rrg_fmt % (dis, workload)),
				rrg_ecmp_ofile=(rrg_ecmp_fmt % (dis, workload)),

				fat_incast_ofile=(fat_incast_fmt % (dis, workload)),
				Xpander_incast_ofile=(Xpander_incast_fmt % (dis, workload)),
				Xpander_ecmp_incast_ofile=(Xpander_ecmp_incast_fmt % (dis, workload)),
				rrg_incast_ofile=(rrg_incast_fmt % (dis, workload)),
				rrg_ecmp_incast_ofile=(rrg_ecmp_incast_fmt % (dis, workload)),
				cdf_file=("distribution/%s.txt" % (dis))
      )


