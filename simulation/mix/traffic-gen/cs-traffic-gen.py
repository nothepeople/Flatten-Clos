import numpy as np
import random
import math
import heapq
from optparse import OptionParser
from custom_rand import CustomRand

random.seed(1000)
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

def traffic_gen(client_racks=16,
                server_racks=16,
                host_every_rack=8, 
                num_tors=32, 
                server_target_mlu=0.7, 
                duration=0.2, 
                bw=1,
                fat_ofile="fat.txt", 

								fc_ecmp_ofile="fc-ecmp.txt",
                fc_uniform_ofile="fc-uniform.txt",
                fc_weighted_ofile="fc-weighted.txt",
                fc_lp_ofile="fc-lp.txt",

                edst_uniform_ofile="edst-uniform.txt",
                edst_weighted_ofile="edst-weighted.txt",
                edst_lp_ofile="edst-lp.txt",
								cdf_file="distribution/web.txt"):
  port = 80
  base_t = 2000000000
  # nhost = hosts
  bandwidth = float(bw*1e9)
  time = duration * 1e9
  client_send_mlu = server_target_mlu / (float(client_racks) / server_racks) 
  if client_send_mlu > 1:
    client_send_mlu = 1
  # print("server_target_mlu", server_target_mlu)
  # print("clients_racks/server_racks: {}".format((float(client_racks) / server_racks) ))
  # print("client_sent_mlu: {}".format(client_send_mlu))
  client_rack_ids = []
  server_rack_ids = []
  for _ in range(client_racks):
    client_id = random.randint(0, num_tors-1)
    while client_id in client_rack_ids:
      client_id = random.randint(0, num_tors-1)
    client_rack_ids.append(client_id)
  
  for _ in range(server_racks):
    server_id = random.randint(0, num_tors-1) 
    while server_id in client_rack_ids or server_id in server_rack_ids:
      server_id = random.randint(0, num_tors-1)
    server_rack_ids.append(server_id)

  clients = []
  servers = []
  for cli_rack in client_rack_ids:
    for i in range(host_every_rack):
      clients.append(cli_rack * host_every_rack + i)
  
  for srv_rack in server_rack_ids:
    for i in range(host_every_rack):
      servers.append(srv_rack * host_every_rack + i)
  
  # print("client rack ids", client_rack_ids)
  # print("server rack ids", server_rack_ids)
  # print("clients: ", clients)
  # print("servers: ", servers)
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
  
  # create a custom uniform generator, which takes a cdf, and generate number according to the cdf
  customRand = CustomRand()
  if not customRand.setCdf(cdf):
    print "Error: Not valid cdf"
    sys.exit(0)
  
  avg = customRand.getAvg()
  avg_inter_arrival = 1/(bandwidth*client_send_mlu/8./avg)*1000000000
  flows = []
  host_send_bytes_limit = bandwidth * duration / 8
  host_list = [(base_t + int(poisson(avg_inter_arrival)), client) for client in clients]
  generated = 0
  while len(host_list) > 0:
		t,src = host_list[0]
		inter_t = int(poisson(avg_inter_arrival))
		new_tuple = (src, t + inter_t)
		dst = servers[ random.randint(0, len(servers) - 1) ]
		
		if (t + inter_t > time + base_t):
			heapq.heappop(host_list)
		else:
			size = int(customRand.rand())
			if size <= 0:
				size = 1
			flows.append( Flow(src, dst, size, t * 1e-9) )
			heapq.heapreplace(host_list, (t + inter_t, src))
			generated += size
  # print("client_racks %d, server_racks %d, generated_traffic / server_racks_bw %.2f flows: %d" % (client_racks, server_racks, float(generated)  / ( bandwidth / 8 * server_racks * host_every_rack * duration), len(flows) ) )

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
  a = [1, 3, 5]
  client_racks = [i * 4 for i in range(1, 9)]
  server_racks = [i * 4 for i in range(1, 9)]
  fat_fmt = "../experiment/flow/fat-tree/cs-model/%s/C%d-S%d/flow.txt"


  fc_ecmp_fmt = "../experiment/flow/fc-ecmp/cs-model/%s/C%d-S%d/flow.txt"
  fc_uniform_fmt = "../experiment/flow/fc-uniform/cs-model/%s/C%d-S%d/flow.txt"
  fc_weighted_fmt = "../experiment/flow/fc-weighted/cs-model/%s/C%d-S%d/flow.txt"
  fc_lp_fmt = "../experiment/flow/fc-lp/cs-model/%s/C%d-S%d/flow.txt"

  edst_uniform_fmt = "../experiment/flow/edst-uniform/cs-model/%s/C%d-S%d/flow.txt"
  edst_weighted_fmt = "../experiment/flow/edst-weighted/cs-model/%s/C%d-S%d/flow.txt"
  edst_lp_fmt = "../experiment/flow/edst-lp/cs-model/%s/C%d-S%d/flow.txt"
  distributions = ["web", "storage"]
  for distribution in distributions:
    for client in client_racks:
      for server in server_racks:
        traffic_gen(
          client_racks=client,
          server_racks=server,
          host_every_rack=4,
          num_tors=64,
          fat_ofile=fat_fmt % (distribution, client, server),
    
          fc_ecmp_ofile=fc_ecmp_fmt % (distribution, client, server),
          fc_uniform_ofile=fc_uniform_fmt % (distribution, client, server),
          fc_weighted_ofile=fc_weighted_fmt % (distribution, client, server),
          fc_lp_ofile=fc_lp_fmt % (distribution, client, server),

          edst_uniform_ofile= edst_uniform_fmt % (distribution, client, server),
          edst_weighted_ofile= edst_weighted_fmt % (distribution, client, server),
          edst_lp_ofile= edst_lp_fmt % (distribution, client, server),
          cdf_file="distribution/%s.txt" % (distribution)
        )

