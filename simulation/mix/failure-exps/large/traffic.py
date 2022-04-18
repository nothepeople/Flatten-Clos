import sys
import random
import math
import heapq
from custom_rand import CustomRand
import numpy as np
import math
import yaml
random.seed(6688)

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


def gen_all_to_all_TM(switches, switch_recv_bandwidth):
    tm = np.zeros(shape=(switches, switches))
    for i in range(switches):
        for j in range(switches):
            if i == j: continue
            # tm[i][j] = 1
            tm[i][j] = switch_recv_bandwidth / float(switches - 1)
    # print(tm)
    return tm

def gen_one_to_one_TM(switches, switch_recv_bandwidth):
    tm = np.zeros(shape=(switches, switches))
    for i in range(switches):
        j = random.randint(0, switches - 1)
        while i == j or tm[i][j] > 0 or np.sum(tm[:, j]) > 0:
            j = random.randint(0, switches - 1)
        tm[i][j] = switch_recv_bandwidth
    # print(tm)
    return tm

def gen_skewed_TM(switches, switch_recv_bandwidth, theta, phi):
    tm = np.zeros(shape=(switches, switches))
    ### theta: fraction of hot racks
    ### phi: concentrated traffic at hot rack switches
    N_hot = int(math.ceil(switches * theta))
    N_cold = switches - N_hot
    # print(f"hot racks : {N_hot}")
    hot_rack_ids = []
    while len(hot_rack_ids) < N_hot:
        rack_id = random.randint(0, switches - 1)
        while rack_id in hot_rack_ids:
            rack_id = random.randint(0, switches - 1)
        hot_rack_ids.append(rack_id)
    # print("hot rack ids: ", hot_rack_ids)

    p_hot_to_hot = phi * phi / (N_hot * N_hot)
    p_cold_to_cold = (1 - phi) * (1 - phi) / (N_cold * N_cold)
    p_hot_to_cold = phi * (1 - phi) / (N_cold * N_hot)
    # print(f"p_hot_to_hot: {p_hot_to_hot}, p_cold_to_cold: {p_cold_to_cold}, p_hot_to_cold: {p_hot_to_cold}")
    s_hot = 0
    for i in range(switches):
        for j in range(switches):
            if i == j: continue
            if i in hot_rack_ids and j in hot_rack_ids:
                tm[i][j] = p_hot_to_hot * switch_recv_bandwidth * switches * 0.3
                s_hot += tm[i][j]
            elif not i in hot_rack_ids and not j in hot_rack_ids:
                tm[i][j] = p_cold_to_cold * switch_recv_bandwidth * switches * 0.3
            else:
                tm[i][j] = p_hot_to_cold * switch_recv_bandwidth * switches * 0.3
                s_hot += tm[i][j]
    return tm



def generate(num_ToRs, total_hosts, host_bw, tm, duration, ofile, cdf_file):
    host_per_ToR = total_hosts // num_ToRs
    host_ids = []
    base_t = 2000000000
    port = 80
    flows = []
    time = duration * 1e9 # 1000 000 000ns
    
    f = open(cdf_file,"r")
    cdf = []
    for line in f.readlines():
        x,y = map(float, line.strip().split(' '))
        cdf.append([x,y])
    customRand = CustomRand()
    if not customRand.setCdf(cdf):
        print ("Error: Not valid cdf")
        sys.exit(0)

    # unit is byte. Commented by qizhou.zqz.
    flow_avg_size = customRand.getAvg()
    print("hosts_per_ToR: {}, flow_avg_size: {}byte".format(host_per_ToR, flow_avg_size))
    for i in range(num_ToRs):
        host_ids.append([i * host_per_ToR + j for j in range(host_per_ToR)])

    flows = []
    generated_size  = 0
    for i in range(num_ToRs):
        for j in range(num_ToRs):
            if i == j or tm[i][j] == 0:
                continue
            # Pod2Pod taffic limit, Gb to bytes.
            p2p_traffic = (tm[i][j] * 1e9 / 8) * (float(time) / 1000000000)
            mlu = (p2p_traffic ) / ( host_bw * 1e9 / 8 * (float(time) / 1000000000) ) / host_per_ToR
            
            flow_avg_inter = 1e9 / (mlu * host_bw * 1e9 / 8 / flow_avg_size)

            # print("ToR-{} ---> ToR-{} p2p_traffic: {:.2f}byte mlu: {} flow_avg_inter: {}ns".format(i, j, p2p_traffic, mlu, flow_avg_inter))
            
            host_list = [(base_t + int(poisson(flow_avg_inter)), src_ToR ) for src_ToR in host_ids[i]]
            
            while len(host_list) > 0:
                t,src = host_list[0]
                inter_t = int(poisson(flow_avg_inter))
                dst = random.choice(host_ids[j])
                if (t + inter_t > time + base_t):
                    heapq.heappop(host_list)
                else:
                    size = int(customRand.rand())
                    if size <= 0:
                        size = 1
                    flows.append( Flow(src, dst, size, (t + inter_t) * 1e-9) )
                    heapq.heapreplace(host_list, (t + inter_t, src))
                    generated_size += size
    
    print("tm: {:.2f}Gb generated: {:.2f}".format(np.sum(tm), generated_size * 8 / 1e9))
    ## Final step
    ## write flow to output file
    out = open(ofile, "w")

    out.write("%d\n" % (len(flows)))

    heapq.heapify(flows)
    while len(flows) > 0:
        out.write("%s 0\n" % (flows[0]))
        heapq.heappop(flows)
    

if __name__ == "__main__":
    with open('common.yaml','r') as f:
        data = yaml.load(f)

    num_tors = data['fc']['switches']
    host_per_tor = data['fc']['to_hosts']
    hosts = num_tors * host_per_tor
    rates = [30, 70, 100]
    time = 0.1 # 0.2s
    for rate in rates:
        a2a_tm = gen_one_to_one_TM(num_tors, host_per_tor * (rate / float(100)))
        generate(
            num_ToRs=num_tors,
            total_hosts=hosts,
            host_bw=1,
            tm = a2a_tm,
            duration=time,
            ofile='traffic/a2a/%d/flow.txt' % (rate),
            cdf_file='web.txt'
        )

        o2o_tm = gen_one_to_one_TM(num_tors, host_per_tor * (rate / float(100)))
        generate(
            num_ToRs=num_tors,
            total_hosts=hosts,
            host_bw=1,
            tm = o2o_tm,
            duration=time,
            ofile='traffic/o2o/%d/flow.txt' % (rate),
            cdf_file='web.txt'
        )

        skewed_tm = gen_skewed_TM(num_tors, host_per_tor * (rate / float(100)), 0.04, 0.75)
        generate(
            num_ToRs=num_tors,
            total_hosts=hosts,
            host_bw=1,
            tm = skewed_tm,
            duration=time,
            ofile='traffic/skewed/%d/flow.txt' % (rate),
            cdf_file='web.txt'
        )
