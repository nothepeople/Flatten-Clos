import os
if __name__ == "__main__":
    CCs = ["dcqcn", "timely", "hpcc", "dctcp"]
    for cc in CCs:
        os.system("ps aux|grep %s | awk '{print $2}' > tmp.out" %(cc))
        f = open('tmp.out')
        for line in f.read().splitlines():
            os.system('kill %s' % line )
        os.remove('tmp.out')
