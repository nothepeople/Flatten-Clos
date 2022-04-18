import os

if __name__ == "__main__":
    patterns = ["a2a", "o2o", "skewed"]
    rates = [30, 70, 100]
    ecmp_cmd_fmt = "python2 waf --run 'scratch/fc mix/failure-exps/large/config/ecmp/%s/%d/config.txt' > out/ecmp_%s_%d & "
    clos_cmd_fmt = "python2 waf --run 'scratch/fat-tree mix/failure-exps/large/config/clos/%s/%d/config.txt' > out/clos_%s_%d & "
    up_down_cmd_fmt = "python2 waf --run 'scratch/fc mix/failure-exps/large/config/up-down/%s/%d/config.txt' > out/up_down_%s_%d & "
    edst_cmd_fmt = "python2 waf --run 'scratch/edst-all-in-one mix/failure-exps/large/config/edst/%s/%d/config.txt' > out/edst_%s_%d & "
    """
    not run: up-down edst
    """
    for pattern in patterns:
        for rate in rates:
            os.system(clos_cmd_fmt % (pattern, rate, pattern, rate))
            os.system(ecmp_cmd_fmt % (pattern, rate, pattern, rate))
            # os.system(up_down_cmd_fmt % (pattern, rate, pattern, rate))
            # os.system(edst_cmd_fmt % (pattern, rate, pattern, rate))