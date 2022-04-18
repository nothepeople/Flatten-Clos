import os

if __name__ == "__main__":
  ecmp = "python2 waf --run 'scratch/fc-ecmp mix/failure-exps/deadlock/config/ecmp/config.txt' > ecmp & "
  up_down = "python2 waf --run 'scratch/edst-all-in-one mix/failure-exps/deadlock/config/up-down/config.txt' > up_down & "
  os.system(ecmp)
  os.system(up_down)