#include <vector>
#include <iostream>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <set>
#include <unistd.h>

#define INF 0xffffff
#define FOR_EACH(vec) {for (int i = 0; i < vec.size(); i++) { std::cout << vec[i] << " "; } std::cout << "\n";}

std::string sw2sw_mat_file;
std::vector<std::vector<int> > graph;
std::unordered_map<uint32_t, 
	std::unordered_map<uint32_t, 
		std::shared_ptr<std::vector<std::vector<int> > > > > sw_paths; // sw id start from zero

uint32_t host_per_switch, switch_num, sw2sw_link_num, total_hosts;
uint32_t k_level_clos;
std::vector<int> port_range;

struct VectorCompare {
  bool smaller(const std::vector<int> &lhs, const std::vector<int> &rhs) const {
    for (int i = 0; i < lhs.size(); i++) {
      if (lhs[i] < rhs[i])
        return true;
    }
    return false; 
  }
  bool operator()(const std::vector<int> &lhs, const std::vector<int> &rhs) const {
    return lhs.size() < rhs.size() || lhs.size() == rhs.size() && smaller(lhs, rhs);
  }
};

void ConstructSwitchConnMatrix(const std::string& mat_file="demo_mat") {
  std::ifstream ifs;
  ifs.open(mat_file.c_str());
  ifs >> host_per_switch >> switch_num >> sw2sw_link_num >> total_hosts;
  ifs >> k_level_clos;
  port_range.resize(k_level_clos * 2, 0);
  for (int i = 0; i < k_level_clos * 2; i ++) {
    ifs >> port_range[i];
  }
  // FOR_EACH(port_range)
  // printf("RRG host_per_switch: %d switches: %d sw2sw_links: %d total_hosts: %d\n", host_per_switch, switch_num, sw2sw_link_num, total_hosts);
  
  graph.resize(switch_num * 3, std::vector<int>(switch_num * 3, INF));
  for (int i = 0; i <switch_num; i++) {
    graph[i][i] = INF;
    graph[i + switch_num][i + switch_num] = INF;
    graph[i + 2 * switch_num][i + 2 * switch_num] = INF;

    graph[i][i + switch_num] = 1;
    graph[i + switch_num][i] = 1;
    
    graph[i + switch_num] [ i + 2 * switch_num] = 1;
    graph[i + 2 * switch_num][i + switch_num] = 1;
  }

  for (int i = 0; i < sw2sw_link_num; i++) {
    int swA, swB, pA, pB;
    ifs >> swA >> swB >> pA >> pB;
    // printf("A: %d B: %d pA: %d pB: %d\n", swA, swB, pA, pB); 
    if (pA >= port_range[0] && pA <= port_range[1]) {// A is left port, B is mid port
      graph[swA][swB + switch_num] = 1;
      graph[swB + switch_num][swA] = 1;
    } else if (pA >= port_range[2] && pA <= port_range[3]) {// A is mid port, B may be left or right port
      if (pB >= port_range[0] && pB <= port_range[1]) { // 
        graph[swA + switch_num][swB] = 1;
        graph[swB][swA + switch_num] = 1;
      } else if (pB >= port_range[4] && pB <= port_range[5]) {
        graph[swA + switch_num][swB + 2 * switch_num] = 1;
        graph[swB + 2 * switch_num][swA + switch_num] = 1;
      }
    } else if (pA >= port_range[4] && pA <= port_range[5]) { // A is right port, B is mid port
      graph[swA + 2 * switch_num][swB + switch_num] = 1;
      graph[swB + switch_num][swA + 2 * switch_num] = 1;
    }
  }
  // std::cout << "is close\n";
	// const_graph = graph;
  ifs.close();
}

void FindPath(int srcRack, int dstRack, int startRack, 
    std::set<std::vector<int>, VectorCompare >& paths, 
    std::vector<int>& path,
    std::vector<int> &visited) {
  if (path.size() > 4) {
    return;
  }
  if (dstRack == startRack) {
    std::vector<int> tmp = path;
    tmp.push_back(dstRack);
    // printf("tmp\n");
    // FOR_EACH(tmp);
    // printf("\n\n");
    std::vector<int> p;
    for (auto node : tmp) {
      if (p.empty()) {
        p.push_back(node);
        continue;
      }
      if (node >= switch_num && node < 2 * switch_num) {
        if (p[p.size() - 1] != (node - switch_num))
          p.push_back(node - switch_num);
      } else if (node >= 2 * switch_num) {
        if (p[p.size() - 1] != (node - switch_num * 2))
          p.push_back(node - 2 * switch_num);
      } else {
        if (p[p.size() - 1] != node)
          p.push_back(node);
      }
    }
    paths.insert(p);
    return;
  }
	for (int i = 0; i < graph[startRack].size(); i++) {
    if (visited[i] || graph[startRack][i] == INF ) {
      continue;
    }
    path.push_back(startRack);
    visited[startRack] = 1;
    FindPath(srcRack, dstRack, i, paths, path, visited);
    visited[startRack] = 0;
    path.pop_back();
  }
}

void CalSwToSwPath() {

  int ii = 0;
  int pairs = 0;
  double minSumPathLen = 0;
  double sumAvgPathLen = 0;
  double sumPaths = 0;
  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      if (i == j) { continue; }
      std::set<std::vector<int>, VectorCompare > paths;
      std::vector<std::vector<int> > vecPaths;
      std::vector<int> dfsPath;
      std::vector<int> visited(switch_num * 3, 0);
      FindPath(i, j, i, paths, dfsPath, visited);
      if (paths.size() == 0) {
        ii ++;
        continue;
      }
      // printf("%d->%d size=%d\n", i, j, paths.size());
      ++pairs;
      sumPaths += paths.size();
      int mini = 0xff;
      double tmpSum = 0;
      for (auto &p : paths) {
        // FOR_EACH(p);
        if (mini > p.size()) mini = p.size();
        tmpSum += p.size();

        vecPaths.push_back(p);
      }
      minSumPathLen += mini;
      sumAvgPathLen += (double)tmpSum / paths.size();
        

			std::shared_ptr<std::vector<std::vector<int> > > p(new std::vector<std::vector<int> >(vecPaths));
			sw_paths[i][j] = p;
		}
  }
  // printf("zero: %d total %d percentile: %.2f\%\n", ii, (switch_num - 1) * (switch_num - 1), (double)ii/ ((switch_num - 1) * (switch_num - 1)) * 100);
  printf("Avg paths: %.2f, avg path lens: %.2f, avg minimum path lens: %.2f\n", sumPaths / pairs, sumAvgPathLen / pairs, minSumPathLen / pairs);
}


int main() {
  ConstructSwitchConnMatrix();
  CalSwToSwPath();
  // std::cout << "Hello multi path RRG\n";
}