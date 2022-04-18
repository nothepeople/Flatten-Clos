// #include "k-short-path.h"
#include <vector>
#include <queue>
#include <iostream>
#include <cstdint>
#include <functional>
#include <string>
#include <list>
#include <algorithm>
#include <memory>
#include <set>
#include <cassert>
#include <fstream>
#include <time.h>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <pthread.h>
#include <unistd.h>

using namespace std;
#define THREAD_NUM 32
#define INF 0xffffff
#define MAKE_PAIR(a, b) std::make_pair(a, b)
#define FOR_EACH(vec) {for (int i = 0; i < vec.size(); i++) { if (i == vec.size() - 1) { std::cout << vec[i];} else { std::cout << vec[i] << " "; } } std::cout << "\n";}

std::unordered_map<int, 
    std::unordered_map<int, 
        std::vector<std::vector<int> > > > sw_paths;

std::mutex path_cal_mutex;
std::condition_variable finish_cal_cv;
uint32_t finish_cal_flag = 0;


/**
 * Heap shortest path: https://zhuanlan.zhihu.com/p/34624812
 */
class Path {
public:
	Path():weight(0){}

	Path(const std::vector<int>& _vertexes, int _weight) : vertexes(_vertexes), weight(_weight) {}

	Path(const Path& p1, const Path &p2) {
		weight = p1.weight + p2.weight;
		for (auto p : p1.vertexes)
			vertexes.push_back(p);
		for (auto p : p2.vertexes)
			vertexes.push_back(p);
	}

	int indexOf(int vex) {
		for (int i = 0; i < vertexes.size(); i++) {
			if (vertexes[i] == vex)
				return i;
		}
		return -1;
	}

  bool operator==(const Path &other) const {
    if (weight != other.weight) {
      return false;
    }
    if (vertexes.size() != other.vertexes.size()) {
      return false;
    }

    for (int i = 0; i < vertexes.size(); i++) {
      if (vertexes[i] != other.vertexes[i])
        return false;
    }
    return true;
  }

  bool operator!=(const Path& other) const {  return !operator==(other);  }

	int get(int index) const { return vertexes[index]; }

	int getVertexNum() const { return vertexes.size(); }

	std::vector<int>& getVertexes() { return vertexes; }

	int getWeight() const { return weight; }
private: 
	std::vector<int> vertexes;
	int weight;
};
typedef std::shared_ptr<Path> PathPtr;

struct PathCompare {
  bool smaller(const PathPtr& left, const PathPtr& right) const {
    for (int i = 0; i < left->getVertexNum(); i++) {
      if (left->getVertexes()[i] < right->getVertexes()[i])
        return true;
    }
    return false;
  }
	bool operator()(const PathPtr& left, const PathPtr& right) {
		return left->getWeight() < right->getWeight() || 
            left->getWeight() == right->getWeight() && left->getVertexNum() < right->getVertexNum() || 
            left->getWeight() == right->getWeight() && left->getVertexNum() == right->getVertexNum() && smaller(left, right);
	}
};


uint32_t calculatePathWeight(const vector<vector<int> >& graph, const std::vector<int> &vex) {
  uint32_t weight = 0;
  for (uint32_t i = 0; i < vex.size() - 1; i++)
    weight += graph[vex[i]][vex[i + 1]];
  return weight;
}

bool isSetContainsPath(const std::set<PathPtr, PathCompare>& candidatePaths, PathPtr path) {
  for (auto candidate : candidatePaths) {
    if (*candidate == *path) 
      return true;
  }
  return false;
}

void testTwoG(const std::vector<std::vector<int> >& g1, const std::vector<std::vector<int> >& g2) {
  for (int i = 0; i < g1.size(); i++) {
    for (int j = 0; j < g1[i].size(); j++) {
      assert(g1[i][j] == g2[i][j]);
    }
  }
}

struct TemporaryChangedDis {
  TemporaryChangedDis(int _s, int _e, int _d) {
    start = _s;
    end = _e;
    dis = _d;
  }
  int start;
  int end;
  int dis;
};

bool isChangedSetContain(const std::vector<TemporaryChangedDis>& changed, int s, int e) {
  if (changed.size() == 0) {
    return false;
  }
  for (auto& ch : changed) {
    if (ch.start == s && ch.end == e)
      return true;
  }
  return false;
}

PathPtr DijkStraShortestPath(const std::vector<std::vector<int> >& g,
                  vector<vector<int> >& graph, 
                  int start, int end,
                  const std::vector<int> unavailableNodes={}, 
                  const std::set<PathPtr, PathCompare>& unavailablePaths={}) {
  int vertex_num = graph.size();
  std::vector<bool> visited(vertex_num, false);
  std::vector<int> dis(vertex_num, INF);
  std::vector<int> prev(vertex_num);
  std::vector<int> result;
  // graph[i][j] = 
  std::vector<TemporaryChangedDis> changed;
  // std::cout << "start test\n";
  testTwoG(g, graph);
  


  //initialize
  // unavaliable edges
  if (unavailablePaths.size() > 0) {
    for (auto pathPtr : unavailablePaths) {
      // FOR_EACH(pathPtr->getVertexes());
      int index = pathPtr->indexOf(start);
      int temp = graph[start][pathPtr->get(index + 1)];
      graph[start][pathPtr->get(index + 1)] = INF;
      if (!isChangedSetContain(changed, start,  pathPtr->get(index + 1))) {
        changed.emplace_back(start, pathPtr->get(index + 1), temp);
        // printf("change start%d end%d weight 1 to INF\n", start,  pathPtr->get(index + 1));
      }
    }
  }

  // unavailable nodes
  if (unavailableNodes.size() > 0) {
    for (auto node : unavailableNodes)
      visited[node] = true;
  }

  for (int i = 0; i < vertex_num; i++) {
    if (i == start) {
      dis[i] = 0;
      prev[i] = -1;
      visited[start] = true;
    } else {
      if (graph[start][i] != INF) { // i connect to start
        dis[i] = graph[start][i];
        prev[i] = start;
      } else {
        dis[i] = INF;
        prev[i] = -1; 
      }
    }
  }

  for (int i = 0; i < vertex_num; i++) {
    int k = -1;
    int min_dis = INF;
    for (int j = 0;  j < vertex_num; j++) {
      if (!visited[j]&& min_dis > dis[j]) {
        min_dis = dis[j];
        k = j;
      }
    }

    if (k == -1) { //source to other are both 
      break;
    }


    visited[k] = true;
    for (int j = 0; j < vertex_num; j++) {
      if (!visited[j] && dis[j] >= dis[k] + graph[k][j]) {
        dis[j] = dis[k] + graph[k][j];
        prev[j] = k;
      }
    }
  }
  
  // resotre  graph distance
  for (auto &ch : changed) {
    graph[ch.start][ch.end] = ch.dis;
    // printf("restore start%d end%d weight  INFto %d\n", ch.start,  ch.end, ch.dis);
  }
  // std::cout << "end test\n";
  testTwoG(g, graph);
  //
  if (dis[end] == INF) {
    // printf("start: %d and end: %d are not connected\n", start,  end);
    return nullptr;
  } 

  result.push_back(end);
  int temp = prev[end];
  while(temp != -1) {
    // std::cout << "iii\n";
    result.push_back(temp);
    temp = prev[temp];
  }
  std::reverse(result.begin(), result.end());
  // FOR_EACH(result);
  // PathPtr path = std::make_shared<Path>(graph, path);
  PathPtr path(new Path(result, calculatePathWeight(graph, result)));
  return path;
}


double TopKShortestPathAlgo(std::vector<std::vector<int> > & pair_paths, 
  const std::vector<std::vector<int> >& g, std::vector<std::vector<int> >& graph, int start, int end, int k) {
  std::set<PathPtr, PathCompare> candidatePaths;
  std::set<PathPtr, PathCompare> kshortestPaths;
  PathPtr shorest_path = DijkStraShortestPath(g, graph, start, end);
  kshortestPaths.insert(shorest_path);

#ifdef KSP_DEBUG
  std::cout << "start shortest path: " << "\n";
  FOR_EACH(shorest_path->getVertexes());
  std::cout << "\n";
#endif

  int remainK = k - 1;
  while(remainK > 0) {
    const auto &shortestVertexes = shorest_path->getVertexes();
    
#ifdef KSP_DEBUG
    std::cout << remainK << "   ,------------------------nnnnn---------------------\n";
    FOR_EACH(shortestVertexes);
#endif

    uint32_t vNum = shortestVertexes.size();
    for (int i = 0; i < vNum - 1; i++) {
      // to do dijkstra
     
      PathPtr viToDstShortestPath = DijkStraShortestPath(g, graph, shortestVertexes[i], end, std::vector<int>(shortestVertexes.begin(), shortestVertexes.begin() + i), kshortestPaths);
      
    
      if (viToDstShortestPath != nullptr){
        uint32_t weight1 = 0;
        for (uint32_t j = 0; j < i; j++) {
          weight1 += graph[shortestVertexes[j]][shortestVertexes[j + 1]];
        }
        Path prevPath(std::vector<int>(shortestVertexes.begin(), shortestVertexes.begin() + i), weight1);
        PathPtr newPath(new Path(prevPath, *viToDstShortestPath));

        
        if (!isSetContainsPath(candidatePaths, newPath)) {
          candidatePaths.insert(newPath);
        }
      }
    }

    // get k shortest path
    if (!candidatePaths.empty()) {
#ifdef KSP_DEBUG
      std::cout <<"current round k = " << remainK << " candidate size: " << candidatePaths.size() << "\n";
      for (auto candidate : candidatePaths) {
        FOR_EACH(candidate->getVertexes());
      }
#endif
    
      shorest_path = *candidatePaths.begin();
      candidatePaths.erase(candidatePaths.begin());
      kshortestPaths.insert(shorest_path);
      
#ifdef KSP_DEBUG
      std::cout << "FOR_EACH selected shortest\n";
      FOR_EACH(shorest_path->getVertexes());

      
      std::cout << "\nAfter erase candidates" << "\n";
      for (auto candidate : candidatePaths) {
        FOR_EACH(candidate->getVertexes());
      }
      //
      std::cout << "FOR_EACH shortest sets\n";
      for (auto sp : kshortestPaths) {
        FOR_EACH(sp->getVertexes());
      }
      std::cout << "k over....\n"; 
      std::cout << "------------------------nnnnn&&&&&&&&&&&&&&&&&&&\n";
#endif
      remainK--;
    } else { // no candidate path
      break;
    }
  }

  // std::cout << kshortestPaths.size() << "\n";
  int min_len = (*kshortestPaths.begin())->getVertexes().size();
  // std::vector<std::vector<int> > paths;
  for (auto p : kshortestPaths) {
      auto &nodes = p->getVertexes();
      pair_paths.push_back(nodes);
      // if (nodes.size() == min_len)
      //   FOR_EACH(nodes);
  }
  // sw_paths[start][end] = paths;
  // printf("\n");
  // avg_len = kshortestPaths.size();
  // avg_len / kshortestPaths.size();
  // if 
  // printf("start=%d end=%d paths=%d\n", start, end, kshortestPaths.size());
  return min_len;
}

void TestPathSet() {
  std::set<PathPtr, PathCompare> candidatePaths;
  std::set<PathPtr> kshortestPaths;
  PathPtr p1(new Path({1, 2, 5}, 8));
  PathPtr p2(new Path({1, 2, 4, 6}, 13));
  PathPtr p3(new Path({1, 2, 2, 2, 6}, 13));
  PathPtr p4(new Path({1, 2, 14}, 17));
  candidatePaths.insert(p1);
  candidatePaths.insert(p2);
  candidatePaths.insert(p3);
  candidatePaths.insert(p4);

  for (auto p : candidatePaths) {
    auto &nodes = p->getVertexes();
    FOR_EACH(nodes);
  }
}

std::string VecToString(const std::vector<int>& vec) {
  std::string res = "";
  for (int i = 0; i < vec.size(); i++) {
    res += std::to_string(vec[i]);
    if (i < vec.size() - 1) res += " ";
  }
  return res;
}

// [start, end)
void ThreadCal(int k,  std::vector<std::vector<int> > sub_tasks, std::vector<std::vector<int> > g2, std::vector<std::vector<int> > graph) {
  clock_t st, ed;
  st = clock();
  for (auto task : sub_tasks) {
    int src = task[0], dst = task[1];
    std::vector<std::vector<int> > paths;
    TopKShortestPathAlgo(paths, g2, graph, src, dst, k);
    // modify shared variable
    std::unique_lock<std::mutex> lck(path_cal_mutex);
    finish_cal_flag += 1;
    sw_paths[src][dst] = paths;
    // printf("%d -> %d : paths: %d\n", src, dst, paths.size());
    finish_cal_cv.notify_one();
  }
  ed = clock();
  // printf("tid = %ld run: %ds CPU=%d\n", this_thread::get_id(), (ed - st) / CLOCKS_PER_SEC, sched_getcpu());
}

int host_per_switch, switch_num, sw2sw_link_num, total_hosts;
void ConstructSwitchConnMatrix(const std::string& mat_file, const std::string& path_dir) {
  std::ifstream ifs;
  ifs.open(mat_file.c_str());
  ifs >> host_per_switch >> switch_num >> sw2sw_link_num >> total_hosts;
  // std::cout << host_per_switch << ", " << "sw num: " << switch_num << " links: " << sw2sw_link_num << "\n";

  std::vector<std::vector<int> > graph(switch_num, std::vector<int>(switch_num, 0));
  
  std::vector<std::vector<int> > g2(switch_num, std::vector<int>(switch_num, 0));
  
  for (int i = 0; i < switch_num; i++)
    for (int j = 0; j <switch_num; j++)
      if (i != j)
        graph[i][j] = INF;
  
  for (int i = 0; i < sw2sw_link_num; i++) {
    int a, b, p1, p2;
    ifs >> a >> b >> p1 >> p2;
    graph[a][b] = 1;
  }
  //
  g2 = graph;
  // std::unordered_map<int, std::unordered_map<int, std::vector<std::vector<int> > > > sw_paths;
  clock_t st, ed;
  st = clock();
  double sum_len = 0;
  int K = 30;
  
  int count = 0;
  std::vector<std::vector<int> > tasks;
  for (int i = 0; i < switch_num; i++) {
    for (int j = i + 1; j < switch_num; j++) {
      count += 1;
      // printf("%d -> %d : task\n", i, j);
      tasks.push_back({i, j});
    }
  }
  // 
  // std::cout << tasks.size() << "\n";
  std::unordered_map<int, std::vector<std::vector<int> > > split_tasks;
  std::vector<int> ids {0};
  int total_task = tasks.size();
  for (int i = 1; i <= THREAD_NUM - 1; i++) {
    ids.push_back(i * (total_task / THREAD_NUM));
  }
  if (ids[ids.size() - 1] < total_task) ids.push_back(total_task);
  // FOR_EACH(ids);
  // std::cout << ids.size() << "\n";
  int ii = 0;
  for (int i = 0; i < THREAD_NUM; i++) {
    std::vector<std::vector<int> > r;
    for (int j = ids[i]; j < ids[i+1]; j++) {
      r.push_back(tasks[j]);
      ii += 1;
    }
    split_tasks[i] = r;
  }


  std::vector<std::unique_ptr< std::thread > > threads;
  for (int i = 0; i < THREAD_NUM; i++) {
    threads.push_back(std::unique_ptr<std::thread>(new std::thread(ThreadCal, K, split_tasks[i], g2, graph)));
  }

  // bind thread to CPU
  for (int i = 0; i < threads.size(); i++) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    pthread_setaffinity_np(threads[i]->native_handle(), sizeof(cpuset), &cpuset);
  }

  for (const auto & th : threads) {
    th->join();
  }
  std::unique_lock<std::mutex> lck(path_cal_mutex);
  // wait until all calculation threads have finished
  while (finish_cal_flag != count ) finish_cal_cv.wait(lck);
  
  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      if (j >= i) continue;
      std::vector<std::vector<int> > i_to_j_paths; 
      for (auto p : sw_paths[j][i]) {
        std::reverse(p.begin(), p.end());
        i_to_j_paths.push_back(p);
      }
      sw_paths[i][j] = i_to_j_paths;
    }
  }
  ed = clock();

  // the main need to wait until all calculation threads have finished
  std::ofstream ofs(path_dir);
  
  char strBuf[256];
  ifs.close();
  sprintf(strBuf, "%d %d %d\n", switch_num * (switch_num - 1), host_per_switch, switch_num);
  ofs << strBuf;

  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      if (i == j) continue;
      sprintf(strBuf, "%d %d %d\n", i, j, sw_paths[i][j].size());
      ofs << strBuf;

      for (const auto &path : sw_paths[i][j]) {
        sprintf(strBuf, "%d %s\n", path.size(), VecToString(path).c_str());
        ofs << strBuf;
      }
    }
  }
  // std::cout << "KSP RUNNING " << (double)(ed - st) / CLOCKS_PER_SEC << "s\n";
  // std::cout << "Avg shortest path len: " << sum_len / (switch_num * switch_num) << "\n";
  ifs.close();
}

int main(int argc,char **argv) {
  if (argc < 3) {
    std::cerr << "ERROR !!! USAGE: ./edst [MAT_DIR] [PATH_DIR]\n";
    exit(-1);
  }
  std::string mat_dir(argv[1]);
  std::string path_dir(argv[2]);
  ConstructSwitchConnMatrix(mat_dir, path_dir);
  return 0;
}