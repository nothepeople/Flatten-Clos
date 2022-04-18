#include <vector>
#include <iostream>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <memory>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <stack>
#include <fstream>

#define INF 0xfffffff

#define FOR_EACH(vec) {\
      for (int l = 0; l < vec.size(); l++) { \
          if (l == vec.size() - 1) { if (vec[l] == INF) std::cout << "INF"; else std::cout << vec[l];} \
          else { if (vec[l] == INF) std::cout << "INF "; else std::cout << vec[l] << " "; } } \
          std::cout << "\n";\
      }
#define K 6
#define I_PLUS(i) ((i % K) + 1)

// #define EDST_DEBUG


int parent(int x, std::vector<int>& nodes) {
  auto it = std::find(nodes.begin(), nodes.end(), x);
  if (it == nodes.begin()) {
    return *it;
  } else if (it == nodes.end()) {
    return -1;
  } else {
    return *(--it);
  }
}

class UnionFindSet {
public:
  UnionFindSet():m_setSize(0) { parent.clear(); }

  void init(int _s) {
    m_setSize = _s;
    parent.resize(m_setSize, -1);
    // ranks.resize(m_setSize, 1);
    for(int i = 0; i < m_setSize; i++) {
      parent[i] = i;
    }
    // FOR_EACH(parent);
  }

  int find(int x) {
    // std::cout << "enter find: ";
    // FOR_EACH(parent);
    while(x != parent[x]) {
      x = parent[x];
      // std::cout << " find: ";
      // FOR_EACH(parent);

    }
    return x;
  }

  int Union(int from, int to) { // union (from set) to (to set)
    int x = find(from);
    int y = find(to);
    parent[x] = y;
  }

  int erase(int from, int to) {
    int x = find(from);
    int y = find(from);
    // if (x == y) {
    //   parent[from] = x;
    //   parent[to] = to;
    // }
  }
private:
  int m_setSize;
  std::vector<int> parent;
  std::vector<int> ranks;
};

class Edge;
typedef std::shared_ptr<Edge> EdgeSharedPtr;

class Edge {
public:
  Edge(int _from, int _to, int _weight=1) 
    : m_from(_from), 
    m_to(_to), 
    m_weight(_weight),
    m_label(nullptr),
    m_forestIndex(0) {}

  Edge()  
    : m_from(-1), 
    m_to(-1), 
    m_weight(-1),
    m_label(nullptr),
    m_forestIndex(0) {}

#ifdef EDST_DEBUG
  ~Edge() { std::cout << " delete edge: " << toString() << "\n"; }
#endif

  void Print() const { printf("%d->%d: %d\n", m_from, m_to, m_weight); }
  
  std::string toString() const { 
    char buf[20];
    snprintf(buf, 20, "(%d, %d)", m_from, m_to);
    return std::string(buf);
  }

  std::vector<int> getEndPoints() {
    return {m_from, m_to};
  }

  bool isLabeled() {
    return m_label != nullptr;
  }

  void markLabel(EdgeSharedPtr label=nullptr) {
    m_label = label;
  }

  EdgeSharedPtr getLabel() {
    return m_label;
  }

  void setForestIndex(int i = -1) {
    if (i == -1) {
      return;
    }
    m_forestIndex = i;
  }

  int getForestIndex() {
    return m_forestIndex;
  }

public:
  /** data member **/
  int m_from;
  int m_to;
  int m_weight;
  int m_forestIndex;
  EdgeSharedPtr m_label;
};


struct EdgePtrCompare {
  bool operator()(EdgeSharedPtr left, EdgeSharedPtr right) const {
    return left->m_weight < right->m_weight || 
            left->m_weight == right->m_weight && left->m_from < right->m_from || 
            left->m_weight == right->m_weight && left->m_from == right->m_from && left->m_to < right->m_to;
  }
};


class Graph {
public:

  Graph() { m_adjacentMatrix.clear(); }

  void ConstructGraph(const std::string &mat_file= "demo_mat") {
    std::ifstream ifs;
    ifs.open(mat_file.c_str());
    int host_per_switch, switch_num, sw2sw_link_num, total_hosts;

 
    ifs >> host_per_switch >> switch_num >> sw2sw_link_num >> total_hosts;

    m_vertexNum = switch_num;
    
    m_adjacentMatrix.resize(m_vertexNum, std::vector<int>(m_vertexNum, INF));
    for (int i = 0; i < m_vertexNum; i++) {
      m_adjacentMatrix[i][i] == 0;
    }

    /*** A demo graph ***/
#ifdef EDST_DEBUG
    std::vector<std::vector<int> > demo = {
      {0, 1, 1, 1, 1},
      {1, 0, 1, INF, 1},
      {1, 1, 0, 1, INF},
      {1, INF, 1, 0, 1},
      {1, 1, INF, 1, 0}
    };

    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 5; j++) {
        m_adjacentMatrix[i][j] = demo[i][j];
      }
    }
#endif
    for (int i = 0; i < sw2sw_link_num; i++) {
      int n1, n2, p1, p2;
      ifs >> n1 >> n2 >> p1 >> p2;
      m_adjacentMatrix[n1][n2] = 1;
    }
    ifs.close();
    printf("read graph over...\n");

  }

  int getVertexes() const {
    return m_vertexNum;
  }

  void insertNonDecreasingEdges(std::set<EdgeSharedPtr, EdgePtrCompare > & edges) {
    for (int i = 0; i < m_vertexNum; i++)
      for (int j = 0; j < m_vertexNum; j++)
        if (j > i && m_adjacentMatrix[i][j] == 1) {
          edges.insert(std::shared_ptr<Edge>(new Edge(i, j, 1) ) );
        }
  }

private:
  int m_vertexNum; // from zero to m_vertexNum
  std::vector<std::vector<int> > m_adjacentMatrix;
};



class Forest {
public:
  Forest(int vertexes, int id) : m_partition(new UnionFindSet() ), m_vertexNum(vertexes), m_forestId(id) { 
    m_edges.clear();
    
    m_partition->init(m_vertexNum); 
    m_edgeMap.clear();
    m_forest.resize(m_vertexNum, std::vector<int>(m_vertexNum, INF) );
    for (int i = 0; i < m_vertexNum; i++) {
      m_forest[i][i] = 0;
    }
  }

  void addEdge(EdgeSharedPtr e) { 
    m_edges.push_back(e); 
    int a = e->getEndPoints()[0], b = e->getEndPoints()[1];
    e->setForestIndex(m_forestId);
    m_edgeMap[a][b] = e;
    m_edgeMap[b][a] = e;

    m_forest[a][b] = 1;
    m_forest[b][a] = 1;
    Union(a, b);
  }

  void eraseEdge(EdgeSharedPtr e) {
    //
    e->setForestIndex(0);
    auto it = std::find(m_edges.begin(), m_edges.end(), e);
    if (it != m_edges.end()) {
      m_edges.erase(it);
    }
    int a = e->getEndPoints()[0], b = e->getEndPoints()[1];
    m_edgeMap[a][b] = nullptr;
    m_edgeMap[b][a] = nullptr;
    //
    if (a != b) {

      m_forest[a][b] = INF;
      m_forest[b][a] = INF;
    }
  }

  std::vector<EdgeSharedPtr>& getEdges() { return m_edges; }


  void passbyPath(std::vector<int>& passByNodes, 
                  std::vector<EdgeSharedPtr> passByEdges, 
                  int start, int end) {
    std::vector<bool> visited(m_vertexNum, false);
    std::vector<int> paths;
    visited[start] = true;
    // printf("MATRIX:\n");
    // for (auto row : m_forest) {
    //   FOR_EACH(row);
    // }
    paths.push_back(start);
    uniquePath(passByNodes, passByEdges, paths, visited, start, end, start);
  }

  int find(int x) {
    return m_partition->find(x);
  }

  void Union(int from, int to) {
    m_partition->Union(from, to);
  }

  void markLabel(int x, int y, EdgeSharedPtr label=nullptr) {
    m_edgeMap[x][y]->markLabel(label);
  }

  bool isLabeled(int x, int y) {
    return m_edgeMap[x][y]->isLabeled();
  }

  EdgeSharedPtr getMappedEdge(int x, int y) {
    return m_edgeMap[x][y];
  }

  void clearEdgeLabel() {
#ifdef EDST_DEBUG
    std::cout << "clearing forest: " << m_forestId << " edges num: " << m_edges.size() << "\n";
#endif
    for (auto edge : m_edges) {
      edge->markLabel(nullptr);
    }
  }

  void DebugForest() {
    for (auto e : m_edges) {
      std::cout << "forest: " << m_forestId << " edge: " << e->toString() << "\n";
    }
  }

  void DebugLabel() {
    for (auto e : m_edges) {
      std::cout << "forest: " << m_forestId << " edges: " << e->toString() << " labeling: " << e->isLabeled() << "\n";
    }
  }



  const std::vector<std::vector<int> >& getForestMatrix() {
    return m_forest;
  }


private:

  void uniquePath(std::vector<int>& passByNodes,
                  std::vector<EdgeSharedPtr>& passByEdges,
                  std::vector<int>& paths, 
                  std::vector<bool>& visited,
                  int src, int dst, int curNode) {
    if (curNode == dst) {
      // std::cout << "find\n";
      // FOR_EACH(paths);
      passByNodes.assign(paths.begin(), paths.end());
      for (int i = 0; i < paths.size() - 1; i++) {
        passByEdges.push_back(m_edgeMap[paths[i]][paths[i + 1]]);
      }
      return;
    }
    for (int i = 0; i < m_vertexNum; i++) {
      if (visited[i] || m_forest[curNode][i] == INF) {
        continue;
      }
      visited[i] = true;
      paths.push_back(i);
      // FOR_EACH(paths)
      uniquePath(passByNodes, passByEdges, paths, visited, src, dst, i);
      visited[i] = false;
      paths.pop_back();
    }
  }

private:
  std::shared_ptr<UnionFindSet > m_partition; // test whether two points are in the same tree
  std::unordered_map<int, std::unordered_map<int, EdgeSharedPtr> > m_edgeMap;
  std::vector<EdgeSharedPtr > m_edges; 
  std::vector<std::vector<int> > m_forest; // may have many trees.
  int m_vertexNum;
  int m_forestId;
};


class MinimumSpanningTreesSolver {
  //
public:
  MinimumSpanningTreesSolver() { }

  MinimumSpanningTreesSolver(const std::string& mat_file, int k = K)
  : m_g(new Graph()),
    m_partitionZero(std::unique_ptr<UnionFindSet>(new UnionFindSet() ) ) {
    
    m_g->ConstructGraph(mat_file);
    m_partitionZero->init(m_g->getVertexes());
    // initilization
    m_K = k;
    std::cout << "m_k" << k << "\n";
    for (int i = 0; i < m_K; i++) {
      m_KForests.push_back( std::shared_ptr<Forest> (new Forest(m_g->getVertexes(), i + 1 ) ) );
    }
    m_g->insertNonDecreasingEdges(m_incraseingEdges);

#ifdef EDST_DEBUG
    std::cout << "forests num: " << m_KForests.size() << "\n";
    // initialize
    for (auto &inc : m_incraseingEdges) {
      std::cout << inc->toString() << "\n";
    }
    printf("construct graph over\n\n");
#endif
  }

  void solve() {
  //
  
    while(!m_incraseingEdges.empty()) {
      EdgeSharedPtr e = *(m_incraseingEdges.begin());
      m_incraseingEdges.erase(m_incraseingEdges.begin());
      std::vector<int> endPoints = e->getEndPoints();

      if (0) { //  process next edge
        continue;
      } else {
        labeling(e);
        if (!m_agumentingSequence.empty()) {
          agumenting();
        } else {
          m_partitionZero->Union(endPoints[0], endPoints[1]);
        }
      }
    }

  }

  void UpdateEdgeDisjointForests();

  void labeling(EdgeSharedPtr eRoot) {
    // clearing edges' label of forest
    clearLabel();
    // debug();
  #ifdef EDST_DEBUG
    labelingDebug();
  #endif

    while (!m_q.empty())
      m_q.pop();
    
    m_agumentingSequence.clear();

    // initialization
    m_q.push(eRoot);
    // forest initialization
    while(1) {
      if (m_q.empty()) {
        break;
      }
      EdgeSharedPtr e = m_q.front();
      m_q.pop();
      // std::cout << "label edge: " << e->toString() << "\n";
      const std::vector<int>& endPoints = e->getEndPoints();
      int x = endPoints[0];
      int y = endPoints[1];
      
      int i = e->getForestIndex();
      int iplus = I_PLUS(i) - 1;
  #ifdef EDST_DEBUG
      std::cout << "edge: " << e->toString() << " forestIdx= " << i << " iplus: " << iplus << "\n";
  #endif

      if  (m_KForests[iplus]->find(x) != m_KForests[iplus]->find(y)) {
        // agumenting sequence
        traceOutAgumentationSequence(eRoot, e);
        return;
      } else {
        std::vector<int > nodes;
        std::vector<EdgeSharedPtr > edges;  
        m_KForests[iplus]->passbyPath(nodes, edges, x, y);
        for (int i = 1; i < nodes.size(); i++) {
          int u = nodes[i];
          int p = parent(u, nodes);
          
          if (!m_KForests[iplus]->isLabeled(u, p)) {
            m_KForests[iplus]->markLabel(u, p, e);
            m_q.push(m_KForests[iplus]->getMappedEdge(u, p));
          }
        }

      }
    }
    return;
  }


  void agumenting() {
  // add to forest
    if (m_agumentingSequence.size() == 1) {
      //
      m_KForests[0]->addEdge(m_agumentingSequence[0]);
    } else {
      //
      int l = m_agumentingSequence.size() - 1;
      for (int i = 0; i < l; i++) {
        int iplus = I_PLUS(i) - 1;
        m_KForests[iplus]->addEdge(m_agumentingSequence[i]);
        m_KForests[iplus]->eraseEdge(m_agumentingSequence[i + 1]);
      }
      m_KForests[I_PLUS(l) - 1]->addEdge(m_agumentingSequence[l]);
    }
  #ifdef EDST_DEBUG
    printf("after augmenting() forests\n");
    debug();
    printf("\n");
  #endif
  }

  int getVertexes() {
    return m_g->getVertexes();
  }

  const std::vector<std::vector<int > >& forestMatrix(int forestIdx) {
    return m_KForests[forestIdx]->getForestMatrix();
  }

private:

  void clearLabel() {
    for (auto forest : m_KForests) {
      forest->clearEdgeLabel();
    }
  }


  void traceOutAgumentationSequence(EdgeSharedPtr e, EdgeSharedPtr e_win) {
    int j = 0;
    m_agumentingSequence.push_back(e_win);
    while(m_agumentingSequence[j] != e) {
      m_agumentingSequence.push_back(m_agumentingSequence[j]->getLabel());
      j++;
    }
    std::reverse(m_agumentingSequence.begin(), m_agumentingSequence.end());
  }

  void debug() {
    for (auto forest : m_KForests) {
      forest->DebugForest();
    }
  }

  void labelingDebug() {
    for (auto forest : m_KForests) {
      forest->DebugLabel();
    } 
  }



private:
  std::unique_ptr<UnionFindSet>  m_partitionZero;

  std::vector<std::shared_ptr<Forest > > m_KForests;
  std::unique_ptr<Graph> m_g;
  std::set<EdgeSharedPtr, EdgePtrCompare > m_incraseingEdges;
  std::queue<EdgeSharedPtr > m_q;
  std::vector<EdgeSharedPtr > m_agumentingSequence; 
  int m_K;
};



class RouteManager {
public:
  RouteManager(const std::string &mat_file) 
    : m_mat_file(mat_file), m_solver(new MinimumSpanningTreesSolver(mat_file, K)) {}

  void calPath(std::unordered_map<uint32_t, 
	                std::unordered_map<uint32_t, 
		                std::vector<std::vector<int> > > >& sw_paths) {
    m_solver->solve();
    m_vertexNum = m_solver->getVertexes();
    
    for (int f = 0; f < K; f++) {
      for (int i = 0; i < m_vertexNum; i++) {
        dijkstraShortestPath(m_solver->forestMatrix(f), sw_paths, i);
      }
    }

// #ifdef EDST_DEBUG
    int pairs = 0;
    double minSumPathLen = 0;
    double sumAvgPathLen = 0;
    double sumPaths = 0;
    for (int i = 0; i < m_vertexNum; i++) {
      for (int j = 0; j < m_vertexNum; j++) {
        if (i == j) continue;
        ++pairs;
        sumPaths += sw_paths[i][j].size();
        
        int mini = 0xff;
        double tmpSum = 0;
        for (auto & path : sw_paths[i][j]) {
          // FOR_EACH(path);
          if (mini > path.size()) mini = path.size();
          tmpSum += path.size();
        }
        minSumPathLen += mini;
        sumAvgPathLen += (double)tmpSum / sw_paths[i][j].size();
        // printf("\n");
      }
    }
// #endif

    printf("Avg paths: %.2f, avg path lens: %.2f, avg minimum path lens: %.2f\n", sumPaths / pairs, sumAvgPathLen / pairs, minSumPathLen / pairs);
  };

private:
  void dijkstraShortestPath(const std::vector<std::vector<int> >& forest, 
                          std::unordered_map<uint32_t, 
	                          std::unordered_map<uint32_t, 
		                          std::vector<std::vector<int> > > >& sw_paths, 
                          int start) {
    std::vector<int> dis(m_vertexNum, INF);
    std::vector<int> prev(m_vertexNum, -1);
    std::vector<int> visited(m_vertexNum, 0);
    dis[start] = 0;
    prev[start] = start;
    // visited[start] = 1;

    typedef std::pair<int, int> DisVexPair;
    std::priority_queue<DisVexPair, std::vector<DisVexPair>, std::greater<DisVexPair> > pq;
    pq.push({dis[start], start});
    while(!pq.empty()) {
      auto p = pq.top();
      pq.pop();
      int u = p.second;
      // std::cout << "u " << u << "dis[u]: " << dis[u] << "\n";
      if (visited[u])
        continue;
      visited[u] = 1;
      for (int v = 0; v < m_vertexNum; v++) {
        if (!visited[v] && dis[v] >= dis[u] + forest[u][v]) {
          dis[v] = dis[u] + forest[u][v];
          prev[v] = u;
          pq.push({dis[v], v});
        }
      }
    }

    for (int end = 0; end < m_vertexNum; end++) {
      if (end == start) continue;
      std::vector<int> path;
      int temp = end;
      while(prev[temp] != temp) {
        path.push_back(temp);
        temp = prev[temp];
      }
      path.push_back(start);
      std::reverse(path.begin(), path.end());
      sw_paths[start][end].push_back(path);
    }
  }
private:
  int m_vertexNum;
  const std::string& m_mat_file;
  std::shared_ptr<MinimumSpanningTreesSolver> m_solver;
};


void testParent() {
  std::vector<int> a = {1, 2, 3};
  printf("parent of %d is %d\n", 1, parent(1, a));
  printf("parent of %d is %d\n", 2, parent(2, a));
  printf("parent of %d is %d\n", 3, parent(3, a));
}

void testUnionFindSet() {
  UnionFindSet ufset;
  ufset.init(5);

  printf("hi\n");
  ufset.Union(0, 1);
  printf("hi\n");
  ufset.Union(0, 2);
  printf("hi\n");
  ufset.Union(0, 3);
  printf("hi\n");
  ufset.Union(0, 4);
  int x = ufset.find(1);
  int y = ufset.find(2);
  std::cout << x << ", " << y << "\n";
}

std::string VecToString(const std::vector<int>& vec) {
  std::string res = "";
  for (int i = 0; i < vec.size(); i++) {
    res += std::to_string(vec[i]);
    if (i < vec.size() - 1) res += " ";
  }
  return res;
}

void testRouteMgr() {
  std::unordered_map<uint32_t, 
	  std::unordered_map<uint32_t, 
		  std::vector<std::vector<int> > > > sw_paths; // sw id start from zero
  clock_t bg, ed;
  RouteManager mgr("demo_mat");
  bg = clock();
  mgr.calPath(sw_paths);
  ed = clock();
  std::cout << "Runinng: " << (double)(ed - bg) / CLOCKS_PER_SEC << "\n";

  std::string edst_uniform_path_dir = "../experiment/topo/edst-uniform/64/path.txt";
  std::string edst_lp_path_dir = "../experiment/topo/edst-lp/64/path.txt";
  std::string edst_weighted_path_dir = "../experiment/topo/edst-weighted/64/path.txt";

  std::ofstream edst_uniform_ofs(edst_uniform_path_dir),
     edst_lp_ofs(edst_lp_path_dir), 
     edst_weighted_ofs(edst_weighted_path_dir);
  
  char strBuf[256];
  int switches=64, to_host=4;
  sprintf(strBuf, "%d %d %d\n", switches * (switches - 1), to_host, switches);
  edst_uniform_ofs << strBuf;
  edst_lp_ofs << strBuf;
  edst_weighted_ofs << strBuf;

  for (int i = 0; i < switches; i++) {
    for (int j = 0; j < switches; j++) {
      if (i == j) continue;
      sprintf(strBuf, "%d %d %d\n", i, j, sw_paths[i][j].size());
      edst_uniform_ofs << strBuf;
      edst_lp_ofs << strBuf;
      edst_weighted_ofs << strBuf;
      for (const auto &path : sw_paths[i][j]) {
        sprintf(strBuf, "%d %s\n", path.size(), VecToString(path).c_str());
        edst_uniform_ofs << strBuf;
        edst_lp_ofs << strBuf;
        edst_weighted_ofs << strBuf;
      }
    }
  }
  return;
}

int main() {
  //
  std::vector<std::vector<int> > graph = {
    {0, 1, INF, INF},
    {1, 0, 1, 1},
    {INF, 1, 0, INF},
    {INF, 1, INF, 0}
  };

  // // testParent();
 
  // MinimumSpanningTreesSolver solver("demo_mat");
  // solver.solve();
  testRouteMgr();
  // testUnionFindSet();
  return 0;
}