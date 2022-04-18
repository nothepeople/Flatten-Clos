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


#include "gurobi_c++.h"

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
    int k_level_clos;
    std::vector<int> port_range;
    ifs >> host_per_switch >> switch_num >> sw2sw_link_num >> total_hosts;
    ifs >> k_level_clos;
    port_range.resize(k_level_clos * 2, 0);
    for (int i = 0; i < k_level_clos * 2; i ++) {
      ifs >> port_range[i];
    }
    
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

#ifdef EDST_DEBUG
    for (int i = 0; i < m_vertexNum; i++) {
      for (int j = 0; j < m_vertexNum; j++) {
        if (i == j) continue;
        printf("%d->%d paths: %lu\n", i, j, sw_paths[i][j].size());
        for (auto & path : sw_paths[i][j]) {
          FOR_EACH(path);
        }
        printf("\n");
      }
    }
#endif
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


//  g++ lp.cc -std=c++11 -I /home/qz/gurobi912/linux64/include/ -L /home/qz/gurobi912/linux64/lib/ -lgurobi_c++ -lgurobi91

void CalculateWeightfraction(std::unordered_map<uint32_t, 
                              std::unordered_map<uint32_t, 
                                std::vector<std::vector<int> > > >& pair_paths, 
                                const std::string& flow_file="flow.txt", const std::string &mat_file="demo_mat") {
  //
  std::unordered_map<int, std::unordered_map<int, std::vector<double> > > weights;
  
  std::ifstream ifs;
  std::vector<std::vector<int> > graph;
  // read mat_file
  ifs.open(mat_file.c_str());
  int host_per_switch, switch_num, sw2sw_link_num, total_hosts;
  int k_level_clos;
  std::vector<int> port_range;
  ifs >> host_per_switch >> switch_num >> sw2sw_link_num >> total_hosts;
  ifs >> k_level_clos;
  port_range.resize(k_level_clos * 2, 0);
  for (int i = 0; i < k_level_clos * 2; i ++) {
    ifs >> port_range[i];
  }
  graph.resize(switch_num, std::vector<int>(switch_num, 0));
  for (int i = 0; i < switch_num; i++) {
    graph[i][i] == 0;
  }

  for (int i = 0; i < sw2sw_link_num; i++) {
    int n1, n2, p1, p2;
    ifs >> n1 >> n2 >> p1 >> p2;
    graph[n1][n2] = 1;
  }
  ifs.close();
  printf("read matrix graph over...\n");
  
  // read flow file
  ifs.open(flow_file.c_str());
  int flow_num;
  ifs >> flow_num;
  std::vector<std::vector<int> > traffic_matrix;
  traffic_matrix.resize(switch_num, std::vector<int>(switch_num, 0));
  std::cout << host_per_switch << ", " << switch_num << "\n";
  for (int i = 0; i < flow_num; i++) {
    //
    int src_host, dst_host, src_port, dst_port, size, flag;
    double time;
    ifs >> src_host >> dst_host >> src_port >> dst_port >> size >> time >> flag;
    traffic_matrix[src_host / host_per_switch][dst_host / host_per_switch] += size;
  }
  //

  double max = 0;
  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      if (traffic_matrix[i][j] == 0) continue;
      if (max < traffic_matrix[i][j]) max = traffic_matrix[i][j];
      // std::cout << i << "->" << j << "traffic is: " << traffic_matrix[i][j] * 8 / 1e9 << "Gb\n";
    }
  }
  std::cout << "max Gbps: " << max * 8/ 1e9 << "\n";

  ifs.close();
  printf("read flow file over...\n");

  /*********************
   * Gurobi Model
   ********************/
  GRBEnv *env = new GRBEnv();
  GRBModel model = GRBModel(*env);
  
  GRBVar mlu = model.addVar(0.0, 1.0, 1.0, GRB_CONTINUOUS, "mlu");
  model.setObjective(1 * mlu, GRB_MINIMIZE);
  std::unordered_map<int, std::unordered_map<int, GRBVar * > > weight_fractions;
  std::unordered_map<int, std::unordered_map<int, GRBLinExpr * > > exprs;
  
  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      int path_num = pair_paths[i][j].size();
      if (path_num == 0) continue;
      double* ubs = new double[path_num];
      for (int k = 0; k < path_num; k++) ubs[k] = 1.0;
      weight_fractions[i][j] = model.addVars(NULL, ubs, NULL, NULL, NULL, path_num);
      delete ubs;
    }
  }

  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      if (graph[i][j] == 1) {
        exprs[i][j] = new GRBLinExpr();
      } else {
        exprs[i][j] = nullptr;
      }
    }
  }

  // add constraints
  for (int i = 0; i < switch_num; i++) { // src_host
    for (int j = 0; j < switch_num; j++) { // dst_host
      if (i == j) continue;
      GRBLinExpr constr = 0;
      for (int k = 0; k < pair_paths[i][j].size(); k++) {
        constr += weight_fractions[i][j][k];
      }
      model.addConstr(constr, GRB_EQUAL, 1);
      // printf("%d %d add weight==1 constraints\n", i, j);
    }
  }

  // flow traverse to single link cannot exceeds bandwidth * MLU
  for (int i = 0; i < switch_num; i++) { // src_host
    for (int j = 0; j < switch_num; j++) { // dst_host
      if (pair_paths[i][j].size() == 0) continue;
      const auto &paths = pair_paths[i][j];
      for (int k = 0; k < paths.size(); k++) { // i -> j has path num's path
        for (int l = 0; l < paths[k].size() - 1; l++) {
          // edge: paths[k][l]->paths[k][l+1]
          *exprs[ paths[k][l] ][ paths[k][l + 1] ] += weight_fractions[i][j][k] * traffic_matrix[i][j];
        }
      }
    }
  }
  

  GRBLinExpr link_limit = 1e9 / 8 * 0.2 * mlu;
  // std::cout << link_limit << "\n";

  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      if (graph[i][j] == 1 && exprs[i][j] != nullptr) {
        model.addConstr( *exprs[i][j], GRB_LESS_EQUAL, link_limit);
      }
    }
  }


  // std::cout << "MLU: " << mlu << "\n";
  std::cout << "before optimize...\n";
  try {
    model.optimize();
    if(model.get(GRB_IntAttr_Status) != GRB_OPTIMAL){
      std::cout << "not optimal " << std::endl;
    }
  } catch (GRBException e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
  }

  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      if (i == j || traffic_matrix[i][j] == 0) continue;
      int path_num = pair_paths[i][j].size();
      for (int k = 0; k < path_num; k++) {
        weights[i][j].push_back(weight_fractions[i][j][k].get(GRB_DoubleAttr_X));
      }
      printf("%d->%d tm=%d ", i, j, traffic_matrix[i][j]);
      FOR_EACH(weights[i][j]);
    }
  }


  // Prevent memory leaks
  for (int i = 0; i < switch_num; i++)
    for (int j = 0; j < switch_num; j++)
      if (weight_fractions[i][j] != NULL) {
        delete weight_fractions[i][j];
      }
  
  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      if (graph[i][j] == 1) {
        // std::cout << "(" << i << ", " << j << "): " << (*exprs[i][j]) << "\n";
        delete exprs[i][j];
      }
    }
  }

  delete env;
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

  CalculateWeightfraction(sw_paths);
}




int main() {
  testRouteMgr();
  return 0;
}