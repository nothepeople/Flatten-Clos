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
#include <string>


#include "ortools/linear_solver/linear_solver.h"

#define INF 0xfffffff

#define FOR_EACH(vec) {\
      for (int l = 0; l < vec.size(); l++) { \
          if (l == vec.size() - 1) { if (vec[l] == INF) std::cout << "INF"; else std::cout << vec[l];} \
          else { if (vec[l] == INF) std::cout << "INF "; else std::cout << vec[l] << " "; } } \
          std::cout << "\n";\
      }


//  g++ lp.cc -std=c++11 -I /home/qz/gurobi912/linux64/include/ -L /home/qz/gurobi912/linux64/lib/ -lgurobi_c++ -lgurobi91
namespace operations_research {

void CalculateWeightfraction(std::unordered_map<uint32_t, 
                              std::unordered_map<uint32_t, 
                                std::vector<std::vector<int> > > >& pair_paths, 
                                const std::string& flow_file="flow.txt", const std::string &mat_file="demo_mat", const std::string &output_file="weight.txt") {
  //
  // std::cout << flow_file << "\n";
  std::unordered_map<int, std::unordered_map<int, std::vector<double> > > weights;
  
  std::ifstream ifs;
  std::vector<std::vector<int> > graph;
  // read mat_file
  ifs.open(mat_file.c_str());
  int host_per_switch, switch_num, sw2sw_link_num, total_hosts;
  ifs >> host_per_switch >> switch_num >> sw2sw_link_num >> total_hosts;
  
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
  // printf("read matrix graph over...\n");
  
  // read flow file
  ifs.open(flow_file.c_str());
  int flow_num;
  ifs >> flow_num;
  std::vector<std::vector<int> > traffic_matrix;
  traffic_matrix.resize(switch_num, std::vector<int>(switch_num, 0));

  // std::cout << host_per_switch << ", " << switch_num << "\n";

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
  // std::cout << "max Gbps: " << max * 8/ 1e9 << "\n";

  ifs.close();
  // printf("read flow file over...\n");

  /*********************
   * LP Model
   ********************/
  std::unordered_map<int, std::unordered_map<int, std::vector<MPVariable* > > > weight_fractions;
  std::unordered_map<int, std::unordered_map<int, MPConstraint * > > constraints;
  
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("GLOP"));
  MPVariable* const mlu = solver->MakeNumVar(0.0, 200, "mlu");
  MPObjective* const objective = solver->MutableObjective();
  objective->SetCoefficient(mlu, 1);
  
  // initialize variable
  int path_len_upper_bound = 0;
  int loop = 0;
  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      int path_num = pair_paths[i][j].size();
      if (path_num == 0) continue;
      loop++;
      for (int k = 0; k < path_num; k++) {
        // max path len
        if (pair_paths[i][j][k].size() > path_len_upper_bound) path_len_upper_bound =  pair_paths[i][j][k].size();
        std::string name = std::to_string(i) + "_" + std::to_string(j) + "_" +  std::to_string(k);
        weight_fractions[i][j].push_back(solver->MakeNumVar(0.0, 1, name));
      }
    }
  }
  // printf("max path len %d, loops %d\n", path_len_upper_bound, loop);

  // for (int i = 0; i < switch_num; i++) {
  //   for (int j = 0; j < switch_num; j++) {
  //     int path_num = pair_paths[i][j].size();
  //     if (path_num == 0) continue;
  //     for (int k = 0; k < path_num; k++) {
  //       // max path len
  //       int path_len = pair_paths[i][j][k].size();
  //       objective->SetCoefficient(weight_fractions[i][j][k], (double)path_len / loop / path_len_upper_bound);
  //     }
  //   }
  // }

  double link_limit = 1e9 / 8 * 0.2;
  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      if (graph[i][j] == 1) {
        std::string constr_name = "edge constr:(" + std::to_string(i) + "," + std::to_string(j) + ")";
        constraints[i][j] = solver->MakeRowConstraint(-solver->infinity(), 0, constr_name);
        constraints[i][j]->SetCoefficient(mlu, -link_limit);
      } else {
        constraints[i][j] = nullptr;
      }
    }
  }

  // add constraints
  for (int i = 0; i < switch_num; i++) { // src_host
    for (int j = 0; j < switch_num; j++) { // dst_host
      if (i == j) continue;
      std::string constr_name = "assgin constr:" + std::to_string(i) + "_" + std::to_string(j);
      MPConstraint* const constr = solver->MakeRowConstraint(1.0, 1.0, constr_name);
      for (int k = 0; k < pair_paths[i][j].size(); k++)
        constr->SetCoefficient(weight_fractions[i][j][k], 1);
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
          constraints[ paths[k][l] ][ paths[k][l + 1] ]->SetCoefficient(weight_fractions[i][j][k], traffic_matrix[i][j]) ;
        }
      }
    }
  }
  
  objective->SetMinimization();
  // solve
  solver->Solve();
  std::cout << "flow_file: " << flow_file <<  " Objective value: " << objective->Value() << "\n";

  std::ofstream ofs;
  ofs.open(output_file.c_str(), std::ios::out);
  ofs << (switch_num - 1) * switch_num << "\n";
  for (int i = 0; i < switch_num; i++) {
    for (int j = 0; j < switch_num; j++) {
      if (i == j) continue;
      int path_num = pair_paths[i][j].size();
      // format is: [src dst path_num w1 w2 ,...] 
      ofs << i << " " << j << " " << path_num << " ";
      for (int k = 0; k < path_num; k++) {
        double solution_value = weight_fractions[i][j][k]->solution_value();
        if (solution_value > 1e-17 && solution_value < 1e-10) solution_value = 0.0;
        // write weight
        if (k == path_num - 1) {
          ofs << solution_value << "\n";
        } else {
          ofs << solution_value  << " ";
        }
      }
    }
  }
  ofs.close();

}

} // namespace operation_research


void ReadPath(const std::string& path_file,
              std::unordered_map<uint32_t, 
                std::unordered_map<uint32_t, 
                  std::vector<std::vector<int> > > >& pair_paths,
              uint32_t& switches,
              uint32_t& host_per_switch) {
	std::ifstream ifs;
	ifs.open(path_file);
	uint32_t pairs = 0, total_hosts;
	ifs >> pairs >> host_per_switch >> switches;
	total_hosts = switches * host_per_switch;
	printf("ToRs %d, hosts_per_ToR %d, total hosts %d\n", switches, host_per_switch, total_hosts);
	for (int i = 0; i < pairs; i++) {
		int src, dst, num_paths;
		ifs >> src >> dst >> num_paths;
		std::vector<std::vector<int> > paths;
		for (int j = 0; j < num_paths; j++) {
			std::vector<int> path;
			int path_nodes, n;
			ifs >> path_nodes;
			for (int k = 0; k < path_nodes; k++) {
				ifs >> n;
				path.push_back(n);
			}
			paths.push_back(path);
		}
		// std::shared_ptr<std::vector<std::vector<int> > > p(new std::vector<std::vector<int> >(paths));
		pair_paths[src][dst] = paths;
	}
	return;
}


/*******************
 * FLATTENED CLOS
 ******************/

void FcWriteWeightFraction(const std::string& mat_file, std::string& path_file) {
  
  std::unordered_map<uint32_t, 
    std::unordered_map<uint32_t, 
      std::vector<std::vector<int> > > > pair_paths;
  uint32_t host_per_switch, switches;
  
  // Read Paths 
  ReadPath( path_file, pair_paths, switches, host_per_switch);
  printf("FC, hosts per switch: %d switches: %d.\n", host_per_switch, switches);
  std::vector<int> client_racks = {4, 8, 12, 16, 20, 24, 28, 32};
  std::vector<int> server_racks = {4, 8, 12, 16, 20, 24, 28, 32};
  for (auto &client : client_racks) {
    for (auto &server : server_racks) {
      char buf1[100];
      char buf2[100];
      
      snprintf(buf1, 100, "../experiment/config/fc-lp/dcqcn/cs-model/64/pfc/web/C%d-S%d/weight.txt", client, server);
      snprintf(buf2, 100, "../experiment/flow/fc-lp/cs-model/web/C%d-S%d/flow.txt", client, server);
      
      std::string output_file(buf1);
      std::string flow_file(buf2);
      operations_research::CalculateWeightfraction(pair_paths, flow_file, mat_file, output_file);
      // std::cout << output_file << "\n" << flow_file << "\n" << mat_file << "\n";
    }
  }
  
  // skew traffic
  // std::vector<double> hot_rack_fractions = {0.04, 0.10, 0.20};
  // std::vector<double> hot_traffics = {0.25, 0.50, 0.75};
  // for (auto hot_rack : hot_rack_fractions) {
  //   for (auto hot_traffc : hot_traffics) {
  //     char buf1[100];
  //     char buf2[100];
  //     snprintf(buf1, 100, "../experiment/config/fc-lp/dcqcn/skew/64/pfc/web/%.2f/%.2f/weight.txt", hot_rack, hot_traffc);
  //     snprintf(buf2, 100, "../experiment/flow/fc-lp/skew/web/%.2f/%.2f/flow.txt", hot_rack, hot_traffc);
  //     std::string output_file(buf1);
  //     std::string flow_file(buf2);
  //     std::cout << output_file << "\n";
  //     std::cout << flow_file << "\n";
  //     operations_research::CalculateWeightfraction(pair_paths, flow_file, mat_file, output_file);
  //   }
  // }

}

/*******************
 ******* EDST ******
 ******************/
void EdstWriteWeightFraction(const std::string& mat_file, std::string& path_file) {
  std::unordered_map<uint32_t, 
	  std::unordered_map<uint32_t, 
		  std::vector<std::vector<int> > > > pair_paths; // sw id start from zero
  uint32_t switches, host_per_switch;
  std::cout << "EDST READ PATH...\n";
  ReadPath( path_file, pair_paths, switches, host_per_switch);

  std::vector<int> client_racks = {4, 8, 12, 16, 20, 24, 28, 32};
  std::vector<int> server_racks = {4 , 8, 12, 16, 20, 24, 28, 32};

  for (auto &client : client_racks) {
    for (auto &server : server_racks) {
      char buf1[100];
      char buf2[100];
      snprintf(buf1, 100, "../experiment/config/edst-lp/dcqcn/cs-model/64/pfc/web/C%d-S%d/weight.txt", client, server);
      snprintf(buf2, 100, "../experiment/flow/edst-lp/cs-model/web/C%d-S%d/flow.txt", client, server);
      std::string output_file(buf1);
      std::string flow_file(buf2);
      
      operations_research::CalculateWeightfraction(pair_paths, flow_file, mat_file, output_file);
    }
  }

  // std::vector<double> hot_rack_fractions = {0.04, 0.10, 0.20};
  // std::vector<double> hot_traffics = {0.25, 0.50, 0.75};
  // for (auto hot_rack : hot_rack_fractions) {
  //   for (auto hot_traffc : hot_traffics) {
  //     char buf1[100];
  //     char buf2[100];
  //     snprintf(buf1, 100, "../experiment/config/edst-lp/dcqcn/skew/64/pfc/web/%.2f/%.2f/weight.txt", 0.04, 0.25);
  //     snprintf(buf2, 100, "../experiment/flow/edst-lp/skew/web/%.2f/%.2f/flow.txt",  0.04, 0.25);
  //     std::cout << "buf1: " << buf1 << "\n";
  //     std::cout << "buf2: " << buf2 << "\n";
  //     std::string output_file(buf1);
  //     std::string flow_file(buf2);
  //     operations_research::CalculateWeightfraction(pair_paths, flow_file, mat_file, output_file);
  //   }
  // }
}

// ./waf --run "scratch/fc-lp mix/experiment/config/fc-lp/dcqcn/skew/64/pfc/web/0.04/0.25/config.txt"

// g++ lp-ortools.cc -std=c++11 -I ~/or-tools/include/ -L ~/or-tools/lib/ -lortools


int main() {
  std::string path_to_mat_file = "/home/zqz/hpcc/simulation/mix/topo-generator/demo_mat";
  std::string path_to_edst_path_file = "/home/zqz/hpcc/simulation/mix/experiment/topo/edst-lp/64/path.txt";
  std::string path_to_fc_path_file = "/home/zqz/hpcc/simulation/mix/experiment/topo/fc-lp/64/path.txt";
  FcWriteWeightFraction(path_to_mat_file, path_to_fc_path_file);
  // 
  std::cout << "Edst optimzier\n";
  EdstWriteWeightFraction(path_to_mat_file, path_to_edst_path_file);
  return 0;
}