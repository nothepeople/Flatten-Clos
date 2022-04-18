#ifndef _K_SHORTEST_PATH_H
#define _K_SHORTEST_PATH_H
#include <vector>
#include <list>
/**
 * @param g: graph uesd to test
 * @param graph: used to calculate shortest path
 * @param start: start point
 * @param end: end point
 * @param k: k shortest paths, result may smaller run k
 */
void TopKShortestPathAlgo(const std::vector<std::vector<int> >& g, std::vector<std::vector<int> >& graph, int start, int end, int k);
#endif