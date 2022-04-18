#ifndef _SACK_BLOCK_SET_H
#define _SACK_BLOCK_SET_H

#include "ns3/sack-block.h"
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <set>
#include <stdint.h>

/**
 * Reference: https://tools.ietf.org/html/rfc2018   SACK
 */
namespace ns3 {

class SackBlockSet {
public:

  SackBlockSet();
  uint32_t DetermineRecvSeqNum(uint32_t seq, uint32_t psize);
  void add(uint32_t seq, uint32_t psize);
  std::string toString();
  uint32_t GetBlockNum() const;
  std::vector<SackBlockPtr> GetBlocks();
  
private:
  // merge two contigious blocks to a new block.
  SackBlockPtr merge(SackBlockPtr blockA, SackBlockPtr blockB);
private: 
  struct SackBlockCompare {
    bool operator()(const SackBlockPtr left, const SackBlockPtr right) const {
      return left->GetLeftEdge() > right->GetRightEdge();
    }
  };
  std::unordered_map<uint32_t, SackBlockPtr> m_lowerBoundToBlock;
  std::unordered_map<uint32_t, SackBlockPtr> m_upperBoundToBlock;
  std::set<SackBlockPtr, SackBlockCompare> m_blocks;
};

};
#endif

