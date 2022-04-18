#include "sack-block-set.h"
#include <vector>
#include <assert.h>


namespace ns3 {

SackBlockSet::SackBlockSet() /* m_cacheInvalid(false) */ {
  m_lowerBoundToBlock.clear();
  m_upperBoundToBlock.clear();
  m_blocks.clear();
}


/**
 * Case 1: sende[0, 1000)Sequenc0，
 *sequenc1000
 * 
 * Case 2: sende[2000, 3000[3000, 4000) [5000, 6000，
 *sequenc4000400
 *
 * Case 3: sende[2000, 3000[4000, 5000)sequenc3000
 */
uint32_t SackBlockSet::DetermineRecvSeqNum(uint32_t seq, uint32_t psize) {
  uint32_t ackNum = seq + psize;
  uint32_t recvNum = ackNum;
  if (m_lowerBoundToBlock.count(ackNum)) {
    SackBlockPtr blockPtr = m_lowerBoundToBlock.at(ackNum);
    m_lowerBoundToBlock.erase(ackNum);
    m_upperBoundToBlock.erase(blockPtr->GetRightEdge());
    recvNum = blockPtr->GetRightEdge();
    m_blocks.erase(blockPtr);
  }
  return recvNum;
}

void SackBlockSet::add(uint32_t seq, uint32_t psize) {
  // make sure param is valid
  assert(seq >= 0 && psize >= 0);
  for (auto block : m_blocks) {
    if (seq >= block->GetLeftEdge() && seq + psize <= block->GetRightEdge()) {
      return;
    }
  }
  // cache [100 200) [300, 400) 
  // arrive[200, 300) 
  SackBlockPtr block = std::make_shared<SackBlock>(seq, seq + psize);
  
  // first merge[100, 200) + [200, 300) = [100, 300)
  SackBlockPtr belowRange = nullptr;
  if (m_upperBoundToBlock.count(block->GetLeftEdge())) {
    belowRange = m_upperBoundToBlock[block->GetLeftEdge()];
    assert(belowRange != nullptr);
    block = merge(belowRange, block);
  }


  SackBlockPtr upperRange = nullptr;
  // merge[100, 300) [300, 400) = [100, 400), note block point to [100, 300)
  if (m_lowerBoundToBlock.count(block->GetRightEdge())) {
    upperRange = m_lowerBoundToBlock[block->GetRightEdge()];
    assert(upperRange != nullptr);
    merge(block, upperRange);
  }
  if (belowRange == nullptr && upperRange == nullptr) {
    m_lowerBoundToBlock[block->GetLeftEdge()] = block;
    m_upperBoundToBlock[block->GetRightEdge()] = block;
    // out of order segment
    m_blocks.insert(block);
  }
  // m_cacheInvalid = true;
}

uint32_t SackBlockSet::GetBlockNum() const {
  return m_blocks.size();
}

std::vector<SackBlockPtr> SackBlockSet::GetBlocks() {
  std::vector<SackBlockPtr> res;
  for (auto block : m_blocks) {
    res.push_back(block);
  }
  return res;
}

// for debug
std::string SackBlockSet::toString() {
  std::string res = "";
  int loop = 1;
  for (auto block: m_blocks) {
    res += "block" + std::to_string(loop) + ":";
    res += "[start:" + std::to_string(block->GetLeftEdge()) + ",";
    res += "end:" + std::to_string(block->GetRightEdge()) + ") "; 
    loop++;
  }
  return res;
}

//[200, 300) + [300, 400)-> [200, 400) 
SackBlockPtr SackBlockSet::merge(SackBlockPtr blockA, SackBlockPtr blockB) {
  m_lowerBoundToBlock.erase(blockA->GetLeftEdge());
  m_upperBoundToBlock.erase(blockA->GetRightEdge());

  m_lowerBoundToBlock.erase(blockB->GetLeftEdge());
  m_upperBoundToBlock.erase(blockB->GetRightEdge());

  //
  m_blocks.erase(blockA);
  m_blocks.erase(blockB);

  SackBlockPtr mergedBlock = std::make_shared<SackBlock>(blockA->GetLeftEdge(), blockB->GetRightEdge());
  m_lowerBoundToBlock[mergedBlock->GetLeftEdge()] = mergedBlock;
  m_upperBoundToBlock[mergedBlock->GetRightEdge()] = mergedBlock;
  m_blocks.insert(mergedBlock);
  return mergedBlock;
}

};