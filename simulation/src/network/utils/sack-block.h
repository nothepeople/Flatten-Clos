#ifndef _SACK_BLOCK_H
#define _SACK_BLOCK_H
#include <stdint.h>
#include <string>
#include <memory>
namespace ns3 {

class SackBlock {
public:
  SackBlock();
  SackBlock(uint32_t start, uint32_t end);
  
  std::string toString();

  void SetLeftEdge(uint32_t _left);
  void SetRightEdge(uint32_t _right);
  uint32_t GetLeftEdge() const;
  uint32_t GetRightEdge() const;
  
  bool isWithin(uint32_t lowerBound, uint32_t upperBound);

private:
  uint32_t m_leftEdge;
  uint32_t m_rightEdge; 
};

typedef std::shared_ptr<SackBlock> SackBlockPtr;

};

#endif