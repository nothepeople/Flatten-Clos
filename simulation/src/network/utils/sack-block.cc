#include "sack-block.h"
#include <string>

namespace ns3 {
SackBlock::SackBlock()
  : m_leftEdge(0),
    m_rightEdge(0) {}

SackBlock::SackBlock(uint32_t _left, uint32_t _right)
   : m_leftEdge(_left), 
     m_rightEdge(_right){}

std::string SackBlock::toString() {
  std::string res = "";
  res += "Seg[" + std::to_string(m_leftEdge) + "," + std::to_string(m_rightEdge) + ")";
  return res;
}

bool SackBlock::isWithin(uint32_t lowerBound, uint32_t upperBound) {
  return lowerBound >= m_leftEdge && upperBound <= m_rightEdge;
}

uint32_t SackBlock::GetLeftEdge() const {
  return m_leftEdge;
}

uint32_t SackBlock::GetRightEdge() const {
  return m_rightEdge;
}

void SackBlock::SetLeftEdge(uint32_t _left) {
  m_leftEdge = _left;
}

void SackBlock::SetRightEdge(uint32_t _right) {
  m_rightEdge = _right;
}

};