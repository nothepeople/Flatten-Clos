#include "sack-header.h"
#include <assert.h>
#include <iostream>
#include "ns3/address-utils.h"
#include "ns3/log.h"
#include "ns3/buffer.h"

NS_LOG_COMPONENT_DEFINE("SackHeader");

namespace ns3 {
NS_OBJECT_ENSURE_REGISTERED (SackHeader);

SackHeader::SackHeader() : m_numBlock(0) {}

SackHeader::~SackHeader() {}

void SackHeader::SetBlockNum(uint32_t blockNum) {
  m_numBlock = blockNum;
}

std::vector<SackBlock> SackHeader::GetSackBlocks() {
  std::vector<SackBlock> res(std::begin(m_cacheBlock), std::begin(m_cacheBlock) + (m_numBlock <= MAX_BLOCKS ? m_numBlock : MAX_BLOCKS));
  return res;
}

void SackHeader::SetBlocks(std::vector<SackBlockPtr > blocks) {
  // assert(blocks.size() <= MAX_BLOCKS);
  for (uint32_t i = 0; 
      i < (blocks.size() <= MAX_BLOCKS ? blocks.size(): MAX_BLOCKS);
      i++) {
    m_cacheBlock[i] = *blocks[i];
  }
}

std::string SackHeader::toString() {
  std::string res="";
  for (uint32_t i = 0; i < MAX_BLOCKS; i++) {
    res += "block" + std::to_string(i + 1) + ":";
    res += "[start:" + std::to_string(m_cacheBlock[i].GetLeftEdge()) + ",";
    res += "end:" + std::to_string(m_cacheBlock[i].GetRightEdge()) + ") "; 
  }
  return res;
}

uint32_t SackHeader::GetSerializedSize() const {
  return sizeof(m_numBlock) + sizeof(decltype(m_cacheBlock[0])) * MAX_BLOCKS;
}

uint32_t SackHeader:: GetStaticSize() {
  return sizeof(m_numBlock) + sizeof(decltype(m_cacheBlock[0])) * MAX_BLOCKS;
}

TypeId SackHeader::GetTypeId() {
  static TypeId tid = TypeId("ns3::SackHeader")
    .SetParent<Header> ()
    .AddConstructor<SackHeader> ();
  return tid;
}

TypeId SackHeader::GetInstanceTypeId (void) const {
  return GetTypeId();
}

void SackHeader::Print (std::ostream &os) const {
  os << "Dump selective range\n";
}

void SackHeader::Serialize(Buffer::Iterator start) const{
  Buffer::Iterator iterator = start;
  for (uint32_t i  = 0; i < MAX_BLOCKS; i++) {
    iterator.WriteU32(m_cacheBlock[i].GetLeftEdge());
    iterator.WriteU32(m_cacheBlock[i].GetRightEdge());
  }
  iterator.WriteU32(m_numBlock);
}

uint32_t SackHeader:: Deserialize(Buffer::Iterator start){
  Buffer::Iterator iterator = start;
  for (uint32_t i = 0; i < MAX_BLOCKS; i++) {
    m_cacheBlock[i].SetLeftEdge(iterator.ReadU32());
    m_cacheBlock[i].SetRightEdge(iterator.ReadU32());
  }
  m_numBlock = iterator.ReadU32();
  return 36;
}

};