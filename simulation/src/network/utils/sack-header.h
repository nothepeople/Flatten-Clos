#ifndef _SELECTIVE_SEG_HEADER_H
#define _SELECTIVE_SEG_HEADER_H

#include <stdint.h>
#include <vector>
#include "ns3/header.h"
#include "ns3/buffer.h"
#include "ns3/sack-block.h"
#include <string>

namespace ns3 {
/**
 * |Ether Header|IP Header|Qbb Header|SackHeader|
 */
class SackHeader : public Header {
public:
  static const uint32_t MAX_BLOCKS = 40;

  SackHeader();
  ~SackHeader();
  
  std::vector<SackBlock> GetSackBlocks();

  void SetBlockNum(uint32_t blockNum);

  void SetBlocks(std::vector<SackBlockPtr > cacheBlocks);
  
  static uint32_t GetStaticSize();
  virtual uint32_t GetSerializedSize(void) const;
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual void Serialize(Buffer::Iterator start) const;
  virtual uint32_t Deserialize(Buffer::Iterator start);
  std::string toString();

public:
  SackBlock m_cacheBlock[MAX_BLOCKS];
  uint32_t m_numBlock;
};


};
#endif