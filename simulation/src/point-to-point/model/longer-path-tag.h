#ifndef LONGER_PATH_TAG_H
#define LONGER_PATH_TAG_H
//added by qizhou.zqz

#include "ns3/tag.h"

namespace ns3 {

class LongerPathTag : public Tag
{
public:
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer buf) const;
  virtual void Deserialize (TagBuffer buf);
  virtual void Print (std::ostream &os) const;
  LongerPathTag ();
  LongerPathTag (uint32_t longer);
  void SetLonger (uint32_t longer);
  uint32_t GetLonger (void) const;
  
private:
  uint32_t m_longer;
};

} // namespace ns3

#endif
