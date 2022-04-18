#ifndef RDMA_FLOW_H
#define RDMA_FLOW_H
#include <ns3/object.h>

namespace ns3 {

class RdmaFlow : public Object{
public:
    uint16_t sport, dport;
    uint32_t sip, dip;
    uint32_t indev, outdev;

    RdmaFlow(uint32_t _sip, uint32_t _dip, uint16_t _sport, uint16_t _dport, uint32_t _indev, uint32_t _outdev);
    static TypeId GetTypeId(void);
    uint32_t GetHash(void);

};

}
#endif