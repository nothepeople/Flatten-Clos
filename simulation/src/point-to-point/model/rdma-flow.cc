#include "rdma-flow.h"

namespace ns3 {

RdmaFlow::RdmaFlow(uint32_t _sip, uint32_t _dip, uint16_t _sport, uint16_t _dport, uint32_t _indev, uint32_t _outdev){
    sip = _sip;
    dip = _dip;
    sport = _sport;
    dport = _dport;
    indev = _indev;
    outdev = _outdev;
}
    
TypeId RdmaFlow::GetTypeId(void){
    static TypeId tid = TypeId("ns3::RdmaFlow")
        .SetParent<Object>();
    return tid;
}

uint32_t RdmaFlow::GetHash() {
    union 
    {
        struct {
            uint32_t sip, dip;
            uint16_t sport, dport;
        };
        char c[12]; 
    } buf;
    buf.sip = sip;
    buf.dip = dip;
    buf.sport = sport;
    buf.dport = dport;
    return Hash32(buf.c, 12);  
}

}