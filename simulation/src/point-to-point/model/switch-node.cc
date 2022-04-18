#include "ns3/ipv4.h"
#include "ns3/packet.h"
#include "ns3/ipv4-header.h"
#include "ns3/pause-header.h"
#include "ns3/flow-id-tag.h"
#include "ns3/boolean.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "switch-node.h"
#include "qbb-net-device.h"
#include "ppp-header.h"
#include "ns3/int-header.h"
#include <cmath>
#include <algorithm>

#include "longer-path-tag.h"

namespace ns3 {

TypeId SwitchNode::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SwitchNode")
    .SetParent<Node> ()
    .AddConstructor<SwitchNode> ()
	.AddAttribute("EcnEnabled",
			"Enable ECN marking.",
			BooleanValue(false),
			MakeBooleanAccessor(&SwitchNode::m_ecnEnabled),
			MakeBooleanChecker())
	.AddAttribute("CcMode",
			"CC mode.",
			UintegerValue(0),
			MakeUintegerAccessor(&SwitchNode::m_ccMode),
			MakeUintegerChecker<uint32_t>())
	.AddAttribute("AckHighPrio",
			"Set high priority for ACK/NACK or not",
			UintegerValue(0),
			MakeUintegerAccessor(&SwitchNode::m_ackHighPrio),
			MakeUintegerChecker<uint32_t>())
	.AddAttribute("MaxRtt",
			"Max Rtt of the network",
			UintegerValue(9000),
			MakeUintegerAccessor(&SwitchNode::m_maxRtt),
			MakeUintegerChecker<uint32_t>())
	.AddAttribute("EnablePfc",
			"Enable PFC or not.",
			BooleanValue(true),
			MakeBooleanAccessor(&SwitchNode::m_enablePfc),
			MakeBooleanChecker())
	.AddTraceSource("GetVictim", "get victim flows", 
			MakeTraceSourceAccessor(&SwitchNode::m_traceVictim))
	.AddTraceSource("GetRoot", "get root cause of pfc",
			MakeTraceSourceAccessor(&SwitchNode::m_traceRoot))
	.AddTraceSource("GetPause", "get pause info", 
			MakeTraceSourceAccessor(&SwitchNode::m_tracePause))
	.AddTraceSource("GetResume", "get resume info",
			MakeTraceSourceAccessor(&SwitchNode::m_traceResume));

	return tid;
}

SwitchNode::SwitchNode(){
	m_ecmpSeed = m_id;
	m_node_type = 1;
	m_mmu = CreateObject<SwitchMmu>();
	for (uint32_t i = 0; i < pCnt; i++)
		for (uint32_t j = 0; j < pCnt; j++)
			for (uint32_t k = 0; k < qCnt; k++)
				m_bytes[i][j][k] = 0;
	for (uint32_t i = 0; i < pCnt; i++)
		m_txBytes[i] = 0;
	for (uint32_t i = 0; i < pCnt; i++)
		m_lastPktSize[i] = m_lastPktTs[i] = 0;
	for (uint32_t i = 0; i < pCnt; i++)
		m_u[i] = 0;
	
	m_uv = CreateObject<UniformRandomVariable>();
	m_uv->SetAttribute ("Min", DoubleValue (0.0));
	m_uv->SetAttribute ("Max", DoubleValue (1.0));
	// ScheduleResetP2pBytes();
	// m_selectedRoute.clear();
	// m_flowletChecker = Simulator::Schedule(NanoSeconds(m_flowletGapNs), &SwitchNode::checkFlowlet, this);
}


void SwitchNode::ClearFlowRecord(Ptr<RdmaQueuePair> qp) {
	// map[Indev][Hash]
	// std::cout << "clear" << "\n";
	for (auto &it: m_passedFlows) {
		uint32_t indev = it.first;
		// std::cout << "indev: " << indev << " qp hash: " << qp->GetHash() << "\n";
		if (m_passedFlows[indev].find(qp->GetHash()) != m_passedFlows[indev].end()) 
		{
			// std::cout << "Id" << GetId() << " Do Clear in Switch.\n";
			m_passedFlows[indev].erase(qp->GetHash());
		} else {
			continue;
		}
	}

	union{
		struct {
			uint32_t sip, dip;
			uint16_t sport, dport;
		};
		char c[12];
	} buf;
	buf.sip = qp->dip.Get();
	buf.dip = qp->sip.Get();
	buf.sport = qp->dport;
	buf.dport = qp->sport;
	uint32_t ackFlowHash = Hash32(buf.c, 12);
	for (auto &it: m_passedFlows) {
		uint32_t indev = it.first;
		if (m_passedFlows[indev].find(ackFlowHash) != m_passedFlows[indev].end()) 
		{
			m_passedFlows[indev].erase(ackFlowHash);
		} else {
			continue;
		}
	}
	
}
/*********************
 * Set PCN's PNCounter
 *********************/
void SwitchNode::SwitchSetPNCounter(uint32_t ifIndex, uint32_t qIndex) {
	m_mmu->SetPNCounter(ifIndex, qIndex);
}

uint32_t SwitchNode::GetEcmpHashValue(Ptr<const Packet> p, CustomHeader &ch) {
	union {
		uint8_t u8[4+4+2+2];
		uint32_t u32[3];
	} buf;
	buf.u32[0] = ch.sip;
	buf.u32[1] = ch.dip;
	if (ch.l3Prot == 0x6)
		buf.u32[2] = ch.tcp.sport | ((uint32_t)ch.tcp.dport << 16);
	else if (ch.l3Prot == 0x11)
		buf.u32[2] = ch.udp.sport | ((uint32_t)ch.udp.dport << 16);
	else if (ch.l3Prot == 0xFC || ch.l3Prot == 0xFD)
		buf.u32[2] = ch.ack.sport | ((uint32_t)ch.ack.dport << 16);

	return EcmpHash(buf.u8, 12, m_ecmpSeed);
}

uint32_t SwitchNode::HostIpToPodNum(uint32_t ip) {
	return ( (ip >> 8) & 0xffff ) / m_hostsPerPod + 1;
}

/**************************
 * Threshold Routing
 **************************/
#define ADDR_TO_HOST(addr) ((addr >> 8) & (0xffff))
void SwitchNode::AddXpanderTableEntry(Ipv4Address &srcAddr, Ipv4Address &dstAddr, uint32_t ifidx, uint32_t remainHopToDst) {
	m_rtXpanderEntries[srcAddr.Get()][dstAddr.Get()][remainHopToDst].push_back(ifidx);
	m_remainHops[srcAddr.Get()][dstAddr.Get()].insert(remainHopToDst);
}

void SwitchNode::AddRRGTableEntry(Ipv4Address &srcAddr, Ipv4Address &dstAddr, uint32_t ifidx) {
	m_rtRRGEntries[srcAddr.Get()][dstAddr.Get()].push_back(ifidx);
}

void SwitchNode::AddRtWeight(Ipv4Address &srcAddr, Ipv4Address &dstAddr, double weight){	
	m_rtWeight[srcAddr.Get()][dstAddr.Get()].push_back(weight);
}

uint32_t SwitchNode::GetFlowHash(Ptr<const Packet> p, CustomHeader &ch) {
	union 
    {
        struct {
            uint16_t sport, dport;
            uint32_t sip, dip;
        };
        char c[12]; 
    } buf;
    buf.sip = ch.sip;
    buf.dip = ch.dip;
    buf.sport = ch.udp.sport;
    buf.dport = ch.udp.dport;
    return Hash32(buf.c, 12);  
}


void SwitchNode::SetSwitchDevRole(SwitchRole role) {
	m_devRole = role;
}

void SwitchNode::SetHostsPerPod(uint32_t hosts) {
	m_hostsPerPod = hosts;
}

void SwitchNode::ResetP2pBytes() {
	for (uint32_t i = 0; i < podNum; ++i) {
		for (uint32_t j = 0; j < podNum; ++j) {
			m_p2pbytes[i][j] = 0;
		}
	}
	m_lastResetP2p = Simulator::Now().GetTimeStep();
	// ScheduleResetP2pBytes();
}

void SwitchNode::ScheduleResetP2pBytes() {
	Simulator::Schedule(MilliSeconds(100), &SwitchNode::ResetP2pBytes, this);
}

void SwitchNode::HelpDumpPortBytes(uint32_t port) {
	uint32_t sum_bytes = 0;
	for (uint32_t i = 0; i < GetNDevices(); i++) {
		for (uint32_t j = 0; j < qCnt; j++) {
			sum_bytes += m_bytes[port][i][j];
		}
	}
	std::cout << Simulator::Now().GetTimeStep() << " ID " << GetId() << " port " << port << " bytes " << sum_bytes << "\n";
}

void SwitchNode::CheckAndSendPfc(uint32_t inDev, uint32_t qIndex){
	Ptr<QbbNetDevice> device = DynamicCast<QbbNetDevice>(m_devices[inDev]);
	if (m_enablePfc && m_mmu->CheckShouldPause(inDev, qIndex)){
		device->SendPfc(qIndex, 0);
		m_mmu->SetPause(inDev, qIndex);
		
		m_tracePause(GetId(), inDev, qIndex);
		TraceAffectedFlow(inDev, qIndex);
		
	}
}

void SwitchNode::CheckAndSendResume(uint32_t inDev, uint32_t qIndex){
	Ptr<QbbNetDevice> device = DynamicCast<QbbNetDevice>(m_devices[inDev]);
	if (m_enablePfc && m_mmu->CheckShouldResume(inDev, qIndex)){
		device->SendPfc(qIndex, 1);
		
		m_mmu->SetResume(inDev, qIndex);
		m_traceResume(GetId(), inDev, qIndex);
	}
}

void SwitchNode::AddNonShortestTableEntry(Ipv4Address &srcAddr, Ipv4Address &dstAddr, uint32_t intf_idx) {
	m_rtNonShortestTable[srcAddr.Get()][dstAddr.Get()].push_back(intf_idx);
}

void SwitchNode::DumpPortTxBytes(FILE *file) {
    fprintf(file, "%u ", GetId());
    for (uint32_t i = 1; i < GetNDevices(); ++i) {
        fprintf(file, "%u:%u ", i, m_txBytes[i]);
    }
    fprintf(file, "\n");
    fflush(file);
}

void SwitchNode::TraceAffectedFlow(uint32_t inDev, uint32_t qIndex) {
	auto& reFlows = m_passedFlows[inDev];
	for (auto &pair: reFlows) {
		Ptr<RdmaFlow> flow = pair.second;
		uint32_t outDev = flow->outdev;
		bool egressCongested = m_mmu->ShouldSendCN(outDev, qIndex);
		Ptr<QbbNetDevice> device = DynamicCast<QbbNetDevice>(m_devices[outDev]);
		if (!device->QueuePaused(qIndex)) {
			if(egressCongested) {
				m_traceRoot(flow, GetId());
			} else {
				m_traceVictim(flow, GetId());
			}
		}
	}
}

void SwitchNode::RecordFlow(uint32_t inDev, uint32_t outDev, CustomHeader &ch){
	// uint64_t nowT = Simulator::Now().GetTimeStep();
	// if(nowT - m_lastClearTime[inDev] > clearPeriod) {
	// 	m_passedFlows[inDev].clear();
	// 	m_lastClearTime[inDev] = nowT;
	// }
	Ptr<RdmaFlow> rdmaFlow = CreateObject<RdmaFlow>(ch.sip, ch.dip, ch.udp.sport, ch.udp.dport, inDev, outDev);
	uint32_t hash = rdmaFlow->GetHash();
	// std::cout << "Flow hash = " << hash << "\n";
	auto &flows = m_passedFlows[inDev];
	// Add new flow
	if (flows.find(hash) == flows.end())
		m_passedFlows[inDev][hash] = rdmaFlow;
}


/*************************
 * Get Outdev method
 ************************/
int SwitchNode::determineOutdevXpander(Ptr<Packet> p, CustomHeader& ch) {
	int selectedOutdev = -1;
	const auto &hopSet = m_remainHops[ch.sip][ch.dip];
	int minHop = *(hopSet.cbegin());
	LongerPathTag tag;
	p->PeekPacketTag(tag);
	bool allowLonger = tag.GetLonger();
	uint32_t flowHash = GetEcmpHashValue(p, ch);
	uint32_t alpha = 1;
	uint32_t pathChangeThreshold = 25 * 1024;
	if (!allowLonger) { // we only allow flow take shortest path
		const auto &minNextHops = m_rtXpanderEntries[ch.sip][ch.dip][minHop];
		NS_ASSERT_MSG(minNextHops.size() > 0, "error, nexthops should not be empty");
		int minBytes = 0xfffffff;
		for (const auto &nhop : minNextHops) {
			if (minBytes > m_mmu->GetEgressBytes(nhop)) {
				minBytes =  m_mmu->GetEgressBytes(nhop);
				selectedOutdev = nhop;
			}
		}
		
	} else { // in current switch we allow packet selecting both shortes path and longer path
		uint32_t minBytes = 0xffffff;

		for (auto remainHop : hopSet) {
			const auto &outdevs = m_rtXpanderEntries[ch.sip][ch.dip][remainHop];
			for (auto outdev : outdevs) {
				if ( minBytes >= m_mmu->GetEgressBytes(outdev) && m_mmu->GetEgressBytes(outdev) <= pathChangeThreshold) {
					selectedOutdev = outdev;
					minBytes = m_mmu->GetEgressBytes(outdev);
				}
			}
			if (selectedOutdev != -1) {
				if (remainHop > minHop) {
					LongerPathTag t;
					p->RemovePacketTag(t);
					p->AddPacketTag(LongerPathTag(0));
				}
				break;
			}
		}

		if (selectedOutdev == -1) {
			const auto &shortestOutdevs = m_rtXpanderEntries[ch.sip][ch.dip][minHop];
			minBytes = 0xffffff;
			for (auto outdev : shortestOutdevs) {
				if (m_mmu->GetEgressBytes(outdev) < minBytes) {
					minBytes = m_mmu->GetEgressBytes(outdev);
					selectedOutdev = outdev;
				}
			}
		}

	}
	if (selectedOutdev == -1) {
		return GetOutDev(p, ch);
	}
	NS_ASSERT_MSG(selectedOutdev != -1, "selectedOutdev cannot be -1");
	return selectedOutdev;
}


int SwitchNode::GetXpanderOutdev(Ptr<Packet> p, CustomHeader &ch) {
	FlowInfo flowInfo(ch.sip, ch.dip, ch.udp.sport, ch.udp.dport);
	// uint32_t multiToOneBytes = 15 * 1024;
	if (!m_selectedRoute.find(flowInfo)) {//first packet of one flow
		int selectedOutdev = determineOutdevXpander(p, ch);
		m_selectedRoute.insert(flowInfo, selectedOutdev);
		return selectedOutdev;
	} else { // not first packet of one flow
		//
		return m_selectedRoute[flowInfo];
	}
	return -1;
}



int SwitchNode::GetXpanderOutdevFlowlet(Ptr<Packet> p, CustomHeader &ch) {
	
	uint32_t flowHash = GetFlowHash(p, ch);
	if (!m_flowletRecord.count(flowHash)) {// first packet of flow(not flowlet)
		int selectedOutdev = determineOutdevXpander(p, ch);
		// save as flowlet
		uint64_t current = Simulator::Now().GetTimeStep();
		m_flowletRecord.insert({flowHash, {selectedOutdev, current}});
		return selectedOutdev;
	} else { // include this flow hash, there may be two situations 
		auto &flowlet = m_flowletRecord[flowHash];
		uint64_t current = Simulator::Now().GetTimeStep();
		if (current - flowlet.lastTs <= m_flowletGapNs) {
			return flowlet.output_port;
		} else {
			// std::cout << "sid: " << GetId() << "flowlet timeout\n";
			flowlet.lastTs = current;
			flowlet.output_port = determineOutdevXpander(p, ch);
			flowlet.flowlet_id++;
			return flowlet.output_port;
		}
	}
	return -1;
}


int SwitchNode::GetUniformHashOutdev(Ptr<Packet> p, CustomHeader &ch) {
	auto &hopSet = m_remainHops[ch.sip][ch.dip];
	if (hopSet.size() == 0) {
		// std::cout << "call this\n";
		return GetOutDev(p, ch);
	}
	uint32_t hash = GetEcmpHashValue(p, ch);
	auto it = hopSet.begin();
	for (int i = 0; i < hash % hopSet.size(); i++, it++);
	
	auto &entries = m_rtXpanderEntries[ch.sip][ch.dip][(*it)];
	NS_ASSERT_MSG(entries.size() > 0, "should not be empty entry");
	return entries[hash % entries.size()];
}

int SwitchNode::GetWeightedHashOutdev(Ptr<Packet> p, CustomHeader &ch) {

	auto &hopSet = m_remainHops[ch.sip][ch.dip];
	uint32_t hash = GetEcmpHashValue(p, ch);
	if (hopSet.size() == 0) {
		return GetOutDev(p, ch);
	} else if (hopSet.size() == 1) {
		auto &entries = m_rtXpanderEntries[ch.sip][ch.dip][*hopSet.begin()];
		NS_ASSERT_MSG(entries.size() > 0, "should not be empty entry");
		return entries[hash % entries.size()];
	} else { // p 50% shortest path
		int hopSize = hopSet.size();
		int choosedHop = -1;
		int hv = hash % ( (hopSize - 1) * 2 );
		if (hv < hopSize - 1) {
			choosedHop = *hopSet.begin();
		} else {
			auto it = hopSet.begin();
			for (int i = 0; i < hv - hopSize - 2; i++, it++);
			choosedHop = *it;
		}
		auto &entries = m_rtXpanderEntries[ch.sip][ch.dip][choosedHop];
		NS_ASSERT_MSG(entries.size() > 0, "should not be empty entry");
		return entries[hash % entries.size()];
	}
}

int SwitchNode::GetCongestionAwareOutdev(Ptr<Packet> p, CustomHeader &ch) {
	return GetXpanderOutdevFlowlet(p, ch);
}

int SwitchNode::GetOutDev(Ptr<const Packet> p, CustomHeader &ch){
	// look up entries
	auto entry = m_rtTable.find(ch.dip);
	// no matching entry
	if (entry == m_rtTable.end())
		return -1;
	// entry found
	auto &nexthops = entry->second;
	uint32_t idx = GetEcmpHashValue(p, ch) % nexthops.size();
	// printf("sid=%d nhops: %d\n", GetId(), nexthops.size());
	return nexthops[idx];
}

int SwitchNode::GetLpOutDev(Ptr<Packet> p, CustomHeader &ch) {
	FlowInfo flowInfo(ch.sip, ch.dip, ch.udp.sport, ch.udp.dport);
	if (!m_selectedRoute.find(flowInfo)) {//first packet of one flow
		int selectedOutdev = -1;
		// 
		double rv = m_uv->GetValue();
		const auto &weights = m_rtWeight[ch.sip][ch.dip];
		const auto &nexthops = m_rtRRGEntries[ch.sip][ch.dip];
		NS_ASSERT_MSG(weights.size() == nexthops.size(), "weight size must equal to nexthops size");

		int i;
		double sum_weight = 0;
		for (i  = 0; i < weights.size(); i++) {
			if (weights[i] == 0) continue;
			if (rv < sum_weight + weights[i]) {
				selectedOutdev = nexthops[i];
				break;
			}
			sum_weight += weights[i];
		}

		if (selectedOutdev == -1) {
			selectedOutdev = GetOutDev(p, ch);
		}
		// NS_ASSERT_MSG(selectedOutdev != -1, "outdev cannot be -1 in GetLpOutdev");
		m_selectedRoute.insert(flowInfo, selectedOutdev);
		return selectedOutdev;
	} else { // not first packet of one flow
		//
		return m_selectedRoute[flowInfo];
	}
}



void SwitchNode::SendToDev(Ptr<Packet>p, CustomHeader &ch){
	uint32_t src_pod = 0, dst_pod = 0;
	if (m_hostsPerPod != 0) {	
		src_pod = ((ch.sip >> 8) & 0xffff) / m_hostsPerPod + 1;
		dst_pod = ((ch.dip >> 8) & 0xffff) / m_hostsPerPod + 1;
	}
	int idx;
	int dscp = ch.GetIpv4DscpValue();

	if (dscp == 32 && enable_ksp ) { // expander topology
		uint32_t snode = ADDR_TO_HOST(ch.sip);
		uint32_t dnode = ADDR_TO_HOST(ch.dip);
		LongerPathTag tag;
		p->PeekPacketTag(tag);
		uint32_t longer = tag.GetLonger();
		if (snode / m_hostsPerSw == dnode / m_hostsPerSw) {
			idx = GetOutDev(p, ch);
		} else {
			idx = GetXpanderOutdev(p, ch);
		}
		NS_ASSERT_MSG(idx != -1, "idx should not be equal to -1 in GetXpanderOutdev()");
	} else if (enable_congestion_aware) {
		// congestion aware
		idx = GetCongestionAwareOutdev(p, ch);
	} else if (enable_uniform_hash) {
		// uniform hash
		idx = GetUniformHashOutdev(p, ch);
	} else if (enable_weighted_hash) {
		// weighted hash
		idx = GetWeightedHashOutdev(p, ch);
	} else if (enable_lp){
		// LP hash
		idx = GetLpOutDev(p, ch);
	} else {
		// ECMP 
		// std::cout << "ECMP\n";
		idx = GetOutDev(p, ch);
	}

	if (idx >= 0){
		NS_ASSERT_MSG(m_devices[idx]->IsLinkUp(), "The routing table look up should return link that is up");

		// determine the qIndex
		uint32_t qIndex;
		if (ch.l3Prot == 0xFF || ch.l3Prot == 0xFE || (m_ackHighPrio && (ch.l3Prot == 0xFD || ch.l3Prot == 0xFC))){  //QCN or PFC or NACK, go highest priority
			qIndex = 0;
		}else{
			qIndex = (ch.l3Prot == 0x06 ? 1 : ch.udp.pg); // if TCP, put to queue 1
		}

		// admission control
		FlowIdTag t;
		p->PeekPacketTag(t);
		uint32_t inDev = t.GetFlowId();
		if (qIndex != 0){ //not highest priority
			if (m_mmu->CheckIngressAdmission(inDev, qIndex, p->GetSize()) && m_mmu->CheckEgressAdmission(idx, qIndex, p->GetSize())){			// Admission control
				m_mmu->UpdateIngressAdmission(inDev, qIndex, p->GetSize());
				m_mmu->UpdateEgressAdmission(idx, qIndex, p->GetSize());
			}else{
				if (ch.l3Prot == 0xFD || ch.l3Prot == 0xFC) {
					std::cout << "Drop ACK" << std::endl;
				}
				return; // Drop
			}
			// record flows
			RecordFlow(inDev, idx, ch);
			CheckAndSendPfc(inDev, qIndex);
		}

		/*************************
		 * Threshold Routing.
		 * 1119 @ 2021-01-22
		 *************************/
		// Pod to pod bytes record.
		m_p2pbytes[src_pod][dst_pod] += p->GetSize();
		// Clear the p2p byte records every second. 
		// if (Simulator::Now().GetTimeStep() - m_lastResetP2p > 6000) {
		// 	ResetP2pBytes();
		// }

		m_bytes[inDev][idx][qIndex] += p->GetSize();
		m_devices[idx]->SwitchSend(qIndex, p, ch);
	}else
		return; // Drop
}

uint32_t SwitchNode::EcmpHash(const uint8_t* key, size_t len, uint32_t seed) {
  uint32_t h = seed;
  if (len > 3) {
    const uint32_t* key_x4 = (const uint32_t*) key;
    size_t i = len >> 2;
    do {
      uint32_t k = *key_x4++;
      k *= 0xcc9e2d51;
      k = (k << 15) | (k >> 17);
      k *= 0x1b873593;
      h ^= k;
      h = (h << 13) | (h >> 19);
      h += (h << 2) + 0xe6546b64;
    } while (--i);
    key = (const uint8_t*) key_x4;
  }
  if (len & 3) {
    size_t i = len & 3;
    uint32_t k = 0;
    key = &key[i - 1];
    do {
      k <<= 8;
      k |= *key--;
    } while (--i);
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    h ^= k;
  }
  h ^= len;
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

void SwitchNode::SetEcmpSeed(uint32_t seed){
	m_ecmpSeed = seed;
}

void SwitchNode::AddTableEntry(Ipv4Address &dstAddr, uint32_t intf_idx){
	uint32_t dip = dstAddr.Get();
	m_rtTable[dip].push_back(intf_idx);
}

void SwitchNode::ClearTable(){
	m_rtTable.clear();
}

// This function can only be called in switch mode
bool SwitchNode::SwitchReceiveFromDevice(Ptr<NetDevice> device, Ptr<Packet> packet, CustomHeader &ch){
	SendToDev(packet, ch);
	return true;
}

void SwitchNode::SwitchNotifyDequeue(uint32_t ifIndex, uint32_t qIndex, Ptr<Packet> p){
	FlowIdTag t;
	p->PeekPacketTag(t);
	if (qIndex != 0){
		uint32_t inDev = t.GetFlowId();
		m_mmu->RemoveFromIngressAdmission(inDev, qIndex, p->GetSize());
		m_mmu->RemoveFromEgressAdmission(ifIndex, qIndex, p->GetSize());
		m_bytes[inDev][ifIndex][qIndex] -= p->GetSize();
		
		/** Test For Threshold Routing **/

		if (m_ccMode != 22) { // CC is not PCN, may be DCQCN, HPCC, TIMELY
			if (m_ecnEnabled){
				bool egressCongested = m_mmu->ShouldSendCN(ifIndex, qIndex);
				if (egressCongested){
					PppHeader ppp;
					Ipv4Header h;
					p->RemoveHeader(ppp);
					p->RemoveHeader(h);
					h.SetEcn((Ipv4Header::EcnType)0x03);
					p->AddHeader(h);
					p->AddHeader(ppp);
				}
			}
			//CheckAndSendPfc(inDev, qIndex);
			CheckAndSendResume(inDev, qIndex);
		} else { // CC is PCN.
			/**********
			 * PCN
			 **********/
			bool egressCongested = m_mmu->PcnSendCN(ifIndex, qIndex);
			if (egressCongested){
				PppHeader ppp;
				Ipv4Header h;
				p->RemoveHeader(ppp);
				p->RemoveHeader(h);
				h.SetEcn((Ipv4Header::EcnType)0x03);
				p->AddHeader(h);
				p->AddHeader(ppp);
			}
			CheckAndSendResume(inDev, qIndex);
		}
	}
	if (1){
		uint8_t* buf = p->GetBuffer();
		if (buf[PppHeader::GetStaticSize() + 9] == 0x11){ // udp packet
			IntHeader *ih = (IntHeader*)&buf[PppHeader::GetStaticSize() + 20 + 8 + 6]; // ppp, ip, udp, SeqTs, INT
			Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(m_devices[ifIndex]);
			if (m_ccMode == 3){ // HPCC
				ih->PushHop(Simulator::Now().GetTimeStep(), m_txBytes[ifIndex], dev->GetQueue()->GetNBytesTotal(), dev->GetDataRate().GetBitRate());
			}else if (m_ccMode == 10){ // HPCC-PINT
				uint64_t t = Simulator::Now().GetTimeStep();
				uint64_t dt = t - m_lastPktTs[ifIndex];
				if (dt > m_maxRtt)
					dt = m_maxRtt;
				uint64_t B = dev->GetDataRate().GetBitRate() / 8; //Bps
				uint64_t qlen = dev->GetQueue()->GetNBytesTotal();
				double newU;

				/**************************
				 * approximate calc
				 *************************/
				int b = 20, m = 16, l = 20; // see log2apprx's paremeters
				int sft = logres_shift(b,l);
				double fct = 1<<sft; // (multiplication factor corresponding to sft)
				double log_T = log2(m_maxRtt)*fct; // log2(T)*fct
				double log_B = log2(B)*fct; // log2(B)*fct
				double log_1e9 = log2(1e9)*fct; // log2(1e9)*fct
				double qterm = 0;
				double byteTerm = 0;
				double uTerm = 0;
				if ((qlen >> 8) > 0){
					int log_dt = log2apprx(dt, b, m, l); // ~log2(dt)*fct
					int log_qlen = log2apprx(qlen >> 8, b, m, l); // ~log2(qlen / 256)*fct
					qterm = pow(2, (
								log_dt + log_qlen + log_1e9 - log_B - 2*log_T
								)/fct
							) * 256;
					// 2^((log2(dt)*fct+log2(qlen/256)*fct+log2(1e9)*fct-log2(B)*fct-2*log2(T)*fct)/fct)*256 ~= dt*qlen*1e9/(B*T^2)
				}
				if (m_lastPktSize[ifIndex] > 0){
					int byte = m_lastPktSize[ifIndex];
					int log_byte = log2apprx(byte, b, m, l);
					byteTerm = pow(2, (
								log_byte + log_1e9 - log_B - log_T
								)/fct
							);
					// 2^((log2(byte)*fct+log2(1e9)*fct-log2(B)*fct-log2(T)*fct)/fct) ~= byte*1e9 / (B*T)
				}
				if (m_maxRtt > dt && m_u[ifIndex] > 0){
					int log_T_dt = log2apprx(m_maxRtt - dt, b, m, l); // ~log2(T-dt)*fct
					int log_u = log2apprx(int(round(m_u[ifIndex] * 8192)), b, m, l); // ~log2(u*512)*fct
					uTerm = pow(2, (
								log_T_dt + log_u - log_T
								)/fct
							) / 8192;
					// 2^((log2(T-dt)*fct+log2(u*512)*fct-log2(T)*fct)/fct)/512 = (T-dt)*u/T
				}
				newU = qterm+byteTerm+uTerm;

				#if 0
				/**************************
				 * accurate calc
				 *************************/
				double weight_ewma = double(dt) / m_maxRtt;
				double u;
				if (m_lastPktSize[ifIndex] == 0)
					u = 0;
				else{
					double txRate = m_lastPktSize[ifIndex] / double(dt); // B/ns
					u = (qlen / m_maxRtt + txRate) * 1e9 / B;
				}
				newU = m_u[ifIndex] * (1 - weight_ewma) + u * weight_ewma;
				printf(" %lf\n", newU);
				#endif

				/************************
				 * update PINT header
				 ***********************/
				uint16_t power = Pint::encode_u(newU);
				if (power > ih->GetPower())
					ih->SetPower(power);

				m_u[ifIndex] = newU;
			}
		}
	}
	m_txBytes[ifIndex] += p->GetSize();
	m_lastPktSize[ifIndex] = p->GetSize();
	m_lastPktTs[ifIndex] = Simulator::Now().GetTimeStep();
}

int SwitchNode::logres_shift(int b, int l){
	static int data[] = {0,0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5};
	return l - data[b];
}

int SwitchNode::log2apprx(int x, int b, int m, int l){
	int x0 = x;
	int msb = int(log2(x)) + 1;
	if (msb > m){
		x = (x >> (msb - m) << (msb - m));
		#if 0
		x += + (1 << (msb - m - 1));
		#else
		int mask = (1 << (msb-m)) - 1;
		if ((x0 & mask) > (rand() & mask))
			x += 1<<(msb-m);
		#endif
	}
	return int(log2(x) * (1<<logres_shift(b, l)));
}

} /* namespace ns3 */
