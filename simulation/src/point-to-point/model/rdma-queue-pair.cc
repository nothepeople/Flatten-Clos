#include <ns3/hash.h>
#include <ns3/uinteger.h>
#include <ns3/seq-ts-header.h>
#include <ns3/udp-header.h>
#include <ns3/ipv4-header.h>
#include <ns3/simulator.h>

#include "ns3/ppp-header.h"
#include "cn-header.h"
#include "rdma-hw.h"
#include "qbb-net-device.h"
#include "rdma-queue-pair.h"

#include <cmath>
#include <algorithm>
namespace ns3 {
/*********************************
 * SELECTIVE ACK CODE BLOCK START
 *********************************/
void RdmaQueuePair::timeout() {
	//RTtimeoutAC
	bool quickTimeout = false;
	bool extendTimerInterval = false;;
	uint64_t interval = Simulator::Now().GetTimeStep() - m_lastSetTimerTs;
	if (interval <= m_rtoLow) {
		quickTimeout = true;
	} 
	//RTO 
	if (quickTimeout && getInflightNPackets() > 3) {
		extendTimerInterval = true;
	} else {
		m_retransmitSN = snd_una;
		m_doRetransmit = true;
		m_findNewHole = false;
	}
	// printf("timeout rtoHigh=%d rtoLow=%d inflightPackets=%d\n", m_rtoLow, m_rtoHigh, getInflightNPackets());
}

int RdmaQueuePair::findNextHole(uint32_t lastRetransSN) {
	if (m_sendUnackSegStartSeqs.size() == 0) {
		return -1;
	}
	for (auto sndUna : m_sendUnackSegStartSeqs) {
		if (sndUna > lastRetransSN) {
			return sndUna;
		}
	}
	return -1;
}

Ptr<Packet> RdmaQueuePair::txFree(uint32_t mtu) {
	NS_ASSERT_MSG(m_win != 0 && m_enableSack, "m_win should not be zero and must enable sack");
	bool sendNew = false;
	uint32_t toSend = 0;
	if (m_doRetransmit && m_retransmitSN >= snd_una) {
		if (m_retransmitSN == snd_una) m_inRecovery = true; // timeout 
		toSend = m_retransmitSN;
		//loss recover
		m_recoverySN = snd_nxt - (m_size % mtu == 0 ? mtu : m_size % mtu);
		m_findNewHole = true;
		m_doRetransmit = false;
		sendNew = true;
	} else { // retranmitted packet has been acked
		if (m_doRetransmit) m_findNewHole = false;
		m_doRetransmit = false;
		// std::cout << "IsWinBound: " << IsWinBound() << "\n";
		if (!IsWinBound()) { // 
			// update send next
			sendNew = true;
			toSend = snd_nxt;
			// 
			m_sendUnackSegStartSeqs.insert(snd_nxt);
			m_seqMapIpId[snd_nxt] = m_ipid;
			uint32_t payload = getSegmentSize(snd_nxt);
			snd_nxt += payload;
			m_ipid += 1;
			//
		}
	}

		// only enable sack, we scheudle retransmission timer

	if (m_findNewHole) {
		// find next to retransmit
		int foundedHole = findNextHole(m_retransmitSN);
		if (foundedHole != -1) {
			m_retransmitSN = foundedHole;//
			m_doRetransmit = true;
		}
		m_findNewHole = false;
	}
	// std::cout << "sendNew: " << sendNew << "\n";
	if (sendNew) {
		// schedule timer
		if (m_rtoEvent.IsExpired() ) {
			m_rtoEvent = Simulator::Schedule(MicroSeconds( getRto() ), &RdmaQueuePair::timeout, this);
			m_lastSetTimerTs = Simulator::Now().GetTimeStep();
		}

		Ptr<Packet> p = makePacket(toSend);
		return p;
	}
	return nullptr;
}


void RdmaQueuePair::processAck(uint32_t ack) {
	// confirmSegment() will destroy the structure of set, 
	// so the iterator will lose its effect
	std::set<uint32_t> sendUackSegSeqs(m_sendUnackSegStartSeqs);
	for (auto seq : sendUackSegSeqs) {
		if (seq + getSegmentSize(seq) <= ack) {
			confirmSegment(seq);
		}
	} 

	// advance window
	while(m_ackSegStartSeqs.count(snd_una)) {
		m_ackSegStartSeqs.erase(snd_una);
		snd_una += getSegmentSize(snd_una);
	}
	snd_nxt = std::max(snd_nxt, snd_una);

}

/**
 * Selectively acknowledge out-of-order packets that arrive safely in sender side.
 */
void RdmaQueuePair::selectiveAcknowledge(Ptr<Packet> p, CustomHeader &ch) {
	uint32_t rcvExpectedSeq = ch.ack.seq;
	std::set<uint32_t> sendUnackStartSeqs(m_sendUnackSegStartSeqs);
	std::vector<SackBlock> blocks = ch.ack.sackHeader.GetSackBlocks();
	for (auto block : blocks)
		for (auto startSeq : sendUnackStartSeqs)
			if (block.isWithin(startSeq, startSeq + getSegmentSize(startSeq)))
				confirmSegment(startSeq);
}

void RdmaQueuePair::handleAckOrNack(Ptr<Packet> p, CustomHeader &ch) {
	uint32_t ack = ch.ack.seq;
	if (ack > snd_nxt) snd_nxt = ack;
	bool newAck = false;
	bool nack = false;


	if (ack > snd_una) {
		newAck = true;
		
		// advance window, advance snd_una
		processAck(ack);

		// schedule retransmission event
		Simulator::Cancel(m_rtoEvent);
		if (snd_una != snd_nxt) {
			m_rtoEvent = Simulator::Schedule(MicroSeconds( getRto() ), &RdmaQueuePair::timeout, this);
			m_lastSetTimerTs = Simulator::Now().GetTimeStep();
		}
	}

	// receive  NACK ?
	if (ch.l3Prot == 0xFD) {
		nack = true;
	}

	if (ack > m_recoverySN) {
		m_inRecovery = false;
		m_findNewHole = false;
		m_doRetransmit = false;
	}

	if (nack) { // do selective acknowledge
		//
		selectiveAcknowledge(p, ch);
		if (m_inRecovery) {
			if (m_sendUnackSegStartSeqs.size() != 0) {
				//SACK
				if (m_doRetransmit && m_sendUnackSegStartSeqs.count(m_retransmitSN)) {
					m_findNewHole = true;
					m_doRetransmit = false;
				} else if (!m_doRetransmit) {
					//
					if (m_sendUnackSegStartSeqs.size() > 0) {
						m_findNewHole = true;
					}
				}
			}
		}
	}

	if (newAck) {
		if (m_inRecovery) {
			//snd_un
			if (m_retransmitSN < snd_una || (m_retransmitSN == snd_una && m_doRetransmit)) {
				m_retransmitSN = snd_una;
				m_doRetransmit = true;
				m_findNewHole = false;
			}
		}
	} else {
		// normal stage or loss recovery stage
		if (!m_inRecovery) {
			if (snd_una < snd_nxt && nack) { // NACK arrived
				m_recoverySN = snd_una;
				m_doRetransmit = true;
				m_findNewHole = false;
			}
		}
	}

}


ns3::Ptr<Packet> RdmaQueuePair::makePacket(uint32_t seq) {
	uint32_t pktSize = getSegmentSize(seq);
	uint32_t ipId = m_seqMapIpId[seq];
	Ptr<Packet> p = Create<Packet> (pktSize);
	// add SeqTsHeader
	SeqTsHeader seqTs;
	seqTs.SetSeq (seq);
	seqTs.SetPG (m_pg);
	p->AddHeader (seqTs);
	// add udp header
	UdpHeader udpHeader;
	udpHeader.SetDestinationPort (dport);
	udpHeader.SetSourcePort (sport);
	p->AddHeader (udpHeader);
	// add ipv4 header
	Ipv4Header ipHeader;
	ipHeader.SetSource (sip);
	ipHeader.SetDestination (dip);
	ipHeader.SetProtocol (0x11);
	ipHeader.SetPayloadSize (p->GetSize());
	ipHeader.SetTtl (64);

	// We use the DSCP value to differentiate whether the packet should take the non_shortest path.
	ipHeader.SetTos (0);
	// Added by qizhou.zqz
	if (GetNonShortest()) {
		// std::cout << " gomiamal " << m_goMinimalPathThreshold << "\n";
		if (snd_nxt <= m_goMinimalPathThreshold)
			ipHeader.SetDscp ((Ipv4Header::DscpType)0x40);
		else 
			ipHeader.SetDscp ((Ipv4Header::DscpType)0x20);
	}
	ipHeader.SetIdentification (ipId);
	p->AddHeader(ipHeader);
	// add ppp header
	PppHeader ppp;
	ppp.SetProtocol (0x0021); // EtherToPpp(0x800), see point-to-point-net-device.cc
	p->AddHeader (ppp);

	return p;
}

uint32_t RdmaQueuePair::getSegmentSize(uint32_t seq) const {
	return std::min((uint64_t)1000, m_size - seq);
}

void RdmaQueuePair::confirmSegment(uint32_t seq) {
	m_sendUnackSegStartSeqs.erase(seq);
	m_seqMapIpId.erase(seq);
	m_ackSegStartSeqs.insert(seq);
}

uint32_t RdmaQueuePair::getInflightNPackets() const {
	return (snd_nxt - snd_una) / 1000;
}

uint32_t RdmaQueuePair::getRto() const {
	// Bandwidth Delay Product: RTT * Bandwidth
	return getInflightNPackets() <= m_N ? m_rtoLow : m_rtoHigh;
}
/*********************************
 * SELECTIVE ACK CODE BLOCK END
 *********************************/

/**************************
 * RdmaQueuePair
 *************************/
TypeId RdmaQueuePair::GetTypeId (void)
{
	static TypeId tid = TypeId ("ns3::RdmaQueuePair")
		.SetParent<Object> ()
		;
	return tid;
}

RdmaQueuePair::RdmaQueuePair(uint16_t pg, Ipv4Address _sip, Ipv4Address _dip, uint16_t _sport, uint16_t _dport){
	/*********
	 * 1G IRN
	 **********/
	m_rtoLow = 300;
	m_rtoHigh = 1200;
	m_N = 3;

	m_recoverySN = 0;
	m_retransmitSN = 0;
	m_inRecovery = false;
	m_doRetransmit = false;
	m_findNewHole = false;
	m_lastSetTimerTs = 0;

	// minimal path threshold
	m_goMinimalPathThreshold = 1000000;
	
	
	m_seqMapIpId.clear();
	m_sendUnackSegStartSeqs.clear();
	m_ackSegStartSeqs.clear();
	
	startTime = Simulator::Now();
	sip = _sip;
	dip = _dip;
	sport = _sport;
	dport = _dport;
	m_size = 0;
	snd_nxt = snd_una = 0;
	m_pg = pg;
	m_ipid = 0;
	m_win = 0;
	m_baseRtt = 0;
	m_max_rate = 0;
	m_var_win = false;
	m_rate = 0;
	m_non_shortest=0;
	m_nextAvail = Time(0);
	mlx.m_alpha = 1;
	mlx.m_alpha_cnp_arrived = false;
	mlx.m_first_cnp = true;
	mlx.m_decrease_cnp_arrived = false;
	mlx.m_rpTimeStage = 0;
	hp.m_lastUpdateSeq = 0;
	for (uint32_t i = 0; i < sizeof(hp.keep) / sizeof(hp.keep[0]); i++)
		hp.keep[i] = 0;
	hp.m_incStage = 0;
	hp.m_lastGap = 0;
	hp.u = 1;
	for (uint32_t i = 0; i < IntHeader::maxHop; i++){
		hp.hopState[i].u = 1;
		hp.hopState[i].incStage = 0;
	}

	tmly.m_lastUpdateSeq = 0;
	tmly.m_incStage = 0;
	tmly.lastRtt = 0;
	tmly.rttDiff = 0;

	dctcp.m_lastUpdateSeq = 0;
	dctcp.m_caState = 0;
	dctcp.m_highSeq = 0;
	dctcp.m_alpha = 1;
	dctcp.m_ecnCnt = 0;
	dctcp.m_batchSizeOfAlpha = 0;

	hpccPint.m_lastUpdateSeq = 0;
	hpccPint.m_incStage = 0;
}

void RdmaQueuePair::SetNonShortest(uint32_t non) {
	m_non_shortest = non;
}

uint32_t RdmaQueuePair::GetNonShortest() const {
	return m_non_shortest;
}

void RdmaQueuePair::SetSize(uint64_t size){
	m_size = size;
}

void RdmaQueuePair::SetWin(uint32_t win){
	m_win = win;
}

void RdmaQueuePair::SetBaseRtt(uint64_t baseRtt){
	m_rtoInterval = baseRtt * 1.5;
	m_baseRtt = baseRtt;
}

void RdmaQueuePair::SetVarWin(bool v){
	m_var_win = v;
}

void RdmaQueuePair::SetAppNotifyCallback(Callback<void> notifyAppFinish){
	m_notifyAppFinish = notifyAppFinish;
}

uint64_t RdmaQueuePair::GetBytesLeft(){
	return m_size >= snd_nxt ? m_size - snd_nxt : 0;
}

uint32_t RdmaQueuePair::GetHash(void){
	union{
		struct {
			uint32_t sip, dip;
			uint16_t sport, dport;
		};
		char c[12];
	} buf;
	buf.sip = sip.Get();
	buf.dip = dip.Get();
	buf.sport = sport;
	buf.dport = dport;
	return Hash32(buf.c, 12);
}

void RdmaQueuePair::Acknowledge(uint64_t ack){
	if (ack > snd_una){
		snd_una = ack;
	}
}

uint64_t RdmaQueuePair::GetOnTheFly(){
	return snd_nxt - snd_una;
}

bool RdmaQueuePair::IsWinBound(){
	uint64_t w = GetWin();
	return w != 0 && GetOnTheFly() >= w;	
}

uint64_t RdmaQueuePair::GetWin(){
	if (m_win == 0)
		return 0;
	uint64_t w;
	if (m_var_win){
		w = m_win * m_rate.GetBitRate() / m_max_rate.GetBitRate();
		if (w == 0)
			w = 1; // must > 0
	}else{
		w = m_win;
	}
	return w;
}

uint64_t RdmaQueuePair::HpGetCurWin(){
	if (m_win == 0)
		return 0;
	uint64_t w;
	if (m_var_win){
		w = m_win * hp.m_curRate.GetBitRate() / m_max_rate.GetBitRate();
		if (w == 0)
			w = 1; // must > 0
	}else{
		w = m_win;
	}
	return w;
}

bool RdmaQueuePair::IsFinished(){
	return snd_una >= m_size;
}

/*********************
 * RdmaRxQueuePair
 ********************/
TypeId RdmaRxQueuePair::GetTypeId (void)
{
	static TypeId tid = TypeId ("ns3::RdmaRxQueuePair")
		.SetParent<Object> ()
		;
	return tid;
}

RdmaRxQueuePair::RdmaRxQueuePair(){
	sip = dip = sport = dport = 0;
	m_ipid = 0;
	ReceiverNextExpectedSeq = 0;
	m_nackTimer = Time(0);
	m_milestone_rx = 0;
	m_lastNACK = 0;
}

uint16_t EtherToPpp (uint16_t proto){
	switch(proto){
		case 0x0800: return 0x0021;   //IPv4
		case 0x86DD: return 0x0057;   //IPv6
		default: NS_ASSERT_MSG (false, "PPP Protocol number not defined!");
	}
	return 0;
}

void RdmaRxQueuePair::CheckAndSendCNP(int ecn, uint32_t recRate, Ptr<RdmaHw> hw) {
	CnHeader cn(dport, 3, (uint8_t)ecn, m_ecn_source.qfb, m_ecn_source.total);
	// std::cout << "recRate in checkandsendcnp " << recRate << " in rxqp.cc \n";
	cn.SetRecRate(recRate);
	Ptr<Packet> p = Create<Packet>(0);
	p->AddHeader(cn);
	Ipv4Header iph;
	iph.SetDestination(Ipv4Address(dip));
	iph.SetSource(Ipv4Address(sip));
	iph.SetProtocol(0xFF);
	iph.SetTtl(64);
	iph.SetPayloadSize(p->GetSize());
	p->AddHeader(iph);
	uint32_t protocolNumber = 2048;
	// AddHeader(p, protocolNumber);	// Attach PPP header
	PppHeader ppp;
	ppp.SetProtocol (EtherToPpp (protocolNumber));
	p->AddHeader (ppp);
	// Trigger transmit	
	uint32_t nic_idx = hw->GetNicIdxOfRxQp(this);
	hw->m_nic[nic_idx].dev->RdmaEnqueueHighPrioQ(p);
	hw->m_nic[nic_idx].dev->TriggerTransmit();
}

void RdmaRxQueuePair::PcnNpStateChange(Ptr<RdmaHw> hw) {

	if (m_NpVar.recNum == 0) {
		m_NpVar.ecnNum = 0;
		m_NpVar.recData = 0;
	} else {
		double recRate = 0; // unit is bps
		if ( (double)m_NpVar.ecnNum / m_NpVar.recNum > 0.95) {
			// generate CNP packet to RP.
			if (m_NpVar.recNum == 1) {
				recRate = (double)(m_NpVar.recData * 8) / (m_NpVar.interArrivalTime) * 1e9 / 1e6;
			} else {
				recRate = (double)(m_NpVar.recData * 8) / 50 * 1e6 / 1e6;
			}
			// std::cout << "recEcn" << m_NpVar.ecnNum << " ecnNum " << m_NpVar.ecnNum 
			// <<  "recData " << m_NpVar.recData <<  "rate decrase: " << recRate << " interarrival: " 
			// << m_NpVar.interArrivalTime << "\n";
		    // send CNP with ECN=1
			CheckAndSendCNP(1, (uint32_t)floor(recRate), hw);
		} else {
			// std::cout << "ecnNum: " << m_NpVar.ecnNum << "\n";
			// std::cout << "ECN = 0\n";
			// send CNP with ECN=0
			CheckAndSendCNP(0, (uint32_t)floor(recRate), hw);
		}
	}
	
	// when the timer expire, the ecnNum, recNum and recData must be set to 0
	m_NpVar.recData = 0;
	m_NpVar.ecnNum = 0;
	m_NpVar.recNum = 0;
	
	PcnTimerEvent = Simulator::Schedule(MicroSeconds(50.0), &RdmaRxQueuePair::PcnNpStateChange, this, hw);
}

uint32_t RdmaRxQueuePair::GetHash(void){
	union{
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

/*********************
 * RdmaQueuePairGroup
 ********************/
TypeId RdmaQueuePairGroup::GetTypeId (void)
{
	static TypeId tid = TypeId ("ns3::RdmaQueuePairGroup")
		.SetParent<Object> ()
		;
	return tid;
}

RdmaQueuePairGroup::RdmaQueuePairGroup(void){
}

uint32_t RdmaQueuePairGroup::GetN(void){
	return m_qps.size();
}

Ptr<RdmaQueuePair> RdmaQueuePairGroup::Get(uint32_t idx){
	return m_qps[idx];
}

Ptr<RdmaQueuePair> RdmaQueuePairGroup::operator[](uint32_t idx){
	return m_qps[idx];
}

void RdmaQueuePairGroup::AddQp(Ptr<RdmaQueuePair> qp){
	m_qps.push_back(qp);
}

#if 0
void RdmaQueuePairGroup::AddRxQp(Ptr<RdmaRxQueuePair> rxQp){
	m_rxQps.push_back(rxQp);
}
#endif

void RdmaQueuePairGroup::Clear(void){
	m_qps.clear();
}

}
