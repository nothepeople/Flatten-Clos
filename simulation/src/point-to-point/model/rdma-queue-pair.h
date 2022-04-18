#ifndef RDMA_QUEUE_PAIR_H
#define RDMA_QUEUE_PAIR_H

#include <ns3/object.h>
#include <ns3/packet.h>
#include <ns3/ipv4-address.h>
#include <ns3/data-rate.h>
#include <ns3/event-id.h>
#include <ns3/custom-header.h>
#include <ns3/int-header.h>
#include <vector>
#include <set>
#include <unordered_map>
#include <memory>
#include <assert.h>
#include "ns3/sack-block-set.h"

namespace ns3 {


class QbbNetDevice;
class RdmaQueuePair : public Object {
public:
	
	/**
	 * SACK releated
	 */
	std::set<uint32_t> m_sendUnackSegStartSeqs;

	// this set contains selectively acked segments. When it contains one segment(ack == snd_una), 
	// this set will be used to advance window.
	std::set<uint32_t> m_ackSegStartSeqs;
	std::unordered_map<uint32_t, uint32_t> m_seqMapIpId;
	
	// only count out of order packet
	bool m_enableSack;
	bool m_enableIRN;
	bool m_enablePfc;
	
	// timeout retransmission, when ENABLE IRN mode.
	EventId m_rtoEvent;
	uint32_t m_rtoLow;
	uint32_t m_rtoHigh;
	uint32_t m_N;
	// IRN
	uint32_t m_rtoInterval;
	uint32_t m_recoverySN;
	uint32_t m_retransmitSN;
	bool m_doRetransmit;
	bool m_inRecovery;
	bool m_findNewHole;
	uint64_t m_lastSetTimerTs;
	ns3::Ptr<Packet> txFree(uint32_t mtu);
	void handleAckOrNack();
	void timeout();

	// 
	uint32_t m_goMinimalPathThreshold;
	/********************
	 * SELECTIVE ACK
	 ********************/
	void selectiveAcknowledge(Ptr<Packet> p, CustomHeader &ch);
	void processAck(uint32_t ack);
	int findNextHole(uint32_t lastRetransSN);
	void handleAckOrNack(Ptr<Packet> p, CustomHeader &ch);

	Ptr<Packet> makePacket(uint32_t seq);
	void confirmSegment(uint32_t seq);
	uint32_t getSegmentSize(uint32_t seq) const;
	uint32_t getRto() const;
	uint32_t getInflightNPackets() const;
	

	Time startTime;
	Ipv4Address sip, dip;
	uint16_t sport, dport;
	uint64_t m_size;
	uint64_t snd_nxt, snd_una; // next seq to send, the highest unacked seq
	uint16_t m_pg;
	uint16_t m_ipid;
	uint32_t m_win; // bound of on-the-fly packets
	uint64_t m_baseRtt; // base RTT of this qp
	DataRate m_max_rate; // max rate
	bool m_var_win; // variable window size
	Time m_nextAvail;	//< Soonest time of next send
	uint32_t wp; // current window of packets
	uint32_t lastPktSize;
	Callback<void> m_notifyAppFinish;
	// for non-shortest queue pair
	uint32_t m_non_shortest;
	/******************************
	 * runtime states
	 *****************************/
	DataRate m_rate;	//< Current rate
	struct {
		DataRate m_targetRate;	//< Target rate
		EventId m_eventUpdateAlpha;
		double m_alpha;
		bool m_alpha_cnp_arrived; // indicate if CNP arrived in the last slot
		bool m_first_cnp; // indicate if the current CNP is the first CNP
		EventId m_eventDecreaseRate;
		bool m_decrease_cnp_arrived; // indicate if CNP arrived in the last slot
		uint32_t m_rpTimeStage;
		EventId m_rpTimer;
	} mlx;
	struct {
		uint32_t m_lastUpdateSeq;
		DataRate m_curRate;
		IntHop hop[IntHeader::maxHop];
		uint32_t keep[IntHeader::maxHop];
		uint32_t m_incStage;
		double m_lastGap;
		double u;
		struct {
			double u;
			DataRate Rc;
			uint32_t incStage;
		}hopState[IntHeader::maxHop];
	} hp;
	struct{
		uint32_t m_lastUpdateSeq;
		DataRate m_curRate;
		uint32_t m_incStage;
		uint64_t lastRtt;
		double rttDiff;
	} tmly;
	struct{
		uint32_t m_lastUpdateSeq;
		uint32_t m_caState;
		uint32_t m_highSeq; // when to exit cwr
		double m_alpha;
		uint32_t m_ecnCnt;
		uint32_t m_batchSizeOfAlpha;
	} dctcp;
	struct{
		uint32_t m_lastUpdateSeq;
		DataRate m_curRate;
		uint32_t m_incStage;
	}hpccPint;
	/************************************
	 * PCN related varaiables. 2021-2-20
	 ************************************/
	struct{
		double m_w;
	} pcn;

	/***********
	 * methods
	 **********/
	static TypeId GetTypeId (void);
	RdmaQueuePair(uint16_t pg, Ipv4Address _sip, Ipv4Address _dip, uint16_t _sport, uint16_t _dport);
	void SetSize(uint64_t size);
	void SetWin(uint32_t win);
	void SetBaseRtt(uint64_t baseRtt);
	void SetVarWin(bool v);
	void SetAppNotifyCallback(Callback<void> notifyAppFinish);
	// Added by qizhou.zqz.
	void SetNonShortest(uint32_t non);
	uint32_t GetNonShortest() const;

	uint64_t GetBytesLeft();
	uint32_t GetHash(void);
	void Acknowledge(uint64_t ack);
	uint64_t GetOnTheFly();
	bool IsWinBound();
	uint64_t GetWin(); // window size calculated from m_rate
	bool IsFinished();
	uint64_t HpGetCurWin(); // window size calculated from hp.m_curRate, used by HPCC
};

class RdmaHw;
class RdmaRxQueuePair : public Object { // Rx side queue pair
public:
	struct ECNAccount{
		uint16_t qIndex;
		uint8_t ecnbits;
		uint16_t qfb;
		uint16_t total;

		ECNAccount() { memset(this, 0, sizeof(ECNAccount));}
	};
	/*****************************
	 * The PCN NP state variables.
	 *****************************/
	struct PcnNpVar{
		uint16_t recNum;
		uint16_t ecnNum;
		uint32_t recData;
		uint64_t interArrivalTime;
		uint64_t pktLastArrivalTime;
		PcnNpVar() { memset(this, 0, sizeof(PcnNpVar)); }
	};
	PcnNpVar m_NpVar;
	ECNAccount m_ecn_source;
	EventId PcnTimerEvent;
	// first packet the rx qp received.
	bool m_firstPktRxQpRecv;
	void PcnNpStateChange(Ptr<RdmaHw> hw);

	/*****************************
	 * cache out of order blocks 
	 *****************************/
	SackBlockSet m_sackSet;

	uint32_t sip, dip;
	uint16_t sport, dport;
	uint16_t m_ipid;
	uint32_t ReceiverNextExpectedSeq;
	Time m_nackTimer;
	int32_t m_milestone_rx;
	uint32_t m_lastNACK;
	EventId QcnTimerEvent; // if destroy this rxQp, remember to cancel this timer

	static TypeId GetTypeId (void);
	RdmaRxQueuePair();
	uint32_t GetHash(void);
private:
	void CheckAndSendCNP(int ecn, uint32_t  recRate, Ptr<RdmaHw> hw);
};

class RdmaQueuePairGroup : public Object {
public:
	std::vector<Ptr<RdmaQueuePair> > m_qps;
	//std::vector<Ptr<RdmaRxQueuePair> > m_rxQps;

	static TypeId GetTypeId (void);
	RdmaQueuePairGroup(void);
	uint32_t GetN(void);
	Ptr<RdmaQueuePair> Get(uint32_t idx);
	Ptr<RdmaQueuePair> operator[](uint32_t idx);
	void AddQp(Ptr<RdmaQueuePair> qp);
	//void AddRxQp(Ptr<RdmaRxQueuePair> rxQp);
	void Clear(void);
};

}

#endif /* RDMA_QUEUE_PAIR_H */
