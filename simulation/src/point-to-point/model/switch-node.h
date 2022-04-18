#ifndef SWITCH_NODE_H
#define SWITCH_NODE_H

#include <unordered_map>
#include <ns3/node.h>
#include <memory>
#include "qbb-net-device.h"
#include "switch-mmu.h"
#include "pint.h"
#include "rdma-flow.h"
#include "ns3/traced-callback.h"
#include "ns3/callback.h"
#include "ns3/random-variable-stream.h"


namespace ns3 {

class Packet;

class FlowInfo {
public:
	uint32_t sip;
	uint32_t dip;
	uint16_t sport;
	uint16_t dport;
	FlowInfo() : sip(0), dip(0), sport(0), dport(0) {}
	FlowInfo(uint32_t _sip, uint32_t _dip, uint16_t _sport, uint16_t _dport) :
		sip(_sip), dip(_dip), sport(_sport), dport(_dport){}
	bool isEqual(const FlowInfo& other) {
		return sip == other.sip &&
					dip == other.dip &&
					sport == other.sport &&
					dport == other.dport;
	}
};

class MyMap {
public:
	MyMap() {
		m_size = 0;
	}
	bool find(const FlowInfo info) {
		// std::cout << "before loop " << "size: " << m_size << "\n";
		for (uint32_t i = 0; i < m_size; i++) {
			// 
			// std::cout << "enter loop " << "\n";
			if (m_flowInfos[i].isEqual(info)) {
				return true;
			}
		}
		return false;
	}

	void insert(const FlowInfo &info, uint32_t outdev) {
		// m_flowInfos.push_back(info);
		// m_outdevs.push_back(outdev);
		m_flowInfos[m_size] = info;
		m_outdevs[m_size] = outdev;
		m_size++;
	}

	uint32_t getSize() {
		return m_size;
	}

	uint32_t& operator[](const FlowInfo &info) {
		uint32_t idx = flowInfoMapIdx(info);
		NS_ASSERT_MSG(idx != 0xffffffff, "idx should not be 0xffffffff in MyMap");
		return m_outdevs[idx];
	}
private:
	uint32_t flowInfoMapIdx(const FlowInfo &info) {
		uint32_t idx = 0xffffffff;
		for (uint32_t i = 0; i < m_size; i++) {
			if (m_flowInfos[i].isEqual(info)) {
				return i;
			}
		}
		return idx;
	}

	FlowInfo m_flowInfos[8000];
	uint32_t 	m_outdevs[8000];
	// std::vector<FlowInfo> m_flowInfos;
	// std::vector<uint32_t> m_outdevs;
	uint32_t m_size;
};

class SwitchNode : public Node{
	static const uint32_t pCnt = 257;	// Number of ports used
	static const uint32_t qCnt = 8;	// Number of queues/priorities used
	uint32_t m_ecmpSeed;
	std::unordered_map<uint32_t, std::vector<int> > m_rtTable; // map from ip address (u32) to possible ECMP port (index of dev)

	// monitor of PFC
	uint32_t m_bytes[pCnt][pCnt][qCnt]; // m_bytes[inDev][outDev][qidx] is the bytes from inDev enqueued for outDev at qidx
	
	uint64_t m_txBytes[pCnt]; // counter of tx bytes

	uint32_t m_lastPktSize[pCnt];
	uint64_t m_lastPktTs[pCnt]; // ns
	double m_u[pCnt];

protected:
	bool m_ecnEnabled;
	uint32_t m_ccMode;
	uint64_t m_maxRtt;
	bool m_enablePfc;

	uint32_t m_ackHighPrio; // set high priority for ACK/NACK

private:
	int GetOutDev(Ptr<const Packet>, CustomHeader &ch);
	void SendToDev(Ptr<Packet>p, CustomHeader &ch);
	static uint32_t EcmpHash(const uint8_t* key, size_t len, uint32_t seed);
	void CheckAndSendPfc(uint32_t inDev, uint32_t qIndex);
	void CheckAndSendResume(uint32_t inDev, uint32_t qIndex);
	// Non-direct routing.
	uint32_t GetEcmpHashValue(Ptr<const Packet> p, CustomHeader &ch);

	int GetXpanderOutdev(Ptr<Packet> p, CustomHeader &ch);
	int GetXpanderOutdevFlowlet(Ptr<Packet> p, CustomHeader &ch);

	int GetUniformHashOutdev(Ptr<Packet> p, CustomHeader &ch);
	int GetWeightedHashOutdev(Ptr<Packet> p, CustomHeader &ch);
	int GetCongestionAwareOutdev(Ptr<Packet> p, CustomHeader &ch);
	int GetLpOutDev(Ptr<Packet> p, CustomHeader &ch);

	int determineOutdevXpander(Ptr<Packet> p, CustomHeader& ch);

	void ResetP2pBytes();
	uint32_t GetFlowHash(Ptr<const Packet> p, CustomHeader &ch);
	void ScheduleResetP2pBytes();

	inline uint32_t HostIpToPodNum (uint32_t ip);
	void HelpDumpPortBytes(uint32_t port);
	uint32_t SumPortBytes(uint32_t port);
public:
	Ptr<SwitchMmu> m_mmu;

	enum SwitchRole{
		NONE=0,
		TOR,
		ASW,
		CSW, 
	};
	
	void AddXpanderTableEntry(Ipv4Address &srcAddr, Ipv4Address &dstAddr, uint32_t ifidx, uint32_t remainHopToDst);
	void AddRRGTableEntry(Ipv4Address &srcAddr, Ipv4Address &dstAddr, uint32_t ifidx);
	void AddRtWeight(Ipv4Address& srcAddr, Ipv4Address &dstAddr, double weight);
	
	/******************
	 * NP ECN for PCN
	 ******************/
	void SwitchSetPNCounter(uint32_t ifIndex, uint32_t qIndex);


	// INVFD related methods.
	void RecordFlow(uint32_t inDev, uint32_t outDev, CustomHeader &ch);
	void TraceAffectedFlow(uint32_t inDev, uint32_t qIndex);
	void DumpPortTxBytes(FILE *file);
	// Added by qizhou.zqz 2021-1-11 19:03. In XIA MEN. Weight >= 1 and Integer.
	void AddNonShortestTableEntry(Ipv4Address &srcAddr, Ipv4Address &dstAddr, uint32_t intf_idx);
	// Added by qizhou.zqz 2021-1-17 19:33. In SJTU library. 
	void SetSwitchDevRole(SwitchRole role);
	// Added by qizhou.zqz 2021-2-19 14:45. In SJTU library.
	// When one flow completed its transmission, the switch have to clear its record.
	void ClearFlowRecord(Ptr<RdmaQueuePair> qp);


	static TypeId GetTypeId (void);
	SwitchNode();
	void SetEcmpSeed(uint32_t seed);
	void AddTableEntry(Ipv4Address &dstAddr, uint32_t intf_idx);
	void ClearTable();
	bool SwitchReceiveFromDevice(Ptr<NetDevice> device, Ptr<Packet> packet, CustomHeader &ch);
	void SwitchNotifyDequeue(uint32_t ifIndex, uint32_t qIndex, Ptr<Packet> p);
	void SetHostsPerPod(uint32_t hosts);

	// for approximate calc in PINT
	int logres_shift(int b, int l);
	int log2apprx(int x, int b, int m, int l); // given x of at most b bits, use most significant m bits of x, calc the result in l bits
	
	uint32_t RealtimeBytes(uint32_t outDev) {
		uint32_t sum_bytes = 0;
		for(uint32_t i = 0; i < GetNDevices(); i++)
			for(uint32_t j = 0; j < qCnt; j++)
				sum_bytes += m_bytes[i][outDev][j];
		return sum_bytes;
	}

	void SetKsp(bool _ksp) { enable_ksp = _ksp; }
	void SetCongestionAware(bool _congestion_aware) { enable_congestion_aware = _congestion_aware; }
	void SetUniformHash(bool _uniform) { enable_uniform_hash = _uniform; }
	void SetWeightedHash(bool _weighted) { enable_weighted_hash = _weighted; }
	void SetFlowletGap(uint32_t _gap) { m_flowletGapNs = _gap; }
	uint32_t GetFlowletGap() { return m_flowletGapNs; }
	void SetHostsPerSwitch(uint32_t hosts) { m_hostsPerSw = hosts; }
	void SetLp(bool _lp) { enable_lp = _lp; }

private:
	// reference Let it flow(NSDI 2017)
	struct FlowletEntry {
		FlowletEntry(int _output_port, uint64_t _ts, int _flowlet_id=1)
			: output_port(_output_port), lastTs(_ts), flowlet_id(_flowlet_id) {}
		// age indicates whether it expires
		FlowletEntry() : output_port(-1), lastTs(0), flowlet_id(1) {}
		int flowlet_id;
		int output_port;
		uint64_t lastTs;
	};
	uint32_t m_hostsPerSw;
	// flowlet
	uint32_t m_flowletGapNs { 500000 }; // 50us
	EventId  m_flowletChecker;
	std::unordered_map<uint32_t, FlowletEntry > m_flowletRecord;
	
	std::unordered_map<uint32_t,  
		std::unordered_map<uint32_t,
			std::unordered_map<uint32_t,  
				std::vector<uint32_t> > > > m_rtXpanderEntries; // map from ip to possible WCMP port.
	std::unordered_map<uint32_t,  
		std::unordered_map<uint32_t,
			std::vector<int> > > m_rtRRGEntries;

	std::unordered_map<uint32_t, 
		std::unordered_map<uint32_t, 
			std::set<uint32_t> > > m_remainHops;
	
	std::unordered_map<uint32_t, 
		std::unordered_map<uint32_t, 
			std::vector<double> > > m_rtWeight;

	bool enable_ksp; //
	bool enable_uniform_hash;
	bool enable_weighted_hash;
	bool enable_congestion_aware;
	bool enable_lp;

	Ptr<UniformRandomVariable> m_uv;

	// In-network Victim Flow Detection(INVFD) Related Variables.
	std::unordered_map<uint32_t, std::unordered_map<uint32_t, Ptr<RdmaFlow>> > m_passedFlows; // record the flow of ingress port.
	std::unordered_map<uint32_t, uint64_t> m_lastClearTime;
	static const uint64_t clearPeriod = 10000; // 10us
	typedef TracedCallback<Ptr<RdmaFlow>, uint32_t>  RdmaFlowTrace;
	RdmaFlowTrace m_traceRoot;
	RdmaFlowTrace m_traceVictim;
	typedef TracedCallback<uint32_t, uint32_t, uint32_t> PfcTrace;
	PfcTrace m_tracePause;
	PfcTrace m_traceResume;

	std::unordered_map<uint32_t,  std::unordered_map<uint32_t,  std::vector<int> > > m_rtNonShortestTable; // map from ip to possible WCMP port.
	// cache route for flow. Aviod packet reordering.
	// std::unordered_map<uint32_t, uint32_t> m_routeCachce;
	/*************************************
	 * Threshold Routing related variables. 
	 *************************************/
	static const uint32_t podNum = 8;
	uint64_t m_p2pbytes[podNum][podNum];
	uint64_t m_threshold[podNum][podNum];
	
	// std::unordered_map<int, int> m_selectedRoute; // flow hash map to selected route
	MyMap m_selectedRoute;
	uint64_t m_lastResetP2p;

	uint32_t m_devRole; // the devRole can be {TOR, ASW, CSW(In fat-tree)}.
	uint32_t m_hostsPerPod;
};

} /* namespace ns3 */

#endif /* SWITCH_NODE_H */
