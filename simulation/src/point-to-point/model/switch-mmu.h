#ifndef SWITCH_MMU_H
#define SWITCH_MMU_H

#include <unordered_map>
#include <ns3/node.h>

namespace ns3 {

class Packet;

class SwitchMmu: public Object{
public:
	static const uint32_t pCnt = 257;	// Number of ports used
	static const uint32_t qCnt = 8;	// Number of queues/priorities used

	static TypeId GetTypeId (void);

	SwitchMmu(void);

	bool CheckIngressAdmission(uint32_t port, uint32_t qIndex, uint32_t psize);
	bool CheckEgressAdmission(uint32_t port, uint32_t qIndex, uint32_t psize);
	void UpdateIngressAdmission(uint32_t port, uint32_t qIndex, uint32_t psize);
	void UpdateEgressAdmission(uint32_t port, uint32_t qIndex, uint32_t psize);
	void RemoveFromIngressAdmission(uint32_t port, uint32_t qIndex, uint32_t psize);
	void RemoveFromEgressAdmission(uint32_t port, uint32_t qIndex, uint32_t psize);

	bool CheckShouldPause(uint32_t port, uint32_t qIndex);
	bool CheckShouldResume(uint32_t port, uint32_t qIndex);
	void SetPause(uint32_t port, uint32_t qIndex);
	void SetResume(uint32_t port, uint32_t qIndex);
	//void GetPauseClasses(uint32_t port, uint32_t qIndex);
	//bool GetResumeClasses(uint32_t port, uint32_t qIndex);

	uint32_t GetPfcThreshold(uint32_t port);
	uint32_t GetSharedUsed(uint32_t port, uint32_t qIndex);

	bool ShouldSendCN(uint32_t ifindex, uint32_t qIndex);
	
	/********************
	 * PCN's ECN marking
	 ********************/
	bool PcnSendCN(uint32_t ifIndex, uint32_t qIndex); 
	void SetPNCounter(uint32_t ifIndex, uint32_t qIndex);
	
	void ConfigEcn(uint32_t port, double _kmin, double _kmax, double _pmax);
	// void ConfigEcn(uint32_t port, uint32_t _kmin, uint32_t _kmax, double _pmax);
	void ConfigHdrm(uint32_t port, uint32_t size);
	void ConfigNPort(uint32_t n_port);
	void ConfigBufferSize(uint32_t size);
	void ConfigAlpha(double alpha);

	uint32_t GetIngressBytes(uint32_t port, uint32_t qIndex);
	bool  CheckShouldBypass(uint32_t port, uint32_t qIndex);

	double GetAlpha () { return alpha; }

	void SetFixedPfcThreshold(uint32_t pfc) { m_fixedPfcThreshold = pfc; }

	uint32_t GetFixedPfcThreshold() { return m_fixedPfcThreshold; }

	uint32_t GetKmax(uint32_t port) { return kmax[port]; }

	uint32_t GetEgressBytes(uint32_t port) { return egress_bytes[port][3]; }

	// 
	uint32_t m_fixedPfcThreshold;

	// config
	uint32_t node_id;
	uint32_t buffer_size;
	uint32_t pfc_a_shift[pCnt];
	uint32_t reserve;
	uint32_t headroom[pCnt];
	uint32_t resume_offset;
	uint32_t kmin[pCnt], kmax[pCnt];
	double pmax[pCnt];
	uint32_t total_hdrm;
	uint32_t total_rsrv;

	// runtime
	uint32_t shared_used_bytes;
	uint32_t hdrm_bytes[pCnt][qCnt];
	uint32_t ingress_bytes[pCnt][qCnt];
	uint32_t paused[pCnt][qCnt];
	uint32_t egress_bytes[pCnt][qCnt];
	// pcn runtime
	uint32_t PNCounter[pCnt];
	uint32_t pktNumInQueue[pCnt][qCnt];
	double alpha;
};

} /* namespace ns3 */

#endif /* SWITCH_MMU_H */

