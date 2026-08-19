/* XDP filter for the rfmux "fastrx" (100 GbE) flow. Redirects matching packets
 * into the AF_XDP socket bound to RX queue 0.  Everything else continues to
 * the kernel. */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

#include "packet.h"

// Maps ctx->rx_queue_index (which is always 0) to AF_XDP socket FD
struct {
	__uint(type, BPF_MAP_TYPE_XSKMAP);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u32);
} xsks_map SEC(".maps");

SEC("xdp.frags")
int xdp_channel_stream_filter(struct xdp_md *ctx) {
	void *data	 = (void *)(long)ctx->data;
	void *data_end = (void *)(long)ctx->data_end;

	/* Match IPv4 protocol */
	struct ethhdr *eth = data;
	if ((void *)(eth + 1) > data_end || eth->h_proto != bpf_htons(0x0800))
		return XDP_PASS;

	/* Match IPv4 header */
	struct iphdr *ip = (void *)(eth + 1);
	if ((void *)(ip + 1) > data_end ||
			(ip->version != 4 || ip->protocol != IPPROTO_UDP) ||
			(ip->daddr != bpf_htonl(FASTRX_MULTICAST_GROUP_NUM)))
		return XDP_PASS;

	/* Check dest port */
	struct udphdr *udp = (void *)ip + (ip->ihl * 4);
	if ((void *)(udp + 1) > data_end || udp->dest != bpf_htons(FASTRX_PORT))
		return XDP_PASS;

	/* Check magic identifier */
	struct fastrx_packet_header *pkt = (void *)(udp + 1);
	if ((void *)(pkt + 1) > data_end || pkt->magic != FASTRX_PACKET_MAGIC)
		return XDP_PASS;

	/* Filter passed! Redirect to userspace */
	return bpf_redirect_map(&xsks_map, ctx->rx_queue_index, XDP_PASS);
}
