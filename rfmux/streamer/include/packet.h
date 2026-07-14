#pragma once

/* C-compatible packet structure definitions
 * This header is included by both C and C++ code
 */

#ifndef __KERNEL__
# include <stdint.h>
#endif

/* Packet magic numbers */
#define READOUT_PACKET_MAGIC 0x5344494b
#define PFB_PACKET_MAGIC 0x736e6962
#define CHANNEL_STREAM_PACKET_MAGIC 0x4348414e  /* "CHAN" */

/* Multicast groups + ports.
 *
 * Two separate multicast groups so an IGMP-snooping switch can keep the
 * high-rate channel-stream traffic off the 1 GbE control-plane ports:
 *
 *   MULTICAST_GROUP (239.192.0.2): readout + PFB packets (1 GbE-friendly).
 *   CHANNEL_STREAM_MULTICAST_GROUP (239.192.0.3): channel-stream only
 *       (100 GbE). Slower switches that bridge from the 100 GbE fabric
 *       must NOT forward this group to 1 GbE ports.
 */
#define MULTICAST_GROUP "239.192.0.2"
#define CHANNEL_STREAM_MULTICAST_GROUP "239.192.0.3"
#define STREAMER_PORT 9876
#define PFB_STREAMER_PORT 9877
#define CHANNEL_STREAM_PORT 9876

/* Cross-platform packed struct support */
#ifdef _MSC_VER
# pragma pack(push, 1)
# define PACKED
#else
# define PACKED __attribute__((packed))
#endif

/* IRIG-B timestamp structure */
#define SS_PER_SECOND 156250000

struct irigb_timestamp {
	uint32_t y;
	uint32_t d;
	uint32_t h;
	uint32_t m;
	uint32_t s;
	uint32_t ss;
	uint32_t c; /* Bits [17:0]=count, [30:29]=source, [31]=recent */
	uint32_t sbs;
} PACKED;

/* PFB Stream Packets */
struct pfb_packet_header {
	uint32_t magic;
	uint8_t version;
	uint8_t mode; /* 0=PFB1, 1=PFB2, 2=PFB4 */
	uint16_t serial;

	uint16_t slot1;
	uint16_t slot2;
	uint16_t slot3;
	uint16_t slot4;

	uint16_t num_samples; /* sample count in this packet */
	uint8_t module;
	uint8_t _reserved;  /* unused, reads 0 */
	uint32_t seq;
} PACKED;

/* Maximum sample count for PFB packets */
#define PFBPACKET_NSAMP_MAX 1000

/* Maximum-size PFB packet buffer for ioctl/DMA allocation
 * This is a fixed-size buffer that can hold any valid PFB packet */
struct pfb_packet_buffer {
	struct pfb_packet_header hdr;
	int32_t samples[PFBPACKET_NSAMP_MAX * 2];  /* I/Q pairs */
	struct irigb_timestamp ts;
} PACKED;

struct readout_packet_header {
	uint32_t magic;
	uint16_t version;
	uint16_t serial;

	uint8_t num_modules;
	uint8_t flags;
	uint8_t fir_stage;
	uint8_t module;

	uint32_t seq;
} PACKED;

#define LONG_PACKET_CHANNELS 1024
#define SHORT_PACKET_CHANNELS 128

#define LONG_PACKET_VERSION 5
#define SHORT_PACKET_VERSION 6

#define LONG_PACKET_SIZE (LONG_PACKET_CHANNELS*8 + \
		sizeof(struct readout_packet_header) + \
		sizeof(struct irigb_timestamp))

#define SHORT_PACKET_SIZE (SHORT_PACKET_CHANNELS*8 + \
		sizeof(struct readout_packet_header) + \
		sizeof(struct irigb_timestamp))

#define PFB_PACKET_SIZE(__nsamp) (sizeof(struct pfb_packet_header) + \
		((__nsamp)*8) + \
		sizeof(struct irigb_timestamp))

/* High-rate "channel streamer" packets; 100GbE */

#define CHANNEL_STREAM_NUM_PIPELINES 8
#define CHANNEL_STREAM_MAX_SAMPLES_PER_PIPELINE 128  /* one full PFB scan */
#define CHANNEL_STREAM_PACKET_VERSION 0

/* Sample truncation window applied by firmware (16-of-24 bits of I/Q). */
#define CHANNEL_STREAM_TRUNC_LOW  0  /* bits 15:0  (LSB-aligned) */
#define CHANNEL_STREAM_TRUNC_MID  1  /* bits 19:4  (mid) */
#define CHANNEL_STREAM_TRUNC_HIGH 2  /* bits 23:8  (MSB-aligned, default) */

struct channel_stream_packet_header {
	uint32_t magic;
	uint32_t seq;

	uint8_t pipe_snapshot;   /* bitmask of enabled pipelines (K=popcount) */
	uint8_t sample_trunc;    /* CHANNEL_STREAM_TRUNC_{LOW,MID,HIGH} */
	uint8_t module;
	uint8_t version;

	uint16_t tag;
	uint16_t serial;
	uint16_t samples_per_packet; /* # I/Q pairs in payload */
	uint8_t  _reserved[6];

	struct irigb_timestamp ts;
	uint8_t  _ts_pad[30];
} PACKED;  /* sizeof == 86 */

/* Largest legal samples_per_packet: K = NUM_PIPELINES = 8 with all 128
 * channels enabled = 1024. Anything larger violates the firmware contract
 * and must be rejected by receivers before being used to size allocations
 * or index into buffers. */
#define CHANNEL_STREAM_MAX_SAMPLES_PER_PACKET \
	(CHANNEL_STREAM_NUM_PIPELINES * CHANNEL_STREAM_MAX_SAMPLES_PER_PIPELINE)

/* Total UDP payload size for a given samples_per_packet (# I/Q pairs). */
#define CHANNEL_STREAM_PACKET_SIZE(__spp) \
	(sizeof(struct channel_stream_packet_header) + \
	 (__spp) * 2 * (int)sizeof(int16_t))

/* Maximum UDP payload: 86 + 1024*4 = 4182 B (jumbo). */
#define MAX_CHANNEL_STREAM_PACKET_SIZE \
	CHANNEL_STREAM_PACKET_SIZE(CHANNEL_STREAM_MAX_SAMPLES_PER_PACKET)

/* Fixed-size buffer for ioctl/DMA allocation (parallels pfb_packet_buffer). */
struct channel_stream_packet_buffer {
	struct channel_stream_packet_header hdr;
	int16_t samples[CHANNEL_STREAM_NUM_PIPELINES *
	                CHANNEL_STREAM_MAX_SAMPLES_PER_PIPELINE * 2];  /* I/Q pairs */
} PACKED;

#ifdef _MSC_VER
# pragma pack(pop)
#endif
