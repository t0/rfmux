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

/* Readout packet constants */
#define MULTICAST_GROUP "239.192.0.2"
#define STREAMER_PORT 9876
#define PFB_STREAMER_PORT 9877

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
#define PFBPACKET_NSAMP_MAX 2000

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

#ifdef _MSC_VER
# pragma pack(pop)
#endif
