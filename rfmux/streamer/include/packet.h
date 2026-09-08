#pragma once

/* C-compatible packet structure definitions
 * This header is included by both C and C++ code
 */

#if defined(__bpf__)
/* A BPF program should use kernel-side types. Presumably stdint.h would be
 * fine it if worked - but "clang -target bpf" declares no host architecture
 * and it gets confused. */
 #include <linux/types.h>
 typedef __u8  uint8_t;
 typedef __u16 uint16_t;
 typedef __u32 uint32_t;
 typedef __u64 uint64_t;
 typedef __s16 int16_t;
 typedef __s32 int32_t;
#else
 #include <stdint.h>
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

/* Channel-stream wire format. Linux-only (see FASTRX_* constants below), but
 * kept in this shared bucket rather than its own header: it is one more wire
 * format alongside PFB and readout above, and the BPF filter (src/bpf.c)
 * that needs it compiles as C, which this header already supports. */
#define FASTRX_PACKET_MAGIC        0x4348414eu  /* "CHAN" */
#define FASTRX_PACKET_VERSION      0
#define FASTRX_MULTICAST_GROUP     "239.192.0.3"
#define FASTRX_MULTICAST_GROUP_NUM 0xefc00003u
#define FASTRX_PORT                9876

#define NUM_PIPELINES              8
#define SAMPLES_PER_PIPELINE       128
#define MAX_SAMPLES_PER_PACKET     (NUM_PIPELINES * SAMPLES_PER_PIPELINE)

/* Sample truncation window applied by firmware (16-of-24 bits of I/Q). */
typedef enum {
	TRUNC_LOW  = 0,  /* bits 15:0  (LSB-aligned) */
	TRUNC_MID  = 1,  /* bits 19:4  (mid) */
	TRUNC_HIGH = 2,  /* bits 23:8  (MSB-aligned, default) */
} fastrx_trunc_t;

struct fastrx_packet_header {
	uint32_t magic;
	uint32_t seq;

	uint8_t  pipe_snapshot;   /* bitmask of enabled pipelines */
	uint8_t  sample_trunc;    /* fastrx_trunc_t */
	uint8_t  module;
	uint8_t  version;

	uint16_t tag;
	uint16_t serial;
	uint16_t samples_per_packet;  /* # I/Q pairs in payload */
	uint8_t  _reserved[6];

	struct irigb_timestamp ts;
	uint8_t  _ts_pad[30];
} PACKED;  /* sizeof == 86 */

#define FASTRX_PACKET_SIZE(__spp) \
	(sizeof(struct fastrx_packet_header) + (__spp) * 2 * (int)sizeof(int16_t))

#ifdef __cplusplus
static_assert(sizeof(struct fastrx_packet_header) == 86,
              "packet header layout no longer matches the wire format");
#else
_Static_assert(sizeof(struct fastrx_packet_header) == 86,
               "packet header layout no longer matches the wire format");
#endif

#ifdef _MSC_VER
# pragma pack(pop)
#endif
