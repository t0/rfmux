#pragma once

/* fastrxd <-> client IPC: the shared control region, its rings, and the
 * descriptor handed to a client for one validated packet.
 *
 * FASTRXD_ABI_VERSION is set by cmake to a hash of this header, so the two
 * processes sharing the region below refuse to talk across a layout change. */
#ifndef FASTRXD_ABI_VERSION
# error "FASTRXD_ABI_VERSION undefined"
#endif

#include <atomic>
#include <cstdint>

#include "packet.h"

#define FASTRXD_SOCKET_DIR "/run/fastrxd"

/* Bitmasks, so must be < 32 */
#define FASTRXD_MAX_CLIENTS 8

/* The UMEM pool: DMA-able frames, and the whole latency budget.  Deep queues
 * (FILL, RX) are sized to this so that no frame is ever held back from the
 * NIC. */
#define FASTRXD_NUM_FRAMES 65536
#define FASTRXD_FRAME_SIZE 4096
#define FASTRXD_MAX_PKT_FRAGS 2
#define FASTRXD_UMEM_SIZE  ((uint64_t)FASTRXD_NUM_FRAMES * FASTRXD_FRAME_SIZE)

struct fastrxd_desc {
	/* Where each pipeline's samples start, as an absolute UMEM offset.  Zero
	 * means the transmitter is not sending that pipeline -- offset 0 is inside
	 * the first frame's XDP headroom, so it can never be a real payload.
	 *
	 * One offset per pipeline rather than a base plus a stride, because the
	 * payload need not be contiguous: pipeline blocks are 512 bytes and
	 * 512-byte aligned, so a block is never itself split, but the boundary
	 * between two of them can fall on a frame edge.  Resolving that here means
	 * a client indexes by pipe number and never reasons about fragments. */
	uint64_t payload_off[NUM_PIPELINES];

	/* Frames this packet's payload occupies, each to be released once.  Distinct
	 * entries: a client holds a reference to every frame it can read from, so it
	 * must clear its ownership bit in each. */
	uint64_t frame_addr[FASTRXD_MAX_PKT_FRAGS];
	uint8_t  n_frags;

	/* The wire header exactly as received, validated by fastrxd (magic,
	 * version, samples_per_packet bounds, and that the payload is really
	 * present) before this descriptor was published. */
	struct fastrx_packet_header hdr;
};

/* Single-producer/single-consumer rings shared across the process boundary.
 *
 * head and tail are free-running counters; the ring is empty when they are
 * equal and full when they differ by the ring size.  Producer writes head;
 * consumer writes tail; a misbehaving client can corrupt its own ring but not
 * fastrxd's state. */

#define FASTRXD_CACHELINE 64

/* Must be power of 2 */
#define FASTRXD_RING_SIZE 256

struct fastrxd_desc_ring {
	alignas(FASTRXD_CACHELINE) std::atomic<uint32_t> head;  /* written by fastrxd */
	alignas(FASTRXD_CACHELINE) std::atomic<uint32_t> tail;  /* written by the client */
	alignas(FASTRXD_CACHELINE) struct fastrxd_desc entries[FASTRXD_RING_SIZE];
};

/* One entry per UMEM frame, so clients can never fill it -- this makes
 * reclamation logic straightforward, even when a client disappears */
struct fastrxd_return_ring {
	alignas(FASTRXD_CACHELINE) std::atomic<uint32_t> head;  /* written by the client */
	alignas(FASTRXD_CACHELINE) std::atomic<uint32_t> tail;  /* written by fastrxd */
	alignas(FASTRXD_CACHELINE) uint64_t entries[FASTRXD_NUM_FRAMES];
};

/* Per-client slot in the control region. */
struct fastrxd_client_slot {
	/* Nonzero once fastrxd has claimed this slot. */
	std::atomic<uint32_t> active;

	/* Set when a client disconnects, cleared once fastrxd's reclaim thread has
	 * drained the descriptors it abandoned and released their frames.  The slot
	 * cannot be handed to a new client while this is set: reusing it early would
	 * leak every frame still referenced by the old tenant's unread descriptors. */
	std::atomic<uint32_t> draining;

	uint32_t client_id;  /* index of this slot in clients[] */
	uint64_t dispatched; /* packets handed to this client */
	uint64_t ring_drops; /* packets skipped: desc ring full */

	/* The client sets this once its consuming thread is actually running. */
	std::atomic<uint32_t> ready;

	struct fastrxd_desc_ring descs; /* fastrxd -> client */
	struct fastrxd_return_ring returns; /* client -> fastrxd */
};

/* Control region layout, mapped from ctl_fd at offset 0. */
struct fastrxd_ctl {
	uint32_t abi_version;

	struct fastrxd_client_slot clients[FASTRXD_MAX_CLIENTS];

	/* Per-frame ownership mask: bit c is set while client c still owes a
	 * release on this frame.  Ingest stores the recipient set at dispatch, each
	 * client clears its own bit after copying, and whoever clears the last bit
	 * returns the frame. */
	alignas(FASTRXD_CACHELINE) std::atomic<uint32_t> frame_owners[FASTRXD_NUM_FRAMES];
};

/* Sent alongside (umem_memfd, ctl_memfd) via SCM_RIGHTS.  The client needs
 * these dimensions to map the UMEM and control regions. */
struct fastrxd_setup_reply {
	uint32_t abi_version;     /* must equal FASTRXD_ABI_VERSION */
	uint32_t client_id;       /* this client's slot index in fastrxd_ctl */
};
