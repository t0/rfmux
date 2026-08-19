/* fastrxd: AF_XDP setup, packet ingest, and multi-client dispatch.
 *
 * Usage: sudo fastrxd [--frame-headroom=N] [--uid=UID] [--gid=GID] <ifname>
 *
 * Starts as root to perform privileged setup once: attaches the embedded XDP
 * filter, creates UMEM and an AF_XDP socket on queue 0, inserts the socket
 * into xsks_map[0], joins the channel-stream multicast group, and pre-fills
 * the FILL ring.  Then drops privileges irreversibly.
 *
 * Clients receive the UMEM memfd (read-only) and a control memfd holding
 * per-client descriptor rings plus a shared frame refcount table (see
 * include/fastrx.h for the layout).
 *
 * Threads:
 *   - ingest : owns the RX ring AND the FILL ring.  Walks the RX ring,
 *              reassembles fragments, validates headers, dispatches descriptors
 *              to every ready client, and collects frames clients have
 *              released and returns them to the FILL ring. Zero-copy.
 *   - main   : accept/disconnect loop over the Unix socket (epoll).
 */

#include <arpa/inet.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <getopt.h>
#include <grp.h>
#include <linux/ethtool.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <xdp/libxdp.h>
#include <xdp/xsk.h>

#include "fastrx.h"

#include <atomic>
#include <immintrin.h>   /* _mm_pause */
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <format>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

/* eBPF is linked in via CMakeLists */
extern const char _binary_fastrx_bpf_o_start[], _binary_fastrx_bpf_o_end[];

static bool verbose = false;

/* FIXME: die() and fail() do different things but the distinction seems
 * poorly motivated and using both printf() and std::format() is odd. */
__attribute__((noreturn, format(printf, 1, 2)))
static void die(const char* fmt, ...) {
	std::fprintf(stderr, "fastrxd: ");

	va_list ap;
	va_start(ap, fmt);
	std::vfprintf(stderr, fmt, ap);
	va_end(ap);
	std::fputc('\n', stderr);
	std::exit(1);
}

__attribute__((format(printf, 1, 2)))
static void warn(const char* fmt, ...) {
	std::fprintf(stderr, "fastrxd: ");

	va_list ap;
	va_start(ap, fmt);
	std::vfprintf(stderr, fmt, ap);
	va_end(ap);
	std::fputc('\n', stderr);
}

/* Build a std::runtime_error with a printf-style message. */
template <typename... A>
[[noreturn]] static void fail(std::format_string<A...> fmt, A&&... args) {
	throw std::runtime_error(std::format(fmt, std::forward<A>(args)...));
}

/* Limit libbpf chatter */
static int libbpf_print(enum libbpf_print_level level, const char* fmt, va_list ap) {
	if (level != LIBBPF_WARN && !verbose)
		return 0;
	std::fprintf(stderr, "fastrxd: libbpf: ");
	return std::vfprintf(stderr, fmt, ap);
}

/* Force the interface down to a single RX queue. */
static bool force_single_rx_queue(const std::string& ifname) {
	int s = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
	if (s < 0) {
		std::perror("fastrxd: socket for ethtool");
		return false;
	}

	struct ifreq ifr = {};
	std::strncpy(ifr.ifr_name, ifname.c_str(), IFNAMSIZ - 1);

	struct ethtool_channels ch = {};
	ch.cmd = ETHTOOL_GCHANNELS;
	ifr.ifr_data = (caddr_t)&ch;
	if (ioctl(s, SIOCETHTOOL, &ifr) < 0) {
		/* Some drivers do not implement channel queries at all. Warn rather
		 * than fail: such a device typically has a single queue anyway, and a
		 * genuine mismatch will show up immediately as zero received packets. */
		warn("cannot query RX channels on %s (%s); assuming a single queue",
				ifname.c_str(), std::strerror(errno));
		close(s);
		return true;
	}

	/* combined_count covers queues that carry both RX and TX; rx_count covers
	 * RX-only queues. Either kind can receive, so both must come down to a
	 * single receiving queue. */
	if (ch.combined_count + ch.rx_count == 1) {
		close(s);
		return true;
	}

	std::fprintf(stderr,
			"fastrxd: %s has %u combined + %u rx queues; reducing to 1 "
			"(only queue 0 is redirected to AF_XDP)\n",
			ifname.c_str(), ch.combined_count, ch.rx_count);

	struct ethtool_channels set = ch;
	set.cmd = ETHTOOL_SCHANNELS;

	/* Keep whichever kind the driver actually offers: some expose only
	 * combined queues, some only separate rx/tx. */
	if (ch.combined_count) {
		set.combined_count = 1;
		set.rx_count = 0;
		set.tx_count = 0;
	} else {
		set.rx_count = 1;
		set.tx_count = ch.tx_count ? 1 : 0;
	}
	ifr.ifr_data = (caddr_t)&set;
	if (ioctl(s, SIOCETHTOOL, &ifr) < 0) {
		std::fprintf(stderr,
				"fastrxd: could not set %s to a single RX queue: %s\n"
				"fastrxd: run manually:  sudo ethtool -L %s combined 1\n",
				ifname.c_str(), std::strerror(errno), ifname.c_str());
		close(s);
		return false;
	}

	/* Confirm it took: a driver may silently clamp to a different value. */
	ch.cmd = ETHTOOL_GCHANNELS;
	ifr.ifr_data = (caddr_t)&ch;
	if (ioctl(s, SIOCETHTOOL, &ifr) == 0 &&
			ch.combined_count + ch.rx_count != 1) {
		std::fprintf(stderr,
				"fastrxd: %s still reports %u combined + %u rx queues after "
				"reconfiguration; packets on queues other than 0 will be lost\n",
				ifname.c_str(), ch.combined_count, ch.rx_count);
		close(s);
		return false;
	}

	close(s);
	return true;
}

/* The client cannot name libxdp's constant (it does not link libxdp), so the
 * value is stated in fastrx.h; check the two agree here, where both are visible. */
static_assert(FASTRXD_FRAME_SIZE == XSK_UMEM__DEFAULT_FRAME_SIZE,
	"FASTRXD_FRAME_SIZE disagrees with XSK_UMEM__DEFAULT_FRAME_SIZE");

constexpr uint32_t kQueueId = 0;

/* XDP passes Ethernet frames verbatim, so we need to know about their headers
 * (or at least the space they take up! Validity checks belong at the BPF, not
 * here.) */
constexpr size_t kNetHdrLen = sizeof(struct ethhdr)
		+ sizeof(struct iphdr)
		+ sizeof(struct udphdr);

class Session {
public:
	xdp_program* prog = nullptr;
	xsk_umem* umem = nullptr;
	xsk_socket* xsk = nullptr;
	void* umem_area = nullptr;
	int memfd = -1; /* UMEM; clients mmap read-only */
	int ctl_fd = -1; /* control region; clients mmap read-write */
	int xsk_fd = -1;
	int igmp_fd = -1; /* holds multicast membership, programs NIC MAC filter */
	int ifindex = 0;

	xsk_ring_prod fill;
	xsk_ring_cons comp;
	xsk_ring_cons rx;
	xsk_ring_prod tx;

	/* Shared control region. */
 	fastrxd_ctl* ctl = nullptr;

	/* Acquires everything: attaches XDP, creates the UMEM and AF_XDP socket,
	 * maps the control region, joins the multicast group, pre-fills the FILL
	 * ring.  Throws std::runtime_error on any failure, so a partly-built Session
	 * is destroyed rather than leaked. */
	Session(const std::string& ifname, uint32_t frame_headroom);

	/* Releases the XDP program, which is attached to the netdev.
	 * Everything else (memfds, mappings, sockets, UMEM) is reclaimed by
	 * the OS at exit */
	~Session() {
		if(prog && ifindex)
			xdp_program__detach(prog, ifindex, XDP_MODE_NATIVE, 0);
		if(prog)
			xdp_program__close(prog);
	}

	Session(const Session&) = delete;
	Session& operator=(const Session&) = delete;

	/* Claim a free slot for a newly connected client. Called from the main
	 * thread; the ingest thread reads 'active' with acquire so it observes a
	 * fully-zeroed ring before the slot goes live. Returns -1 if full. */
	int acquire_slot() {
		for (uint32_t c = 0; c < FASTRXD_MAX_CLIENTS; c++) {
			auto& slot = ctl->clients[c];
			auto& active = slot.active;
			auto& draining = slot.draining;
			if (active.load(std::memory_order_acquire))
				continue;

			/* Still being cleaned up by reclaim: handing it over now would leak
			 * every frame the previous tenant left referenced. */
			if (draining.load(std::memory_order_acquire))
				continue;

			/* Reset the slot before publishing it: a previous tenant may have
			 * left stale indices and counters behind. */
			slot.descs.head.store(0, std::memory_order_relaxed);
			slot.descs.tail.store(0, std::memory_order_relaxed);
			slot.returns.head.store(0, std::memory_order_relaxed);
			slot.returns.tail.store(0, std::memory_order_relaxed);
			slot.ready.store(0, std::memory_order_relaxed);
			slot.dispatched = 0;
			slot.ring_drops = 0;
			slot.client_id = c;

			/* Publishes every store above: the client's acquire load of "active"
			 * pairs with this, so it cannot observe a half-reset slot. */
			active.store(1, std::memory_order_release);
			return (int)c;
		}
		return -1;
	}

	/* Send the shared fds to a newly connected client. */
	bool send_setup(int peer, int client_id) {
		fastrxd_setup_reply reply = {
			.abi_version = FASTRXD_ABI_VERSION,
			.client_id = (uint32_t)client_id,
		};

		/* Client needs two file descriptors: the UMEM memfd (RO) and
		 * the control region (RW) */
		int fds[2] = { memfd, ctl_fd };

		struct iovec iov = {
			.iov_base = &reply,
			.iov_len = sizeof(reply)
		};

		union {
			char buf[CMSG_SPACE(sizeof(fds))];
			struct cmsghdr align; /* forces correct alignment for CMSG_FIRSTHDR */
		} cmsg_u{};

		struct msghdr msg = {};
		msg.msg_iov = &iov;
		msg.msg_iovlen = 1;
		msg.msg_control = cmsg_u.buf;
		msg.msg_controllen = sizeof(cmsg_u.buf);

		struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
		cmsg->cmsg_level = SOL_SOCKET;
		cmsg->cmsg_type = SCM_RIGHTS;
		cmsg->cmsg_len = CMSG_LEN(sizeof(fds));
		std::memcpy(CMSG_DATA(cmsg), fds, sizeof(fds));

		if (sendmsg(peer, &msg, 0) < 0) {
			std::perror("fastrxd: sendmsg");
			return false;
		}
		if (verbose)
			std::fprintf(stderr, "fastrxd: client %d connected (peer fd %d)\n",
					client_id, peer);
		return true;
	}

	/* Ownership mask for the frame at this UMEM address: bit c set while client
	 * c still owes a release. */
	std::atomic<uint32_t>& frame_owners_at(uint64_t frame_addr) {
		return ctl->frame_owners[frame_addr / FASTRXD_FRAME_SIZE];
	}
};

/* Create and map the shared control region. */
Session::Session(const std::string& ifname, uint32_t frame_headroom) {
	int err;

	/* signedness for ifindex is oddly inconsistent */
	ifindex = (int)if_nametoindex(ifname.c_str());
	if (!ifindex)
		fail("unknown interface {}", ifname.c_str());

	size_t bpf_obj_size = (size_t)(_binary_fastrx_bpf_o_end - _binary_fastrx_bpf_o_start);
	bpf_object* bpf_obj = bpf_object__open_mem(_binary_fastrx_bpf_o_start,
			bpf_obj_size, nullptr);
	if (libbpf_get_error(bpf_obj))
		throw std::runtime_error("bpf_object__open_mem failed");

	DECLARE_LIBXDP_OPTS(xdp_program_opts, popts,
			.obj = bpf_obj,
			.prog_name = "xdp_channel_stream_filter");
	prog = xdp_program__create(&popts);
	if (libxdp_get_error(prog)) {
		bpf_object__close(bpf_obj);
		prog = nullptr;   /* not ours to detach */
		throw std::runtime_error("xdp_program__create failed");
	}

	xdp_program__set_xdp_frags_support(prog, true);
	if ((err = xdp_program__attach(prog, ifindex, XDP_MODE_NATIVE, 0))) {
		/* Never attached, so ~Session() must not try to detach it. */
		xdp_program__close(prog);
		prog = nullptr;
		fail("native XDP attach failed: {}\n"
			"fastrxd requires native XDP mode; SKB mode does not support AF_XDP redirect.\n"
			"Ensure the NIC driver supports native XDP and no other XDP program is loaded.",
			std::strerror(-err));
	}

	memfd = memfd_create("xsk_umem", MFD_CLOEXEC);
	if (memfd < 0)
		fail("memfd_create: {}", std::strerror(errno));

	if (ftruncate(memfd, FASTRXD_UMEM_SIZE) < 0)
		fail("ftruncate: {}", std::strerror(errno));

	if (MAP_FAILED == (umem_area = mmap(nullptr, FASTRXD_UMEM_SIZE,
			PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0)))
		fail("mmap umem: {}", std::strerror(errno));

	/* frame_headroom positions the sample payload within the UMEM frame.
	 * d->addr in the rx path points to frame_base + XDP_PACKET_HEADROOM(256) +
	 * frame_headroom, so for the sample fragment to be 512-byte aligned:
	 * (256 + frame_headroom + net_hdr + pkt_hdr) % 512 == 0
	 * (256 + frame_headroom + 128) % 512 == 0
	 * frame_headroom = 128
	 * The value is overridable via --frame-headroom */
	if (frame_headroom >= FASTRXD_FRAME_SIZE - 256u)
		fail("frame_headroom={} exceeds kernel limit ({})",
			frame_headroom, FASTRXD_FRAME_SIZE - 256u - 1);

	/* Refuse a headroom that would split a pipeline block across frames.
	 * Clients index each pipeline's samples at a single offset, so a block must
	 * lie wholly within one frame. */
	constexpr size_t kBlockBytes = SAMPLES_PER_PIPELINE * 2 * sizeof(int16_t);
	size_t payload_at = 256u + frame_headroom + kNetHdrLen +
			sizeof(fastrx_packet_header);
	if (payload_at % kBlockBytes)
		fail("frame_headroom={} puts the payload at frame offset {}, which is not "
			"a multiple of the {}-byte pipeline block: a block would straddle "
			"a frame boundary and be unreadable.  Try {}.",
			frame_headroom, payload_at, kBlockBytes,
			frame_headroom + (kBlockBytes - payload_at % kBlockBytes));

	/* FILL and RX rings are sized to the whole UMEM pool */
	xsk_umem_config umem_cfg = {
		.fill_size = FASTRXD_NUM_FRAMES,
		.comp_size = XSK_RING_CONS__DEFAULT_NUM_DESCS,
		.frame_size = FASTRXD_FRAME_SIZE,
		.frame_headroom = frame_headroom,
		.flags = 0,
	};
	if ((err = xsk_umem__create(&umem, umem_area, FASTRXD_UMEM_SIZE, &fill, &comp, &umem_cfg)))
		fail("xsk_umem__create: {}", std::strerror(-err));

	/* XDP_USE_SG is required, since packets may span multiple UMEM frames. */
	xsk_socket_config xsk_cfg = {
		.rx_size = FASTRXD_NUM_FRAMES,
		.tx_size = 0,
		.libxdp_flags = XSK_LIBXDP_FLAGS__INHIBIT_PROG_LOAD,
		.xdp_flags = 0,
		.bind_flags = XDP_USE_SG,
	};
	if ((err = xsk_socket__create(&xsk, ifname.c_str(), kQueueId, umem, &rx, &tx, &xsk_cfg)))
		fail("xsk_socket__create({}): {}", ifname.c_str(), std::strerror(-err));

	xsk_fd = xsk_socket__fd(xsk);

	bpf_object* obj = xdp_program__bpf_obj(prog);
	bpf_map* xsks_map = bpf_object__find_map_by_name(obj, "xsks_map");
	if (!xsks_map)
		throw std::runtime_error("xsks_map not found in BPF object");

	if ((err = bpf_map__update_elem(xsks_map, &kQueueId, sizeof(kQueueId), &xsk_fd, sizeof(xsk_fd), BPF_ANY)))
		fail("update xsks_map: {}", std::strerror(-err));

	/* Map the shared control region.  memfd pages start zeroed. */
	if ((ctl_fd = memfd_create("fastrxd_ctl", MFD_CLOEXEC)) < 0)
		fail("memfd_create ctl: {}", std::strerror(errno));

	if (ftruncate(ctl_fd, (off_t)sizeof(fastrxd_ctl)) < 0)
		fail("ftruncate ctl: {}", std::strerror(errno));

	void* ctl_p = mmap(nullptr, sizeof(fastrxd_ctl), PROT_READ | PROT_WRITE, MAP_SHARED, ctl_fd, 0);
	if (ctl_p == MAP_FAILED)
		fail("mmap ctl: {}", std::strerror(errno));

	ctl = static_cast<fastrxd_ctl*>(ctl_p);
	ctl->abi_version = FASTRXD_ABI_VERSION;

	if(verbose)
		std::fprintf(stderr, "fastrxd: ctl region %zu B (%u frames, %d client slots)\n",
			sizeof(fastrxd_ctl), FASTRXD_NUM_FRAMES, FASTRXD_MAX_CLIENTS);

	/* Join the multicast group (so network hardware admits traffic - this
	 * includes on-NIC filters even in switch-free deployments.) */
	if ((igmp_fd = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0)) < 0)
		fail("igmp socket: {}", std::strerror(errno));

	struct in_addr group_addr;
	inet_pton(AF_INET, FASTRX_MULTICAST_GROUP, &group_addr);
	struct ip_mreqn mreq = {};
	mreq.imr_multiaddr = group_addr;
	mreq.imr_ifindex = ifindex;
	if (setsockopt(igmp_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0)
		fail("IP_ADD_MEMBERSHIP: {}", std::strerror(errno));

	if(verbose)
		std::fprintf(stderr, "fastrxd: joined %s on ifindex %d (NIC MAC filter programmed)\n",
			FASTRX_MULTICAST_GROUP, ifindex);

	/* Pre-fill the FILL ring with the whole pool */
	uint32_t idx;
	uint32_t reserved = xsk_ring_prod__reserve(&fill, FASTRXD_NUM_FRAMES, &idx);
	for (uint32_t i = 0; i < reserved; i++)
		*xsk_ring_prod__fill_addr(&fill, idx + i) = (uint64_t)i * FASTRXD_FRAME_SIZE;
	xsk_ring_prod__submit(&fill, reserved);

	if (reserved < FASTRXD_NUM_FRAMES)
		warn("FILL ring took only %u of %u frames.\n",
				reserved, FASTRXD_NUM_FRAMES);
	else if (verbose)
		std::fprintf(stderr, "fastrxd: pre-filled all %u FILL ring entries "
			"(%.0f MiB reachable)\n", reserved,
			(double)FASTRXD_UMEM_SIZE / (1 << 20));
}

/* Fragment list for one packet, accumulated as RX descriptors arrive.
 * With XDP_USE_SG a packet may span multiple UMEM frames; the header lives in
 * frag[0] and the sample payload follows at a deterministic offset. */
struct PendingPkt {
	struct Frag {
		uint64_t frame; /* UMEM frame base (for fill-ring return) */
		const uint8_t* data; /* umem_area + d->addr */
		uint32_t len;
		uint64_t off; /* d->addr: offset of data within the UMEM */
	};
	Frag frag[FASTRXD_MAX_PKT_FRAGS];
	int n_frags = 0;
	bool overflow = false;
};

/* The header, copied out of the frame at validation time. Everything after
 * validation works from this copy rather than re-reading UMEM, so the wire bytes
 * are read once and cannot change underneath a decision already made. */
using ValidatedHdr = fastrx_packet_header;

class Ingest {
public:
	explicit Ingest(Session& s) : s_(s) {}

	/* Bring a slot into the ingest thread's per-pass sweeps. Called from the
	 * accept loop once a slot has been handed to a new client. */
	void mark_slot_occupied(int c) {
		occupied_mask_.fetch_or(1u << c, std::memory_order_release);
	}

	/* Retire a slot on disconnect.
	 *
	 * A departed client still owns references: any descriptor left unread in
	 * its desc ring is a frame whose ownership bit it never cleared.  Those
	 * must be dropped or the frame leaks permanently.
	 *
	 * Reclaiming them is always deferred to the ingest thread, where it
	 * can be done race-free after the client is gone. */
	void retire_slot(int c) {
		auto& slot = s_.ctl->clients[c];

		/* Quiet by default: short-lived consumers (get_samples, the HUD) connect
		 * and disconnect hundreds of times a second, so per-connection logging is
		 * pure noise. Anything anomalous still reports. */
		if (verbose || slot.ring_drops)
			std::fprintf(stderr,
				"fastrxd: client %d released: %llu dispatched, "
				"%llu ring drops\n",
				c,
				(unsigned long long)slot.dispatched,
				(unsigned long long)slot.ring_drops);

		/* Order matters: 'active' stops dispatch first (from the next mask
		 * rebuild onward), then 'draining' asks reclaim to clean up.  The slot
		 * stays out of acquire_slot()'s hands until reclaim clears 'draining'. */
		slot.active.store(0, std::memory_order_release);
		drain_count_.fetch_add(1, std::memory_order_acq_rel);
		slot.draining.store(1, std::memory_order_release);
	}

	void start() {
		/* One thread: it owns the RX ring, the FILL ring, and reclamation. */
		ingest_thread_ = std::thread([this] { ingest_loop(); });
	}

	void stop() {
		stop_.store(true, std::memory_order_relaxed);
		if (ingest_thread_.joinable())
			ingest_thread_.join();
	}

	/* Ingest summary, printed at shutdown. */
	void report_stats() const {
		uint64_t rx = rx_packets_.load(std::memory_order_relaxed);
		uint64_t ret = frames_returned_.load(std::memory_order_relaxed);
		std::fprintf(stderr,
			"fastrxd: %llu packets received, %llu frames returned\n",
			(unsigned long long)rx, (unsigned long long)ret);

		if (uint64_t sp = short_pkt_drops_.load(std::memory_order_relaxed))
			warn("%llu packets dropped for partial pipelines: the transmitter "
				"sent a shape clients cannot be told about",
				(unsigned long long)sp);
	}

private:
	/* Hand a frame straight back to the FILL ring.  Called only from the
	 * ingest thread, which owns the FILL ring, so no synchronisation is
	 * needed. */
	void return_frame(uint64_t frame) {
		uint32_t idx;
		uint32_t got = xsk_ring_prod__reserve(&s_.fill, 1, &idx);

		if (got != 1)
			die("FILL ring refused frame 0x%llx! Double release?",
				(unsigned long long)frame);

		*xsk_ring_prod__fill_addr(&s_.fill, idx) = frame;
		xsk_ring_prod__submit(&s_.fill, 1);
		frames_returned_.fetch_add(1, std::memory_order_relaxed);
	}

	void stage_packet_returns(const PendingPkt& pkt) {
		for (int i = 0; i < pkt.n_frags; i++)
			return_frame(pkt.frag[i].frame);
	}

	/* Check a reassembled packet against what actually arrived, and snapshot its
	 * header. */
	bool validate(const PendingPkt& pkt, ValidatedHdr& out) {
		if (pkt.overflow)
			return false; /* more fragments than we can describe */

		const size_t min_len = kNetHdrLen + sizeof(fastrx_packet_header);
		if (pkt.frag[0].len < min_len)
			return false; /* too short to even hold the header */

		auto* hdr = reinterpret_cast<const fastrx_packet_header*>(
				pkt.frag[0].data + kNetHdrLen);
		if (hdr->magic != FASTRX_PACKET_MAGIC ||
				hdr->version != FASTRX_PACKET_VERSION)
			return false; /* not ours */

		uint16_t spp = hdr->samples_per_packet;
		if (spp == 0 || spp > MAX_SAMPLES_PER_PACKET)
			return false;

		size_t total = 0;
		for (int i = 0; i < pkt.n_frags; i++)
			total += pkt.frag[i].len;
		if (total < min_len + (size_t)spp * 2 * sizeof(int16_t))
			return false; /* promised samples did not arrive */

		/* Every present pipeline must contribute a full block. */
		int n_pipes = __builtin_popcount(hdr->pipe_snapshot);
		if (!n_pipes || spp != n_pipes * SAMPLES_PER_PIPELINE) {
			/* Counted, unlike the rejections above: those mean traffic we do not
			 * care about, this one means a transmitter sending a shape we cannot
			 * describe to clients -- worth knowing about. */
			short_pkt_drops_.fetch_add(1, std::memory_order_relaxed);
			return false;
		}

		out = *hdr;
		return true;
	}

	/* Dispatch one validated packet to every active client. */
	void dispatch_packet(const PendingPkt& pkt, const ValidatedHdr& hdr, uint32_t eligible) {
		rx_packets_.fetch_add(1, std::memory_order_relaxed);

		/* Map each pipeline to an absolute UMEM offset, following the fragment
		 * chain where the payload spans two frames.  validate() has already
		 * established that every present pipeline contributes a full block. */
		size_t cum[FASTRXD_MAX_PKT_FRAGS + 1] = {};
		for (int i = 0; i < pkt.n_frags; i++)
			cum[i + 1] = cum[i] + pkt.frag[i].len;

		const size_t block = SAMPLES_PER_PIPELINE * 2 * sizeof(int16_t);
		size_t at = kNetHdrLen + sizeof(fastrx_packet_header);

		fastrxd_desc desc = {};
		for (int p = 0; p < NUM_PIPELINES; p++) {
			if (!(hdr.pipe_snapshot & (1u << p)))
				continue; /* absent */

			/* Which fragment holds this block?  It must lie wholly
			 * inside one. */
			int f = -1;
			for (int i = 0; i < pkt.n_frags; i++)
				if (at >= cum[i] && at + block <= cum[i + 1]) {
					f = i;
					break;
				}

			/* Supposedly unreachable: the constructor rejects any
			 * frame_headroom that would put the payload off a
			 * block boundary, and blocks follow consecutively from
			 * there, so each lies wholly inside one fragment. */
			if (f < 0)
				die("pipeline %d block at packet offset %zu straddles a frame "
					"boundary; frame_headroom validation should have made this "
					"impossible", p, at);

			desc.payload_off[p] = pkt.frag[f].off + (at - cum[f]);
			at += block;
		}

		/* Every frame the payload touches, each released once by each client. */
		desc.n_frags = (uint8_t)pkt.n_frags;
		for (int i = 0; i < pkt.n_frags; i++)
			desc.frame_addr[i] = pkt.frag[i].frame;
		desc.hdr = hdr;

		/* "eligible" contains the recipient set in bitmask form */
		if (!eligible) {
			/* nobody wants it */
			stage_packet_returns(pkt);
			return;
		}

		/* Claim the frame: publish the set of clients that will owe a release. */
		for (int i = 0; i < pkt.n_frags; i++)
			s_.frame_owners_at(desc.frame_addr[i])
					.store(eligible, std::memory_order_relaxed);

		uint32_t cand = eligible;
		while (cand) {
			uint32_t c = (uint32_t)__builtin_ctz(cand);
			cand &= cand - 1;
			auto& slot = s_.ctl->clients[c];
			if (desc_ring_push(slot, desc)) {
				slot.dispatched++;
				continue;
			}
			/* Ring full -- drop the references we just published on this client's
			 * behalf, across every frame, returning any we were last to hold. */
			slot.ring_drops++;
			for (int i = 0; i < pkt.n_frags; i++) {
				auto& owners = s_.frame_owners_at(desc.frame_addr[i]);
				if (owners.fetch_and(~(1u << c), std::memory_order_acq_rel) == (1u << c))
					return_frame(desc.frame_addr[i]);
			}
		}
	}

	/* Push a descriptor into a client's ring */
	static bool desc_ring_push(fastrxd_client_slot& slot, const fastrxd_desc& d) {
		auto& head = slot.descs.head;
		auto& tail = slot.descs.tail;
		uint32_t h = head.load(std::memory_order_relaxed);
		uint32_t t = tail.load(std::memory_order_acquire);

		if ((uint32_t)(h - t) >= FASTRXD_RING_SIZE)
			return false;

		slot.descs.entries[h & (FASTRXD_RING_SIZE - 1)] = d;

		head.store(h + 1, std::memory_order_release);
		return true;
	}

	void ingest_loop() {
		PendingPkt pend;

		while (!stop_.load(std::memory_order_relaxed)) {
			/* Reclamation work, once per RX batch rather than per packet.  It
			 * runs BEFORE the eligibility mask is rebuilt: dispatch and reclaim
			 * share this thread, so a slot reclaim frees here can never be
			 * selected below.  The reverse order would let reclaim close a
			 * slot's rings after the mask was built but before dispatch used
			 * it, stranding a descriptor in a ring nobody will ever drain. */
			collect_client_returns();
			drain_retired_slots();

			/* Recompute which slots are active and ready and may receive. */
			uint32_t occ = occupied_mask_.load(std::memory_order_acquire);
			uint32_t eligible = 0;
			while (occ) {
				uint32_t c = (uint32_t)__builtin_ctz(occ);
				occ &= occ - 1;
				auto& slot = s_.ctl->clients[c];
				auto& active = slot.active;
				auto& ready = slot.ready;
				if (active.load(std::memory_order_acquire) &&
						ready.load(std::memory_order_acquire))
					eligible |= 1u << c;
			}

			uint32_t idx_rx = 0;
			uint32_t avail = xsk_ring_cons__peek(&s_.rx, FASTRXD_RING_SIZE, &idx_rx);
			if (!avail) {
				_mm_pause(); /* Nothing to do */
				continue;
			}

			for (uint32_t i = 0; i < avail; i++) {
				const xdp_desc* d = xsk_ring_cons__rx_desc(&s_.rx, idx_rx + i);

				uint64_t frame_base = d->addr & ~(uint64_t)(FASTRXD_FRAME_SIZE - 1);
				bool more = (d->options & XDP_PKT_CONTD) != 0;

				if (pend.n_frags < FASTRXD_MAX_PKT_FRAGS) {
					pend.frag[pend.n_frags++] = {
						frame_base,
						static_cast<const uint8_t*>(s_.umem_area) + d->addr,
						d->len,
						d->addr };
				} else {
					pend.overflow = true;
					return_frame(frame_base);
				}

				if (more)
					continue;

				/* Last fragment: hand the packet on if it validates, and give
				 * its frames back if it does not. */
				ValidatedHdr vhdr;
				if (validate(pend, vhdr))
					dispatch_packet(pend, vhdr, eligible);
				else
					stage_packet_returns(pend);

				/* Start the next packet's fragment list.
				 *
				 * "pend" lives outside both loops because a
				 * packet can span an ingest pass, so its state
				 * has to survive from one iteration to the next. */
				pend.n_frags = 0;
				pend.overflow = false;
			}

			xsk_ring_cons__release(&s_.rx, avail);
		}
	}

	/* Release frames abandoned by disconnected clients.
	 *
	 * A descriptor left unread in a retired slot's ring holds a reference the
	 * client will never drop, so without this the pool drains and the NIC
	 * starves.  Safe to consume that ring here: 'active' is already clear, so
	 * nothing is producing into it. */
	void drain_retired_slots() {
		for (uint32_t c = 0; c < FASTRXD_MAX_CLIENTS; c++) {
			auto& slot = s_.ctl->clients[c];
			auto& draining = slot.draining;
			if (!draining.load(std::memory_order_acquire))
				continue;

			auto& dhead = slot.descs.head;
			auto& dtail = slot.descs.tail;
			uint32_t h = dhead.load(std::memory_order_acquire);
			uint32_t t = dtail.load(std::memory_order_acquire);

			uint32_t n = (uint32_t)(h - t);
			if (n > FASTRXD_RING_SIZE) n = FASTRXD_RING_SIZE;
			for (uint32_t k = 0; k < n; k++) {
				const fastrxd_desc& d =
					slot.descs.entries[(t + k) & (FASTRXD_RING_SIZE - 1)];

				/* Every frame the descriptor named, not just the first: a
				 * multi-frame payload leaves the client owing one release each. */
				unsigned nf = d.n_frags;
				if (nf > FASTRXD_MAX_PKT_FRAGS)
					continue;       /* corrupt descriptor; leave it alone */
				for (unsigned i = 0; i < nf; i++) {
					uint64_t fa = d.frame_addr[i];
					if (fa % FASTRXD_FRAME_SIZE != 0 ||
							fa / FASTRXD_FRAME_SIZE >= FASTRXD_NUM_FRAMES)
						continue;
					auto& owners = s_.frame_owners_at(fa);
					if (owners.fetch_and(~(1u << c), std::memory_order_acq_rel) == (1u << c))
						return_frame(fa);
				}
			}
			dtail.store(t + n, std::memory_order_release);

			/* Sweep the return ring one more time, from inside the same pass that
			 * just closed the desc ring.  collect_client_returns() ran before us
			 * this pass, but a dying client can push a frame between that sweep
			 * and this point: it drives its bit to empty and stores to
			 * returns.head with no regard for our handshake, since socket close
			 * (which triggered retirement) says nothing about its threads having
			 * finished.  Collecting here, after the desc ring is closed, is what
			 * makes the emptiness test below conclusive rather than a guess. */
			collect_returns_for(c);

			auto& rhead = slot.returns.head;
			auto& rtail = slot.returns.tail;
			if (rhead.load(std::memory_order_acquire) != rtail.load(std::memory_order_acquire))
				continue;   /* still arriving; try again next pass */

			/* Both rings are now closed and empty, and every frame this client
			 * owed has had its bit cleared, so nothing can reference the slot
			 * again.  Clear 'occupied' before 'draining': the former drops it
			 * from the per-pass sweeps, the latter frees it for reuse, and
			 * acquire_slot() may hand it out the instant it sees that. */

			occupied_mask_.fetch_and(~(1u << c), std::memory_order_release);
			draining.store(0, std::memory_order_release);
		}
	}

	/* Drain one client's return ring, handing each frame back to the NIC. */
	void collect_returns_for(uint32_t c) {
		auto& slot = s_.ctl->clients[c];
		auto& head = slot.returns.head;
		auto& tail = slot.returns.tail;
		uint32_t t = tail.load(std::memory_order_relaxed);
		uint32_t h = head.load(std::memory_order_acquire);

		/* Clamp, never mask: a full ring has h - t == FASTRXD_NUM_FRAMES,
		 * which a mask would alias to empty -- and since the client stops
		 * pushing at exactly full, neither side would ever move again.
		 * (A correct client can never fill the ring at all -- see fastrx.h --
		 * so past here only garbage counters are being contained.) */
		uint32_t avail = (uint32_t)(h - t);
		if (!avail)
			return;
		if (avail > FASTRXD_NUM_FRAMES)
			avail = FASTRXD_NUM_FRAMES; /* corrupt head; take one ring's worth */

		/* A frame reaching a return ring means the client cleared the last
		 * ownership bit -- it checks that before pushing. Only the address needs
		 * validating here, so a buggy or hostile client cannot inject an
		 * arbitrary address into the FILL ring. */
		for (uint32_t k = 0; k < avail; k++) {
			uint64_t fa = slot.returns.entries[(t + k) & (FASTRXD_NUM_FRAMES - 1)];
			if (fa % FASTRXD_FRAME_SIZE != 0 || fa / FASTRXD_FRAME_SIZE >= FASTRXD_NUM_FRAMES) {
				warn("client %u returned invalid frame 0x%llx; ignoring",
						c, (unsigned long long)fa);
				continue;
			}
			return_frame(fa);
		}
		/* Consume exactly what we read, valid or not: an invalid entry is
		 * discarded, not retried forever. */
		tail.store(t + avail, std::memory_order_release);
	}

	/* Collect frames clients have finished with. Called once per RX batch. */
	void collect_client_returns() {
		uint32_t occupied = occupied_mask_.load(std::memory_order_relaxed);
		while (occupied) {
			uint32_t c = (uint32_t)__builtin_ctz(occupied);
			occupied &= occupied - 1;
			collect_returns_for(c);
		}
	}

	Session& s_;

	/* Frames handed back to the NIC.  Should track rx_packets_ closely: every
	 * received frame is returned exactly once, so a growing gap means frames are
	 * being held or lost. */
	std::atomic<uint64_t> frames_returned_{0};

	std::thread ingest_thread_;
	std::atomic<bool> stop_{false};
	std::atomic<uint64_t> rx_packets_{0};

	/* Set while any client slot needs draining, so the reclaim loop can skip
	 * probing all eight slots */
	std::atomic<uint32_t> drain_count_{0};

	/* Bit c set while slot c is occupied (active or still draining). */
	std::atomic<uint32_t> occupied_mask_{0};

	/* Packets whose pipelines were not all full-length */
	std::atomic<uint64_t> short_pkt_drops_{0};
};

/* Parse a uid or gid, rejecting anything that is not a plain positive integer. */
static bool parse_id(const char* s, uint32_t* out) {
	errno = 0;
	char* end;
	unsigned long v = std::strtoul(s, &end, 10);

	/* strtoul happily accepts a leading '-' and wraps it, so reject the sign
	 * explicitly rather than trying to detect the wrap afterwards. */
	if (s[0] == '-' || s[0] == '+' || s[0] == '\0')
		return false;
	if (*end != '\0' || errno == ERANGE)
		return false;
	if (v == 0 || v > 0xfffffffeu)   /* 0 is root; (uid_t)-1 means "no change" */
		return false;

	*out = (uint32_t)v;
	return true;
}

static int listen_socket(const char* path, uid_t uid, gid_t gid) {
	/* The directory is only a container, so it can be world-traversable; the
	 * socket inside it is what carries the restriction. mkdir failing with
	 * EEXIST is normal -- /run is a tmpfs, but a restart within one boot finds
	 * the directory already there. */
	char dir[sizeof(((struct sockaddr_un*)nullptr)->sun_path)];
	std::snprintf(dir, sizeof(dir), "%s", path);
	if (char* slash = std::strrchr(dir, '/'); slash && slash != dir) {
		*slash = '\0';
		if (mkdir(dir, 0755) < 0 && errno != EEXIST)
			die("mkdir %s: %m", dir);
		if (chown(dir, uid, gid) < 0)
			die("chown %s: %m", dir);
	}

	struct sockaddr_un addr = {};
	addr.sun_family = AF_UNIX;
	if (std::strlen(path) >= sizeof(addr.sun_path))
		die("socket path too long (max %zu): %s", sizeof(addr.sun_path) - 1, path);
	std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);

	/* A socket left by a previous run would make bind() fail with EADDRINUSE.
	 * Unlinking is safe because a live fastrxd holds the same path: if one is
	 * already running, its socket is removed here and the listen below takes
	 * over -- so refuse instead when something is actually listening. */
	int probe = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (probe >= 0) {
		if (connect(probe, (struct sockaddr*)&addr, sizeof(addr)) == 0)
			die("another fastrxd is already listening on %s", path);
		close(probe);
	}
	if (unlink(path) < 0 && errno != ENOENT)
		die("unlink %s: %m", path);

	int srv = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (srv < 0)
		die("socket: %m");

	/* umask would otherwise clear bits from the mode bind() applies, so set the
	 * permissions explicitly afterwards rather than relying on it. */
	if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0)
		die("bind %s: %m", path);
	if (chown(path, uid, gid) < 0)
		die("chown %s: %m", path);
	if (chmod(path, 0660) < 0)
		die("chmod %s: %m", path);
	if (listen(srv, FASTRXD_MAX_CLIENTS) < 0)
		die("listen: %m");

	return srv;
}

static void usage(FILE* out, const char* argv0) {
	std::fprintf(out, "usage: sudo %s -i <ifname> [options]\n", argv0);
	std::fprintf(out, " -i, --interface=IFNAME interface to capture from (required)\n");
	std::fprintf(out, " -r, --frame-headroom=N UMEM headroom bytes\n");
	std::fprintf(out, " -u, --uid=UID drop privileges to UID (default: $SUDO_UID)\n");
	std::fprintf(out, " -g, --gid=GID drop privileges to GID (default: $SUDO_GID)\n");
	std::fprintf(out, " -s, --socket-path=PATH listen path (default: "
			FASTRXD_SOCKET_DIR "/<ifname>)\n");
	std::fprintf(out, " -v, --verbose log every client connect/disconnect\n");
	std::fprintf(out, " -h, --help show this help\n");
}

int main(int argc, char** argv) {
	/* Default frame_headroom:
	 * (XDP_PACKET_HEADROOM=256) + frame_headroom + net_hdr(42) +
	 * pkt_hdr(86) = 128 bytes total. For 512-byte alignment of each
	 * pipeline block: (256 + 128 + 128) % 512 = 0 */
	uint32_t frame_headroom = 128;

	uint32_t uid = 0;
	uint32_t gid = 0;
	/* nullptr until we know ifname, since the default is derived from it. */
	const char* sock_path = nullptr;
	const char* ifname = nullptr;

	static const struct option long_opts[] = {
		{ "interface", required_argument, nullptr, 'i' },
		{ "frame-headroom", required_argument, nullptr, 'r' },
		{ "uid", required_argument, nullptr, 'u' },
		{ "gid", required_argument, nullptr, 'g' },
		{ "socket-path", required_argument, nullptr, 's' },
		{ "verbose", no_argument, nullptr, 'v' },
		{ "help", no_argument, nullptr, 'h' },
		{ nullptr, 0, nullptr, 0 },
	};

	int opt;
	while ((opt = getopt_long(argc, argv, "i:r:u:g:s:vh", long_opts, nullptr)) != -1) {
		switch (opt) {
		case 'i':
			ifname = optarg;
			break;
		case 'r': {
			char* end;
			unsigned long v = std::strtoul(optarg, &end, 0);
			if (*end != '\0' || v > 4095u) {
				std::fprintf(stderr, "fastrxd: invalid frame_headroom %s\n", optarg);
				return 1;
			}
			frame_headroom = (uint32_t)v;
			break;
		}
		case 'u':
			if (!parse_id(optarg, &uid)) {
				std::fprintf(stderr, "fastrxd: invalid uid '%s'\n", optarg);
				return 1;
			}
			break;
		case 'g':
			if (!parse_id(optarg, &gid)) {
				std::fprintf(stderr, "fastrxd: invalid gid '%s'\n", optarg);
				return 1;
			}
			break;
		case 's':
			sock_path = optarg;
			break;
		case 'v':
			verbose = true;
			break;
		case 'h':
			usage(stdout, argv[0]);
			return 0;
		default:
			usage(stderr, argv[0]);
			return 1;
		}
	}

	/* Report bad usage and missing privilege together. */
	bool bad_usage = false;
	if (!ifname) {
		/* No positional form: a bare interface name would be indistinguishable
		 * from a mistyped flag's operand. */
		warn("no interface: pass -i/--interface");
		bad_usage = true;
	}
	if (optind != argc) {
		warn("unexpected argument '%s'", argv[optind]);
		bad_usage = true;
	}
	if (geteuid() != 0) {
		warn("must run as root (euid %u): fastrxd needs CAP_NET_ADMIN, CAP_BPF\n"
			"and CAP_NET_RAW to attach the XDP program and open an AF_XDP\n"
			"socket, and drops to --uid/--gid immediately afterwards",
			(unsigned)geteuid());
		bad_usage = true;
	}
	if (bad_usage) {
		std::fprintf(stderr, "\n");
		usage(stderr, argv[0]);
		std::fprintf(stderr, "\ne.g.  sudo %s -i %s\n",
				argv[0], ifname ? ifname : "<ifname>");
		return 1;
	}

	/* Default to one socket per interface, so two fastrxd instances on two NICs
	 * do not contend for the same path. --socket-path overrides it outright:
	 * disambiguating between interfaces is the operator's call, not ours. */
	std::string sock_default;
	if (!sock_path) {
		sock_default = std::string(FASTRXD_SOCKET_DIR "/") + ifname;
		sock_path = sock_default.c_str();
	}

	/* Limit libbpf chatter */
	libbpf_set_print(libbpf_print);

	if(verbose)
		std::fprintf(stderr,
			 "fastrxd: ifname=%s, embedded BPF object = %zu bytes\n",
			 ifname,
			 (size_t)(_binary_fastrx_bpf_o_end - _binary_fastrx_bpf_o_start));

	/* Only queue 0 is redirected to AF_XDP, so the NIC must not spread the flow
	 * across queues. Do this before attaching XDP: changing the channel count
	 * resets the rings. */
	if (!force_single_rx_queue(ifname))
		return 1;

	/* Ownership splits at the fork below: the parent destroys the Session (which
	 * detaches the XDP program), the child releases it without destroying, since
	 * it lacks the privilege to detach and must not try. */
	std::unique_ptr<Session> sess;
	try {
		sess = std::make_unique<Session>(ifname, frame_headroom);
	} catch (const std::exception& e) {
		std::fprintf(stderr, "fastrxd: %s\n", e.what());
		return 1;
	}

	/*
	 * Drop privileges
	 */

	/* Fall back to the invoking sudo user where -u/-g weren't given. */
	if (const char* s = getenv("SUDO_UID"); uid == 0 && s)
		parse_id(s, &uid);
	if (const char* s = getenv("SUDO_GID"); gid == 0 && s)
		parse_id(s, &gid);

	/* Still zero: neither -u/-g nor sudo's environment supplied a usable value.
	 * parse_id() has already rejected 0 and anything unparseable, so reaching
	 * here means we have no target to drop to. */
	if (uid == 0 || gid == 0)
		die("no uid/gid to drop to: pass -u/-g or run via sudo");

	/* Bind before dropping privileges: /run is root-owned, so the directory and
	 * socket cannot be created afterwards. Ownership is handed to the target
	 * uid/gid here, which is what makes the socket usable once we are no longer
	 * root, and what restricts who may connect. */
	int srv = listen_socket(sock_path, uid, gid);
	if (verbose)
		std::fprintf(stderr, "fastrxd: listening on %s (uid %u gid %u, mode 0660)\n",
				sock_path, uid, gid);

	/* The XDP program is attached to the netdev, which outlives this
	 * process, so something has to detach it. Detaching needs privilege we
	 * are about to drop irreversibly, so fork() and keep a root process
	 * around to clean up. */
	pid_t worker = fork();
	if (worker < 0)
		die("fork: %m");

	if (worker > 0) {
		/* Parent: privileged, and does nothing but wait.
		 * SIGINT and SIGTERM must be IGNORED here, because ctrl-c
		 * signals the whole foreground process group (including the
		 * parent) and the default action is to terminate immediately,
		 * killing the reaper before waitpid() returns.
		 *
		 * Ignoring them is safe because the child is signalled
		 * independently and runs its own orderly shutdown; the parent
		 * then exits when waitpid() reports the child gone. That also
		 * keeps the detach strictly after the child has stopped
		 * receiving. */
		struct sigaction ign = {};
		ign.sa_handler = SIG_IGN;
		sigemptyset(&ign.sa_mask);
		sigaction(SIGINT, &ign, nullptr);
		sigaction(SIGTERM, &ign, nullptr);
		sigaction(SIGHUP, &ign, nullptr);

		int status = 0;
		while (waitpid(worker, &status, 0) < 0 && errno == EINTR)
			;

		/* sess goes out of scope here: ~Session() detaches the XDP program, the
		 * one resource that outlives this process. */
		return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
	}

	/* Child: the parent owns the XDP program, so release without destroying --
	 * running ~Session() here would try to detach it without privilege, and the
	 * worker must leave that to the reaper. */
	Session& session = *sess.release();

	/* Drop privileges irreversibly and keep no capabilities. */
	if (setgroups(0, nullptr) < 0 ||
			setgid(gid) < 0 ||
			setuid(uid) < 0 ||
			setuid(0) == 0)
		die("Failed to drop privileges: %m");

	/* If the parent dies first nobody is left to detach the XDP program, so ask
	 * the kernel to take us down with it rather than run on unsupervised. */
	if (prctl(PR_SET_PDEATHSIG, SIGTERM) < 0)
		die("prctl(PR_SET_PDEATHSIG): %m");

	if(verbose)
		std::fprintf(stderr, "fastrxd: dropped privileges to uid=%ld gid=%ld\n",
	 			(long)uid, (long)gid);

	static volatile sig_atomic_t shutdown_flag = 0;
	struct sigaction sa = {};
	sa.sa_handler = +[](int) { shutdown_flag = 1; };
	sigemptyset(&sa.sa_mask);
	sigaction(SIGINT, &sa, nullptr);
	sigaction(SIGTERM, &sa, nullptr);
	sigaction(SIGHUP, &sa, nullptr);
	signal(SIGPIPE, SIG_IGN);

	/* Ingest starts before the first client connects: frames cycle through
	 * the fill ring regardless of whether anyone is listening. */
	Ingest ingest(session);
	ingest.start();

	int ep = epoll_create1(EPOLL_CLOEXEC);
	if (ep < 0)
		die("epoll_create1: %m");

	auto ep_add = [&](int fd, uint32_t events) {
		struct epoll_event ev = {};
		ev.events = events;
		ev.data.fd = fd;
		if (epoll_ctl(ep, EPOLL_CTL_ADD, fd, &ev) < 0)
			die("epoll_ctl ADD: %m");
	};
	ep_add(srv, EPOLLIN);

	/* peer fd -> client slot index, so a disconnect can free the right slot. */
	std::map<int, int> slot_of;

	while (!shutdown_flag) {
		struct epoll_event evs[8];
		int n = epoll_wait(ep, evs, 8, -1);
		if (n < 0) {
			if (errno == EINTR)
				continue;
			die("epoll_wait: %m");
		}
		for (int i = 0; i < n && !shutdown_flag; i++) {
			int fd = evs[i].data.fd;
			uint32_t ev = evs[i].events;

			if (fd == srv) {
				int peer = accept4(srv, nullptr, nullptr, SOCK_CLOEXEC);
				if (peer < 0) {
					if (errno == EINTR || errno == EAGAIN)
						continue;
					die("accept: %m");
				}
				int c = session.acquire_slot();
				if (c < 0) {
					std::fprintf(stderr,
							"fastrxd: rejecting client, all %d slots busy\n",
							FASTRXD_MAX_CLIENTS);
					close(peer);
					continue;
				}
				if (!session.send_setup(peer, c)) {
					ingest.retire_slot(c);
					close(peer);
					continue;
				}
				ingest.mark_slot_occupied(c);
				slot_of[peer] = c;
				ep_add(peer, EPOLLRDHUP | EPOLLHUP | EPOLLERR);
				continue;
			}

			if (ev & (EPOLLRDHUP | EPOLLHUP | EPOLLERR)) {
				auto it = slot_of.find(fd);
				if (it != slot_of.end()) {
					if (verbose)
						std::fprintf(stderr,
								"fastrxd: client fd %d (slot %d) disconnected\n",
								fd, it->second);
					ingest.retire_slot(it->second);
					slot_of.erase(it);
				}
				epoll_ctl(ep, EPOLL_CTL_DEL, fd, nullptr);
				close(fd);
			}
		}
	}

	ingest.stop();
	ingest.report_stats();

	/* Remove the socket we created. */
	unlink(sock_path);

	/* Nothing else to undo: the XDP program is detached by the privileged parent
	 * once it reaps this process. */
	return 0;
}
