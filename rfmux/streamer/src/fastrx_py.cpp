/* fastrx Python extension.
 *
 * Maps the UMEM read-only plus the shared control region, and pulls
 * descriptors off its own SPSC ring. Multiple consumers can attach to the same
 * fastrxd and receive the same data, but a slow consumer hurts everyone else.
 */

#include "fastrx.h"

#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <stop_token>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <immintrin.h> /* _mm_pause */
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

class Consumer {
public:
	explicit Consumer(std::string socket_path)
			: socket_path_(std::move(socket_path)) {

		if (socket_path_.empty())
			throw py::value_error("socket_path must not be empty");

		try {
			connect_and_map();
		} catch (...) {
			cleanup();
			throw;
		}
	}

	/* Bring the consumer up. */
	virtual void start() final {
		if (running_)
			return;
		on_start(); /* subclass side up before anything arrives */
		hot_ = std::jthread([this](std::stop_token st) { hot_loop(st); });
		running_ = true;
	}

	/* Wind down, in the reverse order. */
	virtual void stop() final {
		if (!running_)
			return;

		/* Flag first: a subclass blocked in take() waits on this, and would
		 * otherwise sit out its full timeout before noticing. */
		stop_.store(true, std::memory_order_release);
		hot_.request_stop();
		hot_.join();
		on_stop();

		/* cleanup() closes the socket last, prompting fastrxd to reclaim
		 * every frame we still owe. That reclaim is the sole cleanup for
		 * descriptors left in our ring: the hot thread does not drain on
		 * exit (see hot_loop()). */
		cleanup();
		running_ = false;
	}

	virtual ~Consumer() {
		assert(!running_ && "Consumer::stop() must be called before destruction");
	}
	Consumer(const Consumer&) = delete;
	Consumer& operator=(const Consumer&) = delete;

protected:
	/* Called on the hot thread once per packet, with every pipeline it carries.
	 *
	 * iq   one pointer per pipeline, indexed by absolute pipe number. A pipe
	 *      the transmitter is not sending is nullptr, so a subclass tests
	 *      presence rather than decoding hdr.pipe_snapshot itself. Every
	 *      non-null pointer aims into UMEM and dies the moment this returns:
	 *      copy anything you intend to keep.
	 * hdr  the wire header, copied into the descriptor rather than pointing at
	 *      UMEM, so unlike the iq pointers it outlives the call. Carries seq,
	 *      the IRIG-B timestamp, sample_trunc, module, tag and serial.
	 *
	 * Every valid block is exactly SAMPLES_PER_PIPELINE pairs long.
	 */
	virtual void on_packet(const int16_t* const iq[NUM_PIPELINES],
			const fastrx_packet_header& hdr) {

		(void)iq; (void)hdr;
	}

	/* Brought up before the hot thread starts, torn down after it has joined.
	 * Defaults do nothing, so a consumer with no cold side ignores both. */
	virtual void on_start() {}
	virtual void on_stop() {}

public:

	/* False once stop() has run: the control region is unmapped and slot_ is
	 * null, so anything that touches the shared slot must check first. */
	bool running() const { return running_; }

	uint64_t double_releases() const { return double_releases_.load(std::memory_order_relaxed); }
	uint64_t stranded_frames() const { return stranded_frames_.load(std::memory_order_relaxed); }
	uint64_t ring_drops() const { return slot_ ? slot_->ring_drops : 0; }
	uint32_t client_id() const { return client_id_; }
	const std::string& socket_path() const { return socket_path_; }

protected:
	std::atomic<bool> stop_{false};
	fastrxd_client_slot* slot_ = nullptr;
	std::atomic<uint8_t> last_snapshot_{0};

private:
	void connect_and_map() {
		sock_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
		if (sock_ < 0)
			throw std::runtime_error(std::format("fastrx: socket: {}", strerror(errno)));

		struct sockaddr_un addr = {};
		addr.sun_family = AF_UNIX;
		if (socket_path_.size() >= sizeof(addr.sun_path))
			throw std::runtime_error(std::format(
					"fastrx: socket path too long: {}", socket_path_));
		std::memcpy(addr.sun_path, socket_path_.c_str(), socket_path_.size());

		if (connect(sock_, (struct sockaddr*)&addr, sizeof(addr)) < 0)
			/* Name the path: the usual causes are fastrxd not running, a
			 * non-default --socket, or this user not being in the socket's
			 * group -- and the errno alone does not distinguish them. */
			throw std::runtime_error(std::format(
					"fastrx: connect {}: {} "
					"(is fastrxd running, and are you in its group?)",
					socket_path_, strerror(errno)));

		int fds[4] = {-1, -1, -1, -1};
		int nfds = 0;
		{
			struct iovec iov = {.iov_base = &meta_, .iov_len = sizeof(meta_)};
			char cbuf[CMSG_SPACE(sizeof(int) * 8)];
			struct msghdr msg = {};
			msg.msg_iov = &iov;
			msg.msg_iovlen = 1;
			msg.msg_control = cbuf;
			msg.msg_controllen = sizeof(cbuf);
			ssize_t n = recvmsg(sock_, &msg, 0);

			if (n != (ssize_t)sizeof(meta_))
				throw std::runtime_error("fastrx: unexpected setup reply "
						"(is fastrxd running, and are all client slots free?)");

			for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg;
				 cmsg = CMSG_NXTHDR(&msg, cmsg)) {
				if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
					int cnt = (int)((cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int));
					if (cnt > 4) cnt = 4;
					std::memcpy(fds, CMSG_DATA(cmsg), (size_t)cnt * sizeof(int));
					nfds = cnt;
				}
			}
		}
		if (nfds < 2)
			throw std::runtime_error("fastrx: did not receive expected file descriptors");

		if (meta_.abi_version != FASTRXD_ABI_VERSION)
			throw std::runtime_error(std::format(
					"fastrx: ABI mismatch -- fastrxd was built from a different "
					"include/fastrx.h (daemon {:#010x}, client {:#010x}). "
					"Restart fastrxd from the current build.",
					meta_.abi_version, (uint32_t)FASTRXD_ABI_VERSION));

		umem_fd_ = fds[0];
		ctl_fd_ = fds[1];
		client_id_ = meta_.client_id;

		/* UMEM is mapped read-only: consumers discriminate on these bytes but
		 * must never write them. */
		umem_area_ = mmap(nullptr, FASTRXD_UMEM_SIZE, PROT_READ, MAP_SHARED, umem_fd_, 0);
		if (umem_area_ == MAP_FAILED)
			throw std::runtime_error(std::format("fastrx: mmap umem: {}", strerror(errno)));

		ctl_ = static_cast<fastrxd_ctl*>(mmap(nullptr, sizeof(fastrxd_ctl),
					PROT_READ | PROT_WRITE, MAP_SHARED, ctl_fd_, 0));
		if (ctl_ == MAP_FAILED)
			throw std::runtime_error(std::format("fastrx: mmap ctl: {}", strerror(errno)));

		/* The reply and the mapping arrive by different channels, so agreeing with
		 * the reply above does not vouch for the bytes actually mapped here. Both
		 * checks guard the pointer formed just below. */
		if (ctl_->abi_version != FASTRXD_ABI_VERSION)
			throw std::runtime_error(std::format(
					"fastrx: control region ABI mismatch "
					"(region {:#010x}, client {:#010x})",
					ctl_->abi_version, (uint32_t)FASTRXD_ABI_VERSION));

		if (client_id_ >= FASTRXD_MAX_CLIENTS)
			throw std::runtime_error(std::format(
					"fastrx: fastrxd assigned client_id {}, past the {}-slot table",
					client_id_, FASTRXD_MAX_CLIENTS));

		slot_ = &ctl_->clients[client_id_];
	}

	void cleanup() {
		if (umem_area_ && umem_area_ != MAP_FAILED)
			munmap(umem_area_, FASTRXD_UMEM_SIZE);

		if (ctl_ && ctl_ != MAP_FAILED)
			munmap(ctl_, sizeof(fastrxd_ctl));

		umem_area_ = nullptr;
		ctl_ = nullptr;
		slot_ = nullptr;

		if (umem_fd_ >= 0) {
			close(umem_fd_);
			umem_fd_ = -1;
		}

		if (ctl_fd_ >= 0) {
			close(ctl_fd_);
			ctl_fd_ = -1;
		}

		/* Closing the socket signals disconnect; fastrxd frees our slot and
		 * recovers any frames we still hold. */
		if (sock_ >= 0) {
			close(sock_);
			sock_ = -1;
		}
	}

	bool desc_pop(fastrxd_desc& d) {
		auto& head = slot_->descs.head;
		auto& tail = slot_->descs.tail;
		uint32_t t = tail.load(std::memory_order_relaxed);

		if (head.load(std::memory_order_acquire) == t)
			return false;

		d = slot_->descs.entries[t & (FASTRXD_RING_SIZE - 1)];
		tail.store(t + 1, std::memory_order_release);

		return true;
	}

	/* Release our reference to a frame by clearing our ownership bit. Whoever
	 * clears the last bit notifies fastrxd via the return ring; fastrxd is the
	 * sole fill-ring producer. */
	void frame_release(uint64_t frame_addr) {
		auto& owners = ctl_->frame_owners[frame_addr / FASTRXD_FRAME_SIZE];

		const uint32_t bit = 1u << client_id_;
		uint32_t prev = owners.fetch_and(~bit, std::memory_order_acq_rel);

		/* Unreachable: we never owned this frame, or released it twice. */
		if (!(prev & bit)) {
			double_releases_.fetch_add(1, std::memory_order_relaxed);
			return;
		}

		if (prev != bit)
			return; /* another consumer still holds it */

		auto& head = slot_->returns.head;
		auto& tail = slot_->returns.tail;
		uint32_t h = head.load(std::memory_order_relaxed);
		uint32_t t = tail.load(std::memory_order_acquire);
		if ((uint32_t)(h - t) >= FASTRXD_NUM_FRAMES) {
			/* Unreachable: getting here means the frame's last
			 * ownership bit is cleared with no way to tell
			 * fastrxd. There are supposed to be enough
			 * return-queue slots to hold all frames in flight. */
			stranded_frames_.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		slot_->returns.entries[h & (FASTRXD_NUM_FRAMES - 1)] = frame_addr;
		head.store(h + 1, std::memory_order_release);
	}

	void hot_loop(std::stop_token st) {
		while (!st.stop_requested()) {
			fastrxd_desc d;
			if (!desc_pop(d)) {
				_mm_pause();
				continue;
			}

			/* Last-seen snapshot, so callers can tell "my pipe is not in the
			 * stream" from "we kept nothing". */
			last_snapshot_.store(d.hdr.pipe_snapshot, std::memory_order_relaxed);

			/* One pointer per pipeline, nullptr where absent. */
			const int16_t* iq[NUM_PIPELINES] = {};
			for (int p = 0; p < NUM_PIPELINES; p++)
				if (d.payload_off[p])
					iq[p] = reinterpret_cast<const int16_t*>(
						static_cast<const uint8_t*>(umem_area_) +
						d.payload_off[p]);

			on_packet(iq, d.hdr);

			/* Released the moment on_packet() returns: whatever the subclass
			 * wanted, it has copied by now. */
			for (unsigned i = 0; i < d.n_frags; i++)
				frame_release(d.frame_addr[i]);
		}

		/* Withdraw from fastrxd's dispatch mask, so the ring stops
		 * filling. Any frames still in our ring are reclaimed by
		 * fastrxd, which is the best place to do this race-free. */
		slot_->ready.store(0, std::memory_order_release);
	}

	std::string socket_path_;

	int sock_ = -1;
	int umem_fd_ = -1;
	int ctl_fd_ = -1;
	uint32_t client_id_ = 0;

	fastrxd_setup_reply meta_ = {};
	void* umem_area_ = nullptr;
	fastrxd_ctl* ctl_ = nullptr;


	std::jthread hot_;
	bool running_ = false;

	std::atomic<uint64_t> double_releases_{0};
	std::atomic<uint64_t> stranded_frames_{0};
};

class PacketStream : public Consumer {
public:
	explicit PacketStream(std::string socket_path)
		: Consumer(std::move(socket_path)) {}

	~PacketStream() override { stop(); }

	/* Collect exactly n_packets, or fewer on timeout.
	 *
	 * Blocks the calling (Python) thread with the GIL released. Exactly one
	 * futex wake per capture: the hot thread signals when the buffer is full,
	 * not when each packet lands. */
	py::dict capture(uint32_t n_packets, int pipe, double timeout_s) {
		if (pipe < 1 || pipe > NUM_PIPELINES)
			throw py::value_error(std::format(
					"pipe must be in 1..{}, got {}", NUM_PIPELINES, pipe));
		if (n_packets == 0)
			throw py::value_error("n_packets must be positive");
		if (state_.load(std::memory_order_acquire) != State::idle)
			throw std::runtime_error(
				"fastrx: capture() already in progress on this consumer");
		/* stop() runs cleanup(), which unmaps the control region and nulls slot_.
		 * Without this the store to slot_->ready below dereferences null and
		 * takes the interpreter with it. */
		if (!running() || !slot_)
			throw std::runtime_error(
					"fastrx: consumer is stopped; construct a new one");

		/* Allocated fresh, and never reused: this memory leaves for Python at
		 * the end of the call, so there is nothing to hand back or share. */
		const size_t n_iq = (size_t)n_packets * SAMPLES_PER_PIPELINE * 2;
		auto iq = std::make_unique<int16_t[]>(n_iq);
		auto seqs = std::make_unique<uint32_t[]>(n_packets);

		/* Fault in every page before the hot thread can be asked to write it. */
		std::memset(iq.get(), 0, (size_t)n_packets * SAMPLES_PER_PIPELINE * 2 * sizeof(int16_t));
		std::memset(seqs.get(), 0, (size_t)n_packets * sizeof(uint32_t));

		{
			py::gil_scoped_release rel;

			/* Publish the destination before opening the gate. The release
			 * store on state_ pairs with the hot thread's acquire load, so it
			 * cannot see 'capturing' without also seeing these pointers. */
			iq_ = iq.get();
			seqs_ = seqs.get();
			want_ = n_packets;
			pipe_ = pipe - 1;
			count_.store(0, std::memory_order_relaxed);
			restarts_.store(0, std::memory_order_relaxed);
			state_.store(State::capturing, std::memory_order_release);

			/* Only now ask fastrxd to send anything. Between captures our slot
			 * is active but not ready, so fastrxd skips us entirely rather than
			 * filling a ring we would only drain to discard. */
			slot_->ready.store(1, std::memory_order_release);

			std::unique_lock<std::mutex> lk(done_mu_);
			done_cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), [this] {
				return state_.load(std::memory_order_acquire) == State::full
					|| stop_.load(std::memory_order_relaxed);
			});

			/* Stop the flow first, then close the gate: with ready clear
			 * fastrxd stops choosing us, so no further packet can be written
			 * into a buffer we are about to hand to Python. */
			slot_->ready.store(0, std::memory_order_release);
			state_.store(State::idle, std::memory_order_release);
			iq_ = nullptr;
			seqs_ = nullptr;
		}

		uint32_t got = count_.load(std::memory_order_acquire);
		if (got > n_packets)
			got = n_packets;
		return pack(std::move(iq), std::move(seqs), got, n_packets);
	}


protected:
	/* Hot thread: append to the capture buffer, or discard.
	 *
	 * No allocation, no lock, no notify except once at the end. The only writes
	 * are a memcpy into memory nobody else touches and a relaxed store. */
	void on_packet(const int16_t* const iq[NUM_PIPELINES],
				const fastrx_packet_header& hdr) override {

		if (state_.load(std::memory_order_acquire) != State::capturing)
			return;

		if (!iq[pipe_])
			return; /* our pipe is absent from this packet */

		uint32_t i = count_.load(std::memory_order_relaxed);

		/* Ensure contiguous seq */
		if (i && hdr.seq != last_seq_ + 1) {
			restarts_.fetch_add(1, std::memory_order_relaxed);
			i = 0;
		}
		last_seq_ = hdr.seq;

		if (i >= want_) {
			/* Buffer full: close the gate and wake the collector. Taking the
			 * mutex briefly is what makes the wait_for predicate safe against a
			 * missed wakeup; it happens once per capture, not once per packet. */
			state_.store(State::full, std::memory_order_release);
			{ std::lock_guard<std::mutex> lk(done_mu_); }
			done_cv_.notify_one();
			return;
		}

		std::memcpy(iq_ + (size_t)i * SAMPLES_PER_PIPELINE * 2,
				iq[pipe_],
				SAMPLES_PER_PIPELINE * 2 * sizeof(int16_t));
		seqs_[i] = hdr.seq;
		last_snapshot_.store(hdr.pipe_snapshot, std::memory_order_relaxed);
		count_.store(i + 1, std::memory_order_release);
	}

private:
	/* Turn the filled buffers into numpy arrays, transferring ownership. */
	py::dict pack(std::unique_ptr<int16_t[]> iq,
			std::unique_ptr<uint32_t[]> seqs,
			uint32_t got,
			uint32_t want) {

		const py::ssize_t n = got;
		const py::ssize_t ch = SAMPLES_PER_PIPELINE;
		const py::ssize_t s2 = sizeof(int16_t);

		int16_t* iq_raw = iq.get();
		uint32_t* seqs_raw = seqs.get();

		py::capsule iq_owner(iq.release(), [](void* p) {
			delete[] static_cast<int16_t*>(p);
		});
		py::capsule seq_owner(seqs.release(), [](void* p) {
			delete[] static_cast<uint32_t*>(p);
		});

		/* row stride is a whole packet; column stride steps I->I (or Q->Q) */
		auto arr_i = py::array_t<int16_t>({n, ch}, {ch * 2 * s2, 2 * s2}, iq_raw, iq_owner);
		auto arr_q = py::array_t<int16_t>({n, ch}, {ch * 2 * s2, 2 * s2}, iq_raw + 1, iq_owner);
		auto arr_s = py::array_t<uint32_t>({n}, {sizeof(uint32_t)}, seqs_raw, seq_owner);

		return py::dict(
			"i"_a=arr_i,
			"q"_a=arr_q,
			"seq"_a=arr_s,
			"pipe"_a=pipe_ + 1,
			"complete"_a=(got == want),
			"restarts"_a=restarts_.load(std::memory_order_relaxed),
			"pipe_snapshot"_a = last_snapshot_.load(std::memory_order_relaxed));
	}

	/* Where the hot thread stands relative to a capture.
	 *
	 * idle -> capturing -> full -> idle. Only capture() moves it out of full or
	 * into capturing; only on_packet() moves it into full. A single atomic is
	 * enough because the two threads never both write it in the same state. */
	enum class State : uint32_t { idle, capturing, full };
	std::atomic<State> state_{State::idle};

	/* Destination for the capture in progress. Written by capture() before
	 * state_ goes to capturing, and read by the hot thread only while it is. */
	int16_t* iq_ = nullptr;
	uint32_t* seqs_ = nullptr;
	uint32_t want_ = 0;
	int pipe_ = 0; /* set by capture(), read by on_packet() */
	std::atomic<uint32_t> count_{0};

	/* Sequence number of the previous accepted packet, for the contiguity
	 * check. Hot-thread-private: only on_packet() touches it. */
	uint32_t last_seq_ = 0;

	/* Times this capture was abandoned and restarted over a sequence gap.
	 * Reset by capture() and reported in its result dict: nonzero is
	 * survivable, but means the consumer is barely keeping up. */
	std::atomic<uint64_t> restarts_{0};

	/* Signalled once, when the buffer fills. */
	std::mutex done_mu_;
	std::condition_variable done_cv_;
};

/* Offline reader for capture files written by earlier runs. Entirely
 * independent of the live path -- no UMEM, no fastrxd, no threads. */
class PacketFile {
public:
	explicit PacketFile(const std::string& path) {
		fd_ = ::open(path.c_str(), O_RDONLY);
		if (fd_ < 0)
			throw std::runtime_error(std::format(
					"PacketFile: open {}: {}", path, strerror(errno)));
		struct stat st;
		if (fstat(fd_, &st) < 0)
			throw std::runtime_error(std::format("PacketFile: fstat: {}", strerror(errno)));
		file_size_ = (size_t)st.st_size;
		if (file_size_ == 0)
			throw std::runtime_error("PacketFile: file is empty");
		if ((data_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0)) == MAP_FAILED)
			throw std::runtime_error(std::format("PacketFile: mmap: {}", strerror(errno)));

		auto* hdr0 = reinterpret_cast<const fastrx_packet_header*>(data_);
		if (hdr0->magic != FASTRX_PACKET_MAGIC)
			throw std::runtime_error("PacketFile: bad magic in first packet");

		spp_ = hdr0->samples_per_packet;
		stride_ = sizeof(fastrx_packet_header) + (size_t)spp_ * 2 * sizeof(int16_t);
		n_pipes_ = __builtin_popcount(hdr0->pipe_snapshot);
		pipe_snapshot_ = hdr0->pipe_snapshot;

		if (file_size_ % stride_ != 0)
			throw std::runtime_error("PacketFile: file size not a multiple of packet stride");
		num_packets_ = file_size_ / stride_;

		size_t spp_off = offsetof(fastrx_packet_header, samples_per_packet);
		size_t snap_off = offsetof(fastrx_packet_header, pipe_snapshot);
		for (size_t i = 1; i < num_packets_; i++) {
			const uint8_t* h = (uint8_t*)data_ + i * stride_;
			uint16_t spp;
			memcpy(&spp, h + spp_off, sizeof(spp));

			uint8_t snap;
			memcpy(&snap, h + snap_off, sizeof(snap));

			if (spp != spp_ || snap != pipe_snapshot_)
				throw std::runtime_error(std::format(
						"PacketFile: non-uniform packets at index {}", i));
		}
	}

	~PacketFile() {
		if (data_ && data_ != MAP_FAILED) munmap(data_, file_size_);
		if (fd_ >= 0) ::close(fd_);
	}

	PacketFile(const PacketFile&) = delete;
	PacketFile& operator=(const PacketFile&) = delete;

	size_t num_packets() const { return num_packets_; }
	uint16_t samples_per_packet() const { return spp_; }
	uint8_t pipe_snapshot() const { return pipe_snapshot_; }
	int n_pipes() const { return n_pipes_; }

	py::array_t<uint32_t> seq() const {
		py::ssize_t n = (py::ssize_t)num_packets_;
		py::ssize_t stride = (py::ssize_t)stride_;
		size_t off = offsetof(fastrx_packet_header, seq);
		return py::array_t<uint32_t>({n}, {stride},
				reinterpret_cast<const uint32_t*>((uint8_t*)data_ + off),
				py::cast(this));
	}

	py::array_t<int16_t> pipe_iq(int pipe_idx) const {
		if (pipe_idx < 0 || pipe_idx >= NUM_PIPELINES)
			throw py::index_error("pipe index out of range");
		if (!(pipe_snapshot_ & (1 << pipe_idx)))
			throw std::runtime_error("pipe not present in pipe_snapshot");
		int spp_per_pipe = spp_ / n_pipes_;
		size_t pipe_offset = (size_t)__builtin_popcount(pipe_snapshot_ & ((1 << pipe_idx) - 1))
			* (size_t)spp_per_pipe * 2 * sizeof(int16_t);
		size_t base_off = sizeof(fastrx_packet_header) + pipe_offset;
		py::ssize_t n = (py::ssize_t)num_packets_;
		py::ssize_t stride = (py::ssize_t)stride_;
		return py::array_t<int16_t>(
				{n, (py::ssize_t)spp_per_pipe, (py::ssize_t)2},
				{stride, (py::ssize_t)(2 * sizeof(int16_t)), (py::ssize_t)sizeof(int16_t)},
				reinterpret_cast<const int16_t*>((uint8_t*)data_ + base_off),
				py::cast(this));
	}

private:
	int fd_ = -1;
	void* data_ = nullptr;
	size_t file_size_ = 0;
	size_t num_packets_ = 0;
	size_t stride_ = 0;
	uint16_t spp_ = 0;
	uint8_t pipe_snapshot_ = 0;
	int n_pipes_ = 0;
};

PYBIND11_MODULE(_fastrx, m) {
	m.doc() = "AF_XDP fast packet capture for channel-stream data";

	m.attr("NUM_PIPELINES") = NUM_PIPELINES;
	m.attr("MAX_SAMPLES") = SAMPLES_PER_PIPELINE;
	m.attr("ABI_VERSION") = FASTRXD_ABI_VERSION;
	m.attr("MAX_CLIENTS") = FASTRXD_MAX_CLIENTS;

	py::class_<PacketStream>(m, "Consumer",
			"One fastrxd connection, reusable for any number of captures.\n\n"
			"The pipeline is chosen per capture, not at construction: a "
			"connection is expensive and pipe-agnostic, so one consumer serves "
			"every pipeline. 'pipe' is 1-indexed, as everywhere in the Python "
			"API.")
		.def(py::init([](std::string socket_path) {
				 auto* c = new PacketStream(std::move(socket_path));
				 /* Threads start here, not in the constructor: on_packet() is
				  * virtual, and a constructor-started thread would dispatch to
				  * Consumer's version rather than PacketStream's. */
				 c->start();
				 return c;
			 }),
			 "socket_path"_a)
		.def_property_readonly("socket_path", &PacketStream::socket_path)
		.def("capture", &PacketStream::capture,
			 "n_packets"_a, "pipe"_a = 1, "timeout"_a = 5.0,
			 "Collect exactly n_packets, or fewer on timeout.\n\n"
			 "Reusable: call as often as you like on one instance. The returned "
			 "'i' and 'q' arrays are strided views over one buffer whose "
			 "ownership passes to Python, so nothing is copied.")
		.def("stop", &PacketStream::stop)
		.def_property_readonly("double_releases", &PacketStream::double_releases)
		.def_property_readonly("stranded_frames", &PacketStream::stranded_frames)
		.def_property_readonly("ring_drops", &PacketStream::ring_drops)
		.def_property_readonly("client_id", &PacketStream::client_id)
		.def("__enter__", [](PacketStream& c) -> Consumer& { return c; })
		.def("__exit__", [](PacketStream& c, py::object, py::object, py::object) {
			c.stop();
		});

	/* Directory only: the socket inside it is named for the interface fastrxd
	 * serves, so callers compose SOCKET_DIR + "/" + ifname. */
	m.attr("SOCKET_DIR") = FASTRXD_SOCKET_DIR;

	py::class_<PacketFile>(m, "PacketFile")
		.def(py::init<const std::string&>(), "path"_a)
		.def_property_readonly("num_packets", &PacketFile::num_packets)
		.def_property_readonly("samples_per_packet", &PacketFile::samples_per_packet)
		.def_property_readonly("pipe_snapshot", &PacketFile::pipe_snapshot)
		.def_property_readonly("n_pipes", &PacketFile::n_pipes)
		.def("seq", &PacketFile::seq)
		.def("pipe_iq", &PacketFile::pipe_iq, "pipe_idx"_a)
		.def("__len__", &PacketFile::num_packets)
		.def("__enter__", [](PacketFile& f) -> PacketFile& { return f; },
			 py::return_value_policy::reference)
		.def("__exit__", [](PacketFile&, py::object, py::object, py::object) {});
}
