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
#include <optional>
#include <stdexcept>
#include <string>
#include <stop_token>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <immintrin.h> /* _mm_pause */
#include <liburing.h>
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
		/* cleanup() is called here in case an instance is constructed
		 * but never started */
		cleanup();
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

	/* First thing the hot thread runs, before any descriptor is popped. */
	virtual void on_hot_start() {}

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
		/* cleanup() can be called several times and must be
		 * idempotent. */
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
		on_hot_start();

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

class PacketCapture : public Consumer {
public:
	explicit PacketCapture(std::string socket_path)
		: Consumer(std::move(socket_path)) {}

	~PacketCapture() override { stop(); }

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

/* Records packets to disk until stopped.
 *
 * Data path: the hot thread copies each accepted record into an SPSC byte
 * ring and never syscalls; one writer thread submits 4 KiB-aligned chunks of
 * that ring to the file via io_uring with O_DIRECT, and completions advance
 * the ring tail. The ring is a memfd mapped twice back-to-back, so a record
 * reservation is always virtually contiguous and a chunk can be submitted
 * from any stream offset without caring about the wrap. O_DIRECT avoids
 * copies into the page cache.
 *
 * If the disk falls behind, the ring fills and records are dropped (counted in
 * overruns, and visible afterwards as seq gaps in the file). */
class PacketWriter : public Consumer {
public:
	PacketWriter(std::string socket_path,
			std::string path,
			uint8_t pipe_mask,
			uint64_t n_packets,
			uint64_t ring_bytes,
			unsigned queue_depth)
		: Consumer(std::move(socket_path)),
		  path_(std::move(path)),
		  mask_(pipe_mask),
		  stride_(stride_for(pipe_mask)),
		  limit_(n_packets),
		  queue_depth_(queue_depth),
		  ring_bytes_((ring_bytes + kChunkAlign - 1) & ~(kChunkAlign - 1)) {

		if (path_.empty())
			throw py::value_error("path must not be empty");

		/* Deliberately required: latching geometry from the stream instead
		 * would make the file's shape depend on whichever packet won the
		 * race with start(). A caller who wants "record what is streaming
		 * now" reads pipe_snapshot from a capture, and says so. */
		if (!mask_)
			throw py::value_error("pipe_mask must name at least one pipe");

		if (queue_depth_ < 1 || queue_depth_ > 1024)
			throw py::value_error("queue_depth must be in 1..1024");

		if (ring_bytes_ < (1 << 20))
			throw py::value_error("ring must be at least 1 MiB");
	}

	~PacketWriter() override { stop(); }

	uint64_t packets() const { return packets_.load(std::memory_order_relaxed); }
	uint64_t bytes_written() const { return tail_.load(std::memory_order_relaxed); }
	uint64_t overruns() const { return overruns_.load(std::memory_order_relaxed); }
	uint64_t dropouts() const { return dropouts_.load(std::memory_order_relaxed); }
	const std::string& path() const { return path_; }

	uint8_t pipe_mask() const { return mask_; }

	std::optional<std::string> error() {
		std::lock_guard<std::mutex> lk(err_mu_);
		return error_.empty() ? std::nullopt
				: std::optional<std::string>(error_);
	}

	/* True once an n_packets limit has been reached. Never true for an
	 * unbounded recording. */
	bool complete() const {
		return state_.load(std::memory_order_acquire) == State::done;
	}

	/* Block until the n_packets limit is reached, the recording fails, or
	 * stop() is called from another thread. timeout_s < 0 waits forever.
	 * Returns whether the limit was actually reached. */
	bool wait(double timeout_s) {
		auto pred = [this] {
			State s = state_.load(std::memory_order_acquire);
			return s == State::done || s == State::failed
					|| stop_.load(std::memory_order_relaxed);
		};
		std::unique_lock<std::mutex> lk(done_mu_);
		if (timeout_s < 0)
			done_cv_.wait(lk, pred);
		else
			done_cv_.wait_for(lk,
					std::chrono::duration<double>(timeout_s), pred);
		return complete();
	}

protected:
	void on_start() override {
		try {
			open_file();
			map_ring();
			setup_uring();

			writer_ = std::jthread([this](std::stop_token st) {
				writer_loop(st);
			});
		} catch (...) {
			teardown();
			throw;
		}
	}

	void on_hot_start() override {
		slot_->ready.store(1, std::memory_order_release);
	}

	/* Hot thread: append one record to the ring, or drop it. No allocation,
	 * no lock, no syscall -- memcpys and relaxed counters only. */
	void on_packet(const int16_t* const iq[NUM_PIPELINES],
			const fastrx_packet_header& hdr) override {

		if (state_.load(std::memory_order_acquire) != State::recording)
			return;

		uint64_t head = head_.load(std::memory_order_relaxed);
		uint64_t tail = tail_.load(std::memory_order_acquire);
		if (ring_bytes_ - (head - tail) < stride_) {
			/* The disk is not keeping up. Count and drop -- the packet
			 * survives in the file as a seq gap, and the caller decides
			 * afterwards what an acceptable overrun count is. */
			overruns_.fetch_add(1, std::memory_order_relaxed);
			return;
		}

		uint8_t* rec = map_ + (head % ring_bytes_);
		std::memcpy(rec, &hdr, sizeof(hdr));
		uint8_t* p = rec + sizeof(hdr);

		/* Zero-extend over pipeline drop-out: a recorded pipe missing from
		 * this packet gets a zero block rather than costing the whole
		 * record. The record's own pipe_snapshot says which blocks are
		 * real, so provenance is per-record and free -- and the file's
		 * timeline stays contiguous across a transmitter reconfiguration.
		 * In the happy path the branch below is always taken and costs
		 * nothing. */
		bool dropout = false;
		for (int pipe = 0; pipe < NUM_PIPELINES; pipe++) {
			if (!(mask_ & (1u << pipe)))
				continue;
			if (iq[pipe])
				std::memcpy(p, iq[pipe], kBlockBytes);
			else {
				std::memset(p, 0, kBlockBytes);
				dropout = true;
			}
			p += kBlockBytes;
		}
		std::memset(p, 0, stride_ - (size_t)(p - rec)); /* <= 7 pad bytes */

		if (dropout)
			dropouts_.fetch_add(1, std::memory_order_relaxed);

		uint64_t n = packets_.fetch_add(1, std::memory_order_relaxed) + 1;
		head_.store(head + stride_, std::memory_order_release);

		if (limit_ && n >= limit_) {
			/* Enough. Stop accepting, and drop ready so fastrxd stops
			 * choosing us at the source rather than filling a ring we
			 * would only drain to discard. Signalled once per recording,
			 * like PacketCapture's buffer-full wakeup: the brief mutex
			 * grab is what makes wait()'s predicate missed-wakeup-safe. */
			state_.store(State::done, std::memory_order_release);
			slot_->ready.store(0, std::memory_order_release);
			{ std::lock_guard<std::mutex> lk(done_mu_); }
			done_cv_.notify_all();
		}
	}

	/* Cold side, after the hot thread has joined: flush, finalize, close. */
	void on_stop() override {
		/* stop_ is set and the hot thread has joined; wake any wait()er
		 * whose predicate includes stop_ -- Consumer::stop() itself never
		 * notifies, so without this the stop_ clause has no wakeup. */
		{ std::lock_guard<std::mutex> lk(done_mu_); }
		done_cv_.notify_all();

		/* Pad the byte stream to a 4 KiB boundary so the final O_DIRECT
		 * write is legal. head_ is ours now -- the hot thread has
		 * joined. */
		uint64_t head = head_.load(std::memory_order_relaxed);
		uint64_t pad = (uint64_t)(-(int64_t)head) & (kChunkAlign - 1);
		if (pad) {
			/* Wait for room; the writer is still draining, so this
			 * terminates unless it has failed -- and then nothing
			 * we write matters anyway. */
			while (state_.load(std::memory_order_acquire) != State::failed
					&& ring_bytes_ - (head - tail_.load(std::memory_order_acquire)) < pad)
				_mm_pause();
			if (state_.load(std::memory_order_acquire) != State::failed) {
				std::memset(map_ + (head % ring_bytes_), 0, pad);
				head_.store(head + pad, std::memory_order_release);
			}
		}

		drain_.store(true, std::memory_order_release);
		writer_.request_stop();
		if (writer_.joinable())
			writer_.join();

		if (state_.load(std::memory_order_acquire) != State::failed) {
			/* The data is on disk; make the header agree. */
			write_file_header(packets_.load(std::memory_order_relaxed));
			if (fd_ >= 0 && fsync(fd_) < 0)
				fail(std::format("{}: fsync: {}", path_, strerror(errno)));
		}

		teardown();
	}

private:
	static constexpr size_t kBlockBytes = SAMPLES_PER_PIPELINE * 2 * sizeof(int16_t);

	/* O_DIRECT submission granularity: every write is this-aligned in both
	 * file offset and length, which satisfies O_DIRECT on any device with
	 * logical blocks up to 4 KiB (i.e. all of them). It doubles as the page
	 * alignment the double-mapped ring needs; this code is gated to x86_64
	 * (see CMakeLists.txt), where pages are 4 KiB. */
	static constexpr uint64_t kChunkAlign = 4096;

	static constexpr uint64_t kChunkMax = 1 << 20; /* largest single write */

	static_assert(FASTRX_FILE_HEADER_BYTES % kChunkAlign == 0,
			"record 0 must start on a chunk boundary");
	static_assert(kChunkMax % kChunkAlign == 0,
			"chunks are built from whole alignment units");

	static uint32_t stride_for(uint8_t mask) {
		size_t s = sizeof(fastrx_packet_header) + (size_t)__builtin_popcount(mask) * kBlockBytes;
		return (uint32_t)((s + 7) & ~size_t(7));
	}

	void fail(std::string msg) {
		{
			std::lock_guard<std::mutex> lk(err_mu_);
			if (error_.empty())
				error_ = std::move(msg);
		}
		state_.store(State::failed, std::memory_order_release);

		/* A wait()er has nothing left to wait for. The empty scope is the
		 * missed-wakeup fence: a waiter holds done_mu_ from evaluating its
		 * predicate until it blocks, so acquiring the mutex here -- even
		 * for an instant -- means our notify cannot land in that window.
		 * (Same idiom as PacketCapture's buffer-full wakeup.) */
		{ std::lock_guard<std::mutex> lk(done_mu_); }
		done_cv_.notify_all();
	}

	void open_file() {
		fd_ = ::open(path_.c_str(),
			O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC,
			0644);
		if (fd_ < 0)
			throw std::runtime_error(std::format(
					"PacketWriter: open {}: {}{}", path_, strerror(errno),
					errno == EINVAL
						? " (does this filesystem support O_DIRECT? tmpfs does not)"
						: ""));

		if (posix_memalign(&hdr_scratch_, kChunkAlign, FASTRX_FILE_HEADER_BYTES))
			throw std::bad_alloc();
	}

	/* The magic ring buffer: one memfd, mapped twice back-to-back, so any
	 * stride_-sized reservation and any submitted chunk is virtually
	 * contiguous no matter where it falls relative to the wrap. */
	void map_ring() {
		ring_fd_ = memfd_create("fastrx-writer-ring", MFD_CLOEXEC);
		if (ring_fd_ < 0)
			throw std::runtime_error(std::format(
					"PacketWriter: memfd_create: {}", strerror(errno)));
		if (ftruncate(ring_fd_, (off_t)ring_bytes_) < 0)
			throw std::runtime_error(std::format(
					"PacketWriter: ftruncate: {}", strerror(errno)));

		void* v = mmap(nullptr, 2 * ring_bytes_, PROT_NONE,
				MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (v == MAP_FAILED)
			throw std::runtime_error(std::format(
					"PacketWriter: mmap reserve: {}", strerror(errno)));
		map_ = static_cast<uint8_t*>(v);

		if (mmap(map_, ring_bytes_, PROT_READ | PROT_WRITE,
					MAP_SHARED | MAP_FIXED, ring_fd_, 0) == MAP_FAILED
				|| mmap(map_ + ring_bytes_, ring_bytes_,
					PROT_READ | PROT_WRITE,
					MAP_SHARED | MAP_FIXED, ring_fd_, 0) == MAP_FAILED)
			throw std::runtime_error(std::format(
					"PacketWriter: mmap ring: {}", strerror(errno)));

		/* Best effort: shmem THP cuts TLB pressure at GB/s rates. */
		madvise(map_, 2 * ring_bytes_, MADV_HUGEPAGE);

		/* Fault every page in both mappings now, so the hot thread never
		 * takes one mid-record. */
		std::memset(map_, 0, ring_bytes_);
		for (uint64_t off = 0; off < ring_bytes_; off += kChunkAlign)
			(void)*(volatile uint8_t*)(map_ + ring_bytes_ + off);
	}

	void setup_uring() {
		int rc = io_uring_queue_init(queue_depth_, &uring_, 0);
		if (rc < 0)
			throw std::runtime_error(std::format(
					"PacketWriter: io_uring_queue_init: {} (io_uring "
					"disabled by sysctl or seccomp?)", strerror(-rc)));
		uring_up_ = true;

		/* Registering the ring as a fixed buffer skips per-write page
		 * pinning, but charges the whole ring against RLIMIT_MEMLOCK --
		 * so failure is expected for unprivileged users with big rings.
		 * Plain write SQEs pin transiently and are not charged; same
		 * loop, one opcode different. */
		struct iovec iov = {.iov_base = map_, .iov_len = 2 * ring_bytes_};
		use_fixed_ = io_uring_register_buffers(&uring_, &iov, 1) == 0;
	}

	void prep_write_at(io_uring_sqe* sqe, uint64_t stream_off, uint32_t len,
			size_t idx) {
		uint8_t* buf = map_ + (stream_off % ring_bytes_);
		uint64_t file_off = FASTRX_FILE_HEADER_BYTES + stream_off;
		if (use_fixed_)
			io_uring_prep_write_fixed(sqe, fd_, buf, len, file_off, 0);
		else
			io_uring_prep_write(sqe, fd_, buf, len, file_off);
		sqe->user_data = idx;
	}

	io_uring_sqe* get_sqe() {
		io_uring_sqe* sqe = io_uring_get_sqe(&uring_);
		if (!sqe) {
			/* SQ full of unsubmitted entries; flush and retry once. */
			io_uring_submit(&uring_);
			sqe = io_uring_get_sqe(&uring_);
		}
		return sqe;
	}

	/* One thread owns the whole file side: it submits chunks, reaps
	 * completions (which arrive out of order at queue depth), and advances
	 * the ring tail over the contiguous completed prefix. Single-owner
	 * state, so none of it needs a lock. */
	void writer_loop(std::stop_token st) {
		(void)st; /* drain_ carries the shutdown signal; see on_stop() */

		/* A provisional header first, so even a recording that dies keeps
		 * a readable file; num_records stays 0 until the clean-close
		 * rewrite in on_stop(). */
		if (!write_file_header(0))
			return; /* fail() already called */

		struct Chunk {
			uint64_t start;     /* stream offset */
			uint32_t len;
			uint32_t remaining; /* 0 == fully written */
		};
		std::vector<Chunk> chunks(queue_depth_);
		size_t front = 0, in_flight = 0;
		uint64_t submitted = 0;
		bool queued = false;

		for (;;) {
			bool draining = drain_.load(std::memory_order_acquire);
			uint64_t head = head_.load(std::memory_order_acquire);

			/* Queue as much 4 KiB-aligned work as there are slots for. */
			while (in_flight < queue_depth_) {
				uint64_t avail = (head - submitted) & ~(kChunkAlign - 1);
				if (!avail)
					break;
				uint32_t len = (uint32_t)std::min(avail, kChunkMax);
				io_uring_sqe* sqe = get_sqe();
				if (!sqe)
					break;
				size_t idx = (front + in_flight) % queue_depth_;
				prep_write_at(sqe, submitted, len, idx);
				chunks[idx] = {submitted, len, len};
				submitted += len;
				in_flight++;
				queued = true;
			}
			if (queued) {
				int rc = io_uring_submit(&uring_);
				if (rc < 0) {
					fail(std::format("io_uring_submit: {}", strerror(-rc)));
					return;
				}
				queued = false;
			}

			if (!in_flight) {
				if (draining && submitted == head)
					return; /* every byte acknowledged */
				/* Idle. A millisecond of poll latency is nothing next
				 * to disk latency, and only this thread pays it. */
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				continue;
			}

			/* Reap: wait briefly for one completion, then everything
			 * that is ready behind it. */
			io_uring_cqe* cqe = nullptr;
			__kernel_timespec ts = {.tv_sec = 0, .tv_nsec = 2'000'000};
			int rc = io_uring_wait_cqe_timeout(&uring_, &cqe, &ts);
			if (rc == -ETIME || rc == -EINTR)
				continue;
			if (rc < 0) {
				fail(std::format("io_uring_wait: {}", strerror(-rc)));
				return;
			}

			while (cqe) {
				size_t idx = (size_t)cqe->user_data;
				Chunk& c = chunks[idx];
				int res = cqe->res;
				io_uring_cqe_seen(&uring_, cqe);

				if (res < 0) {
					fail(std::format("{}: write: {}", path_,
							strerror(-res)));
					return;
				}
				if ((uint32_t)res < c.remaining) {
					/* Short write (ENOSPC territory): push the rest of
					 * this chunk back in. The slot stays in-flight, so
					 * the tail cannot pass it. */
					uint64_t done = c.start + (c.len - c.remaining)
							+ (uint32_t)res;
					c.remaining -= (uint32_t)res;
					io_uring_sqe* sqe = get_sqe();
					if (!sqe) {
						fail("io_uring: no sqe for short-write resubmit");
						return;
					}
					prep_write_at(sqe, done, c.remaining, idx);
					queued = true;
				} else {
					c.remaining = 0;
				}

				/* The tail only advances over the contiguous completed
				 * prefix: a finished chunk behind an unfinished one
				 * stays reserved. */
				while (in_flight && chunks[front].remaining == 0) {
					tail_.store(chunks[front].start + chunks[front].len,
							std::memory_order_release);
					front = (front + 1) % queue_depth_;
					in_flight--;
				}

				if (io_uring_peek_cqe(&uring_, &cqe) < 0)
					cqe = nullptr;
			}
		}
	}

	/* Called twice per recording: once by the writer thread before any data
	 * (num_records = 0), once from on_stop() after the join, with the truth.
	 * Never concurrently. */
	bool write_file_header(uint64_t count) {
		std::memset(hdr_scratch_, 0, FASTRX_FILE_HEADER_BYTES);
		auto* h = static_cast<fastrx_file_header*>(hdr_scratch_);
		h->magic = FASTRX_FILE_MAGIC;
		h->version = FASTRX_FILE_VERSION;
		h->record_stride = stride_;
		h->samples_per_pipe = SAMPLES_PER_PIPELINE;
		h->pipe_mask = mask_;
		h->num_records = count;

		ssize_t n = pwrite(fd_, hdr_scratch_, FASTRX_FILE_HEADER_BYTES, 0);
		if (n != FASTRX_FILE_HEADER_BYTES) {
			fail(std::format("{}: header write: {}", path_,
					n < 0 ? strerror(errno) : "short write"));
			return false;
		}
		return true;
	}

	void teardown() {
		/* Belt and braces for the on_start() failure path; after a normal
		 * on_stop() the writer has already joined. */
		if (writer_.joinable()) {
			drain_.store(true, std::memory_order_release);
			writer_.request_stop();
			writer_.join();
		}
		if (uring_up_) {
			io_uring_queue_exit(&uring_);
			uring_up_ = false;
		}
		if (map_) {
			munmap(map_, 2 * ring_bytes_);
			map_ = nullptr;
		}
		if (ring_fd_ >= 0) {
			close(ring_fd_);
			ring_fd_ = -1;
		}
		if (fd_ >= 0) {
			close(fd_);
			fd_ = -1;
		}
		free(hdr_scratch_);
		hdr_scratch_ = nullptr;
	}

	std::string path_;

	/* Geometry, fixed at construction: settled before any thread exists,
	 * so the hot path never negotiates structure with the stream. */
	const uint8_t mask_;
	const uint32_t stride_;

	uint64_t limit_; /* records to stop after; 0 = unbounded */
	unsigned queue_depth_;
	uint64_t ring_bytes_;

	int fd_ = -1;
	int ring_fd_ = -1;
	uint8_t* map_ = nullptr;      /* 2 * ring_bytes_ of virtual space */
	void* hdr_scratch_ = nullptr; /* one aligned block for the file header */

	io_uring uring_ = {};
	bool uring_up_ = false;
	bool use_fixed_ = false;

	std::jthread writer_;

	/* Free-running byte-stream counters: head is produced (hot thread),
	 * tail is written-and-acknowledged (writer thread). */
	std::atomic<uint64_t> head_{0};
	std::atomic<uint64_t> tail_{0};
	std::atomic<bool> drain_{false};

	/* recording -> done when an n_packets limit is reached; either ->
	 * failed, from any thread that hits an unrecoverable error. */
	enum class State : uint32_t { recording, done, failed };
	std::atomic<State> state_{State::recording};

	std::atomic<uint64_t> packets_{0};
	std::atomic<uint64_t> overruns_{0};
	std::atomic<uint64_t> dropouts_{0};

	/* Signalled at most twice: once if the limit is reached, once on
	 * failure. */
	std::mutex done_mu_;
	std::condition_variable done_cv_;

	std::mutex err_mu_;
	std::string error_;
};

/* Offline reader for capture files written by earlier runs. Entirely
 * independent of the live path -- no UMEM, no fastrxd, no threads. */
class PacketFile {
public:
	explicit PacketFile(const std::string& path) {
		fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
		if (fd_ < 0)
			throw std::runtime_error(std::format(
					"PacketFile: open {}: {}", path, strerror(errno)));
		struct stat st;
		if (fstat(fd_, &st) < 0)
			throw std::runtime_error(std::format("PacketFile: fstat: {}", strerror(errno)));
		file_size_ = (size_t)st.st_size;
		if (file_size_ < FASTRX_FILE_HEADER_BYTES)
			throw std::runtime_error("PacketFile too short");
		if ((data_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0)) == MAP_FAILED)
			throw std::runtime_error(std::format("PacketFile: mmap: {}", strerror(errno)));

		auto* h = reinterpret_cast<const fastrx_file_header*>(data_);
		if (h->magic != FASTRX_FILE_MAGIC)
			throw std::runtime_error("PacketFile: bad magic!");
		if (h->version != FASTRX_FILE_VERSION)
			throw std::runtime_error(std::format(
					"PacketFile: format version {} (this build reads {})",
					h->version, FASTRX_FILE_VERSION));

		stride_ = h->record_stride;
		spp_ = h->samples_per_pipe;
		pipe_mask_ = h->pipe_mask;
		n_pipes_ = __builtin_popcount(pipe_mask_);
		num_records_ = h->num_records;

		if (!pipe_mask_)
			throw std::runtime_error("PacketFile: no pipes recorded");

		const size_t payload = (size_t)n_pipes_ * spp_ * 2 * sizeof(int16_t);
		if (stride_ < sizeof(fastrx_packet_header) + payload || stride_ % 8)
			throw std::runtime_error(std::format(
					"PacketFile: record stride {} cannot hold {} pipes "
					"of {} samples", stride_, n_pipes_, spp_));

		/* The file may end with up to a chunk of O_DIRECT padding beyond
		 * the last record, so the size bounds the count rather than
		 * determining it. */
		const uint64_t cap = (file_size_ - FASTRX_FILE_HEADER_BYTES) / stride_;
		if (num_records_ > cap)
			throw std::runtime_error(std::format(
					"PacketFile: header claims {} records but the file "
					"holds at most {}", num_records_, cap));

		if (num_records_ == 0 && cap) {
			/* num_records is only rewritten when a recording closes
			 * cleanly, so zero-with-data means the writer died. Every
			 * record leads with the wire header: count magics. */
			while (num_records_ < cap) {
				uint32_t magic;
				std::memcpy(&magic, record(num_records_), sizeof(magic));
				if (magic != FASTRX_PACKET_MAGIC)
					break;
				num_records_++;
			}
		}
	}

	~PacketFile() {
		if (data_ && data_ != MAP_FAILED) munmap(data_, file_size_);
		if (fd_ >= 0) ::close(fd_);
	}

	PacketFile(const PacketFile&) = delete;
	PacketFile& operator=(const PacketFile&) = delete;

	size_t num_packets() const { return num_records_; }
	uint16_t samples_per_pipe() const { return spp_; }
	uint16_t pipe_mask() const { return pipe_mask_; }
	uint32_t record_stride() const { return stride_; }
	int n_pipes() const { return n_pipes_; }

	/* The recorded pipes, 1-indexed as everywhere in the Python API. */
	std::vector<int> pipes() const {
		std::vector<int> out;
		for (int p = 0; p < NUM_PIPELINES; p++)
			if (pipe_mask_ & (1u << p))
				out.push_back(p + 1);
		return out;
	}

	py::array_t<uint32_t> seq() const {
		return py::array_t<uint32_t>(
				{(py::ssize_t)num_records_}, {(py::ssize_t)stride_},
				reinterpret_cast<const uint32_t*>(
						record(0) + offsetof(fastrx_packet_header, seq)),
				py::cast(this));
	}

	/* IRIG-B timestamps, one structured element per record. */
	py::array_t<irigb_timestamp> ts() const {
		return py::array_t<irigb_timestamp>(
				{(py::ssize_t)num_records_}, {(py::ssize_t)stride_},
				reinterpret_cast<const irigb_timestamp*>(
						record(0) + offsetof(fastrx_packet_header, ts)),
				py::cast(this));
	}

	/* Every record's wire header, as one structured array. seq() and ts()
	 * are conveniences over two of its fields; anything else the stream
	 * carries (serial, sample_trunc, module, pipe_snapshot, ...) is read
	 * from here -- usually from element 0, since these are constant over
	 * a recording in practice. */
	py::array_t<fastrx_packet_header> headers() const {
		return py::array_t<fastrx_packet_header>(
				{(py::ssize_t)num_records_}, {(py::ssize_t)stride_},
				reinterpret_cast<const fastrx_packet_header*>(record(0)),
				py::cast(this));
	}

	/* (num_packets, samples_per_pipe, 2) int16 view of one pipe's I/Q,
	 * zero-copy over the mapping. pipe is 1-indexed. */
	py::array_t<int16_t> pipe_iq(int pipe) const {
		if (pipe < 1 || pipe > NUM_PIPELINES)
			throw py::value_error(std::format(
					"pipe must be in 1..{}, got {}", NUM_PIPELINES, pipe));
		uint8_t bit = (uint8_t)(1u << (pipe - 1));
		if (!(pipe_mask_ & bit))
			throw py::value_error(std::format(
					"pipe {} was not recorded in this file", pipe));

		size_t rank = (size_t)__builtin_popcount(pipe_mask_ & (bit - 1));
		const uint8_t* base = record(0) + sizeof(fastrx_packet_header)
				+ rank * (size_t)spp_ * 2 * sizeof(int16_t);
		return py::array_t<int16_t>(
				{(py::ssize_t)num_records_, (py::ssize_t)spp_, (py::ssize_t)2},
				{(py::ssize_t)stride_, (py::ssize_t)(2 * sizeof(int16_t)),
					(py::ssize_t)sizeof(int16_t)},
				reinterpret_cast<const int16_t*>(base),
				py::cast(this));
	}

private:
	const uint8_t* record(uint64_t i) const {
		return static_cast<const uint8_t*>(data_) + FASTRX_FILE_HEADER_BYTES + i * stride_;
	}

	int fd_ = -1;
	void* data_ = nullptr;
	size_t file_size_ = 0;
	uint64_t num_records_ = 0;
	uint32_t stride_ = 0;
	uint16_t spp_ = 0;
	uint16_t pipe_mask_ = 0;
	int n_pipes_ = 0;
};

PYBIND11_MODULE(_fastrx, m) {
	m.doc() = "AF_XDP fast packet capture for channel-stream data";

	m.attr("NUM_PIPELINES") = NUM_PIPELINES;
	m.attr("MAX_SAMPLES") = SAMPLES_PER_PIPELINE;
	m.attr("ABI_VERSION") = FASTRXD_ABI_VERSION;
	m.attr("MAX_CLIENTS") = FASTRXD_MAX_CLIENTS;

	PYBIND11_NUMPY_DTYPE(irigb_timestamp, y, d, h, m, s, ss, c, sbs);
	PYBIND11_NUMPY_DTYPE(fastrx_packet_header, magic, seq, pipe_snapshot,
			sample_trunc, module, version, tag, serial,
			samples_per_packet, ts);

	py::class_<PacketCapture>(m, "PacketCapture")
		.def(py::init([](std::string socket_path) {
				 auto c = std::make_unique<PacketCapture>(std::move(socket_path));
				 /* Threads start here, not in the constructor: on_packet() is
				  * virtual, and a constructor-started thread would dispatch to
				  * Consumer's version rather than PacketCapture's. */
				 c->start();
				 return c.release();
			 }),
			 "socket_path"_a)
		.def_property_readonly("socket_path", &PacketCapture::socket_path)
		.def("capture", &PacketCapture::capture,
			 "n_packets"_a, "pipe"_a = 1, "timeout"_a = 5.0,
			 "Collect exactly n_packets, or fewer on timeout.\n\n"
			 "Reusable: call as often as you like on one instance. The returned "
			 "'i' and 'q' arrays are strided views over one buffer whose "
			 "ownership passes to Python, so nothing is copied.")
		.def("stop", &PacketCapture::stop)
		.def_property_readonly("double_releases", &PacketCapture::double_releases)
		.def_property_readonly("stranded_frames", &PacketCapture::stranded_frames)
		.def_property_readonly("ring_drops", &PacketCapture::ring_drops)
		.def_property_readonly("client_id", &PacketCapture::client_id)
		.def("__enter__", [](PacketCapture& c) -> PacketCapture& { return c; },
			 py::return_value_policy::reference)
		.def("__exit__", [](PacketCapture& c, py::object, py::object, py::object) {
			c.stop();
		});

	py::class_<PacketWriter>(m, "PacketWriter")
		.def(py::init([](std::string socket_path, std::string path,
					uint8_t pipe_mask, uint64_t n_packets,
					uint64_t ring_bytes, unsigned queue_depth) {
				 auto w = std::make_unique<PacketWriter>(
						 std::move(socket_path), std::move(path),
						 pipe_mask, n_packets, ring_bytes, queue_depth);
				 w->start();
				 return w.release();
			 }),
			 "socket_path"_a, "path"_a, "pipe_mask"_a, "n_packets"_a = 0,
			 "ring_bytes"_a = (uint64_t)256 << 20, "queue_depth"_a = 32)
		.def("wait", [](PacketWriter& w, double timeout) {
				bool done;
				{
					py::gil_scoped_release rel;
					done = w.wait(timeout);
				}
				if (auto e = w.error())
					throw std::runtime_error(*e);
				return done;
			},
			"timeout"_a = -1.0,
			"Block until the n_packets limit is reached (returning True), "
			"the timeout expires (False), or the recording fails (raises). "
			"A negative timeout waits forever -- only meaningful with a "
			"limit set, since an unbounded recording never completes.")
		.def_property_readonly("complete", &PacketWriter::complete)
		.def("stop", [](PacketWriter& w) {
				{
					/* Draining the ring takes as long as the disk needs;
					 * nothing here requires the GIL. */
					py::gil_scoped_release rel;
					w.stop();
				}
				if (auto e = w.error())
					throw std::runtime_error(*e);
			},
			"Flush, finalize the file header, and close.\n\n"
			"Raises if the recording failed at any point (e.g. a disk "
			"error). Overruns are not failures; check .overruns.")
		.def_property_readonly("path", &PacketWriter::path)
		.def_property_readonly("pipe_mask", &PacketWriter::pipe_mask)
		.def_property_readonly("packets", &PacketWriter::packets)
		.def_property_readonly("bytes_written", &PacketWriter::bytes_written)
		.def_property_readonly("overruns", &PacketWriter::overruns)
		.def_property_readonly("dropouts", &PacketWriter::dropouts)
		.def_property_readonly("error", [](PacketWriter& w) { return w.error(); })
		.def_property_readonly("socket_path", &PacketWriter::socket_path)
		.def_property_readonly("double_releases", &PacketWriter::double_releases)
		.def_property_readonly("stranded_frames", &PacketWriter::stranded_frames)
		.def_property_readonly("ring_drops", &PacketWriter::ring_drops)
		.def_property_readonly("client_id", &PacketWriter::client_id)
		.def("__enter__", [](PacketWriter& w) -> PacketWriter& { return w; },
			 py::return_value_policy::reference)
		.def("__exit__", [](PacketWriter& w, py::object exc_type, py::object,
					py::object) {
			{
				py::gil_scoped_release rel;
				w.stop();
			}
			/* Report a failed recording -- but never mask an exception
			 * already unwinding through the with-block. */
			if (exc_type.is_none())
				if (auto e = w.error())
					throw std::runtime_error(*e);
			return false;
		});

	/* Directory only: the socket inside it is named for the interface fastrxd
	 * serves, so callers compose SOCKET_DIR + "/" + ifname. */
	m.attr("SOCKET_DIR") = FASTRXD_SOCKET_DIR;

	py::class_<PacketFile>(m, "PacketFile",
			"Zero-copy reader for PacketWriter recordings.\n\n"
			"Arrays returned by seq(), ts() and pipe_iq() are strided views "
			"over the file mapping, valid for the life of this object.")
		.def(py::init<const std::string&>(), "path"_a)
		.def_property_readonly("num_packets", &PacketFile::num_packets)
		.def_property_readonly("samples_per_pipe", &PacketFile::samples_per_pipe)
		.def_property_readonly("pipe_mask", &PacketFile::pipe_mask)
		.def_property_readonly("pipes", &PacketFile::pipes)
		.def_property_readonly("n_pipes", &PacketFile::n_pipes)
		.def_property_readonly("record_stride", &PacketFile::record_stride)
		.def("seq", &PacketFile::seq)
		.def("ts", &PacketFile::ts)
		.def("headers", &PacketFile::headers,
			 "Every record's wire header, as a structured array.\n\n"
			 "Stream metadata (serial, sample_trunc, module, pipe_snapshot, "
			 "...) is read from here, usually from element 0.")
		.def("pipe_iq", &PacketFile::pipe_iq, "pipe"_a)
		.def("__len__", &PacketFile::num_packets)
		.def("__enter__", [](PacketFile& f) -> PacketFile& { return f; },
			 py::return_value_policy::reference)
		.def("__exit__", [](PacketFile&, py::object, py::object, py::object) {});
}
