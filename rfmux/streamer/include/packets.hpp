#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <tuple>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <stdexcept>
#include <cstring>

#include <pybind11/pybind11.h>
#include <packet.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

namespace py = pybind11;

namespace packets {

	class PacketType;
	class Packet;
	class PacketQueue;
	class PacketReceiver;

	struct Timestamp {
		int32_t y, d, h, m, s;
		int32_t ss;
		uint32_t c, sbs;

		Timestamp() = default;
		Timestamp(const struct irigb_timestamp& ts)
		: y(ts.y), d(ts.d), h(ts.h), m(ts.m), s(ts.s),
		  ss(ts.ss), c(ts.c), sbs(ts.sbs) {}

		enum class Source { BACKPLANE = 0, TEST = 1, SMA = 2, GND = 3 };

		Source get_source() const {
			return static_cast<Source>((c >> 29) & 0x3);
		}

		void set_source(Source src) {
			c = (c & ~(0x3 << 29)) | (static_cast<uint32_t>(src) << 29);
		}

		bool is_recent() const { return c & 0x80000000; }
		void set_recent(bool recent) {
			if (recent)
				c |= 0x80000000;
			else
				c &= ~0x80000000;
		}

		Timestamp normalized() const;
		// Seconds of day; NaN when the stamp is not disciplined.
		double seconds_of_day() const;
		void renormalize() { *this = normalized(); }
	};

	/* Type-erased packet container. */
	class Packet {
	public:
		Packet(const PacketType* type, std::vector<char>&& data)
			: type_(type), data_(std::move(data)) {}

		const PacketType* type() const { return type_; }
		const void* data() const { return data_.data(); }
		void* data() { return data_.data(); }
		size_t size() const { return data_.size(); }

		// Convenience accessors (delegate to type)
		uint16_t serial() const;
		uint8_t module() const;
		uint32_t seq() const;
		Timestamp timestamp() const;
		py::object to_python() const;

		// For priority queue ordering by sequence number
		bool operator<(const Packet& other) const {
			return seq_ascends(seq(), other.seq());
		}

	private:
		static bool seq_ascends(uint32_t sa, uint32_t sb) {
			return static_cast<uint32_t>(sb - sa) < 0x80000000u;
		}

		const PacketType* type_;
		std::vector<char> data_;
	};

	class ReadoutPacket : public readout_packet_header {
	public:
		using Header = readout_packet_header;

		ReadoutPacket() = default;

		static ReadoutPacket from_bytes(const void* data, size_t len);
		py::bytes to_bytes() const;

		// Raw int32 I/Q pairs as stored on the wire
		const auto& raw_samples() const { return raw_samples_; }
		auto& raw_samples() { return raw_samples_; }

		const Timestamp& timestamp() const { return timestamp_; }
		Timestamp& timestamp() { return timestamp_; }

		int get_num_channels() const {
			return version == LONG_PACKET_VERSION
				   ? LONG_PACKET_CHANNELS
				   : SHORT_PACKET_CHANNELS;
		}

	private:
		std::vector<int32_t> raw_samples_;  // interleaved I/Q pairs
		Timestamp timestamp_;
	};

	class PFBPacket : public pfb_packet_header {
	public:
		using Header = pfb_packet_header;

		PFBPacket() = default;

		static PFBPacket from_bytes(const void* data, size_t len);
		py::bytes to_bytes() const;

		// Raw int32 I/Q pairs as stored on the wire
		const auto& raw_samples() const { return raw_samples_; }
		auto& raw_samples() { return raw_samples_; }

		const Timestamp& timestamp() const { return timestamp_; }
		Timestamp& timestamp() { return timestamp_; }

		int get_num_samples() const { return num_samples; }

	private:
		std::vector<int32_t> raw_samples_;  // interleaved I/Q pairs
		Timestamp timestamp_;
	};

	class PacketType {
	public:
		virtual ~PacketType() = default;

		virtual uint32_t magic() const = 0;
		virtual int port() const = 0;
		virtual size_t max_size() const = 0;

		virtual size_t packet_size(const void* data, size_t available) const = 0;
		virtual bool validate(const void* data, size_t len) const = 0;
		virtual uint16_t get_serial(const void* data) const = 0;
		virtual uint8_t get_module(const void* data) const = 0;
		virtual uint32_t get_seq(const void* data) const = 0;
		virtual Timestamp get_timestamp(const void* data, size_t len) const = 0;

		virtual py::object to_python(const void* data, size_t len) const {
			return py::bytes(static_cast<const char*>(data), len);
		}
	};

	class PacketQueue {
	public:
		PacketQueue(size_t max_size = 1024) : max_size_(max_size) {}

		std::optional<Packet> pop(std::optional<int> timeout_ms = std::nullopt);
		std::optional<Packet> try_pop();
		// Pop up to max_packets into *out* under one lock, stopping
		// at the first packet *accept* refuses (it is left queued).
		size_t pop_while(std::vector<Packet>& out, size_t max_packets,
		                 const std::function<bool(const Packet&)>& accept);
		void push(Packet&& packet);
		void clear();

		bool empty() const;
		size_t size() const;
		size_t max_size() const { return max_size_; }

		struct Stats {
			uint64_t packets_received = 0;
			// Queue overflow: the consumer could not keep up.
			uint64_t packets_dropped = 0;
			// Discontinuity EVENTS in the sequence numbers. One burst
			// of a thousand lost packets counts once, so this measures
			// how bursty the loss is, not how much of it there is.
			uint64_t sequence_gaps = 0;
			// Packets that never arrived, counted individually. This is
			// the one to compare against packets_received.
			uint64_t packets_missing = 0;
			uint32_t last_seq = 0;
		};

		Stats get_stats() const;
		void reset_stats();

	private:
		mutable std::mutex mutex_;
		std::condition_variable cv_;
		std::deque<Packet> queue_;
		const size_t max_size_;
		Stats stats_;
	};

	class PacketReceiver {
	public:
		PacketReceiver(std::shared_ptr<PacketType> type, py::object socket, size_t reorder_window = 256, size_t queue_max_size = 4096, size_t flush_threshold = 256);
		~PacketReceiver();

		PacketReceiver(const PacketReceiver&) = delete;
		PacketReceiver& operator=(const PacketReceiver&) = delete;

		size_t receive_batch(size_t batch_size = 256, std::optional<int> timeout_ms = std::nullopt);
		// Every held packet to its queue, ordered.  Packets received
		// afterwards are held again, so call once receive_batch has stopped.
		void flush_all();

		std::shared_ptr<PacketQueue> get_queue(uint16_t serial, uint8_t module);

		// Get all active queues as (serial, module, queue) tuples
		std::vector<std::tuple<uint16_t, uint8_t, std::shared_ptr<PacketQueue>>> get_all_queues();

		struct Stats {
			uint64_t total_packets_received = 0;
			uint64_t total_bytes_received = 0;
			uint64_t invalid_packets = 0;
			uint64_t wrong_magic = 0;
		};

		Stats get_stats() const;
		void reset_stats();

		const PacketType* type() const { return type_.get(); }
		int sockfd() const { return sockfd_; }

	private:
		void process_packet(std::vector<char>&& data);
		// Release all but *keep* of the packets held for (serial, module).
		void flush_reorder_buffer(uint16_t serial, uint8_t module, size_t keep);

		std::shared_ptr<PacketType> type_;
		int sockfd_;
		size_t reorder_window_;
		size_t queue_max_size_;
		size_t flush_threshold_;

		// recvmmsg scratch, reused across calls (why: see
		// receive_batch). Touched only by the thread that calls
		// receive_batch.
#ifdef __linux__
		std::vector<struct mmsghdr> rx_msgs_;
		std::vector<struct iovec> rx_iovecs_;
		std::vector<std::vector<char>> rx_buffers_;
		size_t rx_capacity_ = 0;
#endif

		using QueueKey = std::tuple<uint16_t, uint8_t>;

		struct PacketComparator {
			bool operator()(const Packet& a, const Packet& b) const {
				return b < a;
			}
		};

		std::map<QueueKey, std::priority_queue<Packet, std::vector<Packet>, PacketComparator>> reorder_buffers_;
		std::map<QueueKey, std::shared_ptr<PacketQueue>> queues_;
		mutable std::mutex queues_mutex_;

		mutable std::mutex stats_mutex_;
		Stats stats_;

		py::object socket_;
	};

	struct ReadoutPacketReceiver : PacketReceiver {
		ReadoutPacketReceiver(py::object socket, size_t reorder_window = 256, size_t queue_max_size = 4096, size_t flush_threshold = 256);
	};

	struct PFBPacketReceiver : PacketReceiver {
		PFBPacketReceiver(py::object socket, size_t reorder_window = 256, size_t queue_max_size = 4096, size_t flush_threshold = 256);
	};
}
