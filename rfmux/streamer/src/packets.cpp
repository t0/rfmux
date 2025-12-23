#include "packets.hpp"

#include <iostream>
#include <chrono>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <sys/select.h>
#include <unistd.h>
#endif

namespace packets {
	uint16_t Packet::serial() const {
		return type_->get_serial(data());
	}

	uint8_t Packet::module() const {
		return type_->get_module(data());
	}

	uint32_t Packet::seq() const {
		return type_->get_seq(data());
	}

	Timestamp Packet::timestamp() const {
		return type_->get_timestamp(data(), size());
	}

	py::object Packet::to_python() const {
		return type_->to_python(data(), size());
	}

	Timestamp Timestamp::normalized() const {
		Timestamp result = *this;
		if (!result.is_recent())
			return result;

		// Carry subseconds -> seconds
		int32_t carry = result.ss / SS_PER_SECOND;
		result.ss = result.ss % SS_PER_SECOND;
		if (result.ss < 0) {
			result.ss += SS_PER_SECOND;
			carry--;
		}
		result.s += carry;

		// Carry seconds -> minutes
		carry = result.s / 60;
		result.s = result.s % 60;
		if (result.s < 0) {
			result.s += 60;
			carry--;
		}
		result.m += carry;

		// Carry minutes -> hours
		carry = result.m / 60;
		result.m = result.m % 60;
		if (result.m < 0) {
			result.m += 60;
			carry--;
		}
		result.h += carry;

		// Carry hours -> days
		carry = result.h / 24;
		result.h = result.h % 24;
		if (result.h < 0) {
			result.h += 24;
			carry--;
		}
		result.d += carry;

		// Carry days -> years (ignoring leap years)
		carry = (result.d - 1) / 365;
		result.d = ((result.d - 1) % 365) + 1;
		if (result.d < 1) {
			result.d += 365;
			carry--;
		}
		result.y = (result.y + carry) % 100;
		if (result.y < 0)
			result.y += 100;

		return result;
	}

	ReadoutPacket ReadoutPacket::from_bytes(const void* data, size_t len) {
		if (len != LONG_PACKET_SIZE && len != SHORT_PACKET_SIZE)
			throw std::runtime_error("Invalid readout packet size");

		ReadoutPacket pkt;

		// Copy header fields directly into base class
		static_cast<readout_packet_header&>(pkt) = *static_cast<const readout_packet_header*>(data);
		if (pkt.magic != READOUT_PACKET_MAGIC)
			throw std::runtime_error("Invalid readout packet magic");

		// Determine number of channels
		int num_channels;
		if (pkt.version == LONG_PACKET_VERSION)
			num_channels = LONG_PACKET_CHANNELS;
		else if (pkt.version == SHORT_PACKET_VERSION)
			num_channels = SHORT_PACKET_CHANNELS;
		else
			throw std::runtime_error("Invalid readout packet version");

		// Parse samples (I/Q pairs)
		const auto* samples_ptr = reinterpret_cast<const int32_t*>(
			static_cast<const char*>(data) + sizeof(readout_packet_header)
		);

		pkt.samples_.resize(num_channels);
		for (int i = 0; i < num_channels; i++)
			pkt.samples_[i] = std::complex<double>(
					samples_ptr[2*i],
					samples_ptr[2*i+1]) / 256.;

		// Parse timestamp
		const auto* ts_ptr = reinterpret_cast<const irigb_timestamp*>(
			static_cast<const char*>(data) + sizeof(readout_packet_header) +
			num_channels * 2 * sizeof(int32_t)
		);
		pkt.timestamp_ = Timestamp(*ts_ptr);

		return pkt;
	}

	py::bytes ReadoutPacket::to_bytes() const {
		int num_channels = get_num_channels();
		size_t packet_size = (version == LONG_PACKET_VERSION) ? LONG_PACKET_SIZE : SHORT_PACKET_SIZE;

		std::vector<char> buffer(packet_size);

		// Write header
		std::memcpy(buffer.data(), static_cast<const readout_packet_header*>(this), sizeof(readout_packet_header));

		// Write samples (convert double back to int32)
		auto* samples_ptr = reinterpret_cast<int32_t*>(buffer.data() + sizeof(readout_packet_header));
		for (int i = 0; i < num_channels; i++) {
			std::complex<double> sample = (i < (int)samples_.size()) ? samples_[i] : std::complex<double>(0, 0);
			samples_ptr[2 * i] = static_cast<int32_t>(sample.real() * 256.);
			samples_ptr[2 * i + 1] = static_cast<int32_t>(sample.imag() * 256.);
		}

		// Write timestamp
		irigb_timestamp ts_out;
		ts_out.y = timestamp_.y;
		ts_out.d = timestamp_.d;
		ts_out.h = timestamp_.h;
		ts_out.m = timestamp_.m;
		ts_out.s = timestamp_.s;
		ts_out.ss = timestamp_.ss;
		ts_out.c = timestamp_.c;
		ts_out.sbs = timestamp_.sbs;

		std::memcpy(
			buffer.data() + sizeof(readout_packet_header) + num_channels * 2 * sizeof(int32_t),
			&ts_out,
			sizeof(irigb_timestamp)
		);

		return py::bytes(buffer.data(), buffer.size());
	}

	PFBPacket PFBPacket::from_bytes(const void* data, size_t len) {
		if (len < sizeof(pfb_packet_header) + sizeof(irigb_timestamp))
			throw std::runtime_error("PFB packet too small");

		PFBPacket pkt;

		// Copy header fields directly into base class
		static_cast<pfb_packet_header&>(pkt) = *static_cast<const pfb_packet_header*>(data);
		if (pkt.magic != PFB_PACKET_MAGIC)
			throw std::runtime_error("Invalid PFB packet magic");

		if (pkt.num_samples > PFBPACKET_NSAMP_MAX)
			throw std::runtime_error("PFB sample count exceeds maximum");

		// Calculate expected size
		size_t expected_size = sizeof(pfb_packet_header) +
				  (pkt.num_samples * 2 * sizeof(int32_t)) +
				  sizeof(irigb_timestamp);
		if (len != expected_size)
			throw std::runtime_error("PFB packet size mismatch");

		// Parse samples (I/Q pairs)
		const auto* samples_ptr = reinterpret_cast<const int32_t*>(
			static_cast<const char*>(data) + sizeof(pfb_packet_header)
		);

		pkt.samples_.reserve(pkt.num_samples);
		for (int i = 0; i < pkt.num_samples; i++) {
			double i_val = samples_ptr[2 * i] / 256.;
			double q_val = samples_ptr[2 * i + 1] / 256.;
			pkt.samples_.emplace_back(i_val, q_val);
		}

		// Parse timestamp
		const auto* ts_ptr = reinterpret_cast<const irigb_timestamp*>(
			static_cast<const char*>(data) + sizeof(pfb_packet_header) +
			pkt.num_samples * 2 * sizeof(int32_t)
		);
		pkt.timestamp_ = Timestamp(*ts_ptr);

		return pkt;
	}

	py::bytes PFBPacket::to_bytes() const {
		size_t packet_size = sizeof(pfb_packet_header) +
							 (num_samples * 2 * sizeof(int32_t)) +
							 sizeof(irigb_timestamp);

		std::vector<char> buffer(packet_size);

		// Write header
		std::memcpy(buffer.data(), static_cast<const pfb_packet_header*>(this), sizeof(pfb_packet_header));

		// Write samples (convert double back to int32)
		auto* samples_ptr = reinterpret_cast<int32_t*>(buffer.data() + sizeof(pfb_packet_header));
		for (int i = 0; i < num_samples; i++) {
			std::complex<double> sample = (i < (int)samples_.size()) ? samples_[i] : std::complex<double>(0, 0);
			samples_ptr[2 * i] = static_cast<int32_t>(sample.real() * 256.);
			samples_ptr[2 * i + 1] = static_cast<int32_t>(sample.imag() * 256.);
		}

		// Write timestamp
		irigb_timestamp ts_out;
		ts_out.y = timestamp_.y;
		ts_out.d = timestamp_.d;
		ts_out.h = timestamp_.h;
		ts_out.m = timestamp_.m;
		ts_out.s = timestamp_.s;
		ts_out.ss = timestamp_.ss;
		ts_out.c = timestamp_.c;
		ts_out.sbs = timestamp_.sbs;

		std::memcpy(
			buffer.data() + sizeof(pfb_packet_header) + num_samples * 2 * sizeof(int32_t),
			&ts_out,
			sizeof(irigb_timestamp)
		);

		return py::bytes(buffer.data(), buffer.size());
	}

	std::optional<Packet> PacketQueue::pop(std::optional<int> timeout_ms) {
		std::unique_lock<std::mutex> lock(mutex_);

		if (timeout_ms.has_value()) {
			auto timeout = std::chrono::milliseconds(*timeout_ms);
			if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); }))
				return std::nullopt;  // Timeout
		} else
			cv_.wait(lock, [this] { return !queue_.empty(); });

		Packet packet = std::move(queue_.front());
		queue_.pop_front();

		return packet;
	}

	std::optional<Packet> PacketQueue::try_pop() {
		std::lock_guard<std::mutex> lock(mutex_);

		if (queue_.empty())
			return std::nullopt;

		Packet packet = std::move(queue_.front());
		queue_.pop_front();

		return packet;
	}

	void PacketQueue::push(Packet&& packet) {
		std::lock_guard<std::mutex> lock(mutex_);

		// Check for sequence gaps
		uint32_t seq = packet.seq();
		if (stats_.packets_received > 0 && stats_.last_seq != 0) {
			uint32_t expected_seq = stats_.last_seq + 1;
			if (seq != expected_seq)
				stats_.sequence_gaps++;
		}
		stats_.last_seq = seq;
		stats_.packets_received++;

		// Check if queue is full
		if (queue_.size() >= max_size_) {
			queue_.pop_front();
			stats_.packets_dropped++;
		}

		queue_.push_back(std::move(packet));
		cv_.notify_one();
	}

	bool PacketQueue::empty() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return queue_.empty();
	}

	size_t PacketQueue::size() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return queue_.size();
	}

	PacketQueue::Stats PacketQueue::get_stats() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return stats_;
	}

	void PacketQueue::reset_stats() {
		std::lock_guard<std::mutex> lock(mutex_);
		stats_ = Stats{};
	}

	PacketReceiver::PacketReceiver(std::shared_ptr<PacketType> type, py::object socket, size_t reorder_window)
			: type_(type), reorder_window_(reorder_window), socket_(socket) {
		// Extract file descriptor from Python socket
		sockfd_ = socket.attr("fileno")().cast<int>();
	}

	PacketReceiver::~PacketReceiver() { }

	size_t PacketReceiver::receive_batch(size_t batch_size, std::optional<int> timeout_ms) {
		const size_t MAX_PACKET_SIZE = type_->max_size();
		size_t packets_received = 0;
#ifdef __linux__
		// Linux: use recvmmsg for batch reception
		std::vector<struct mmsghdr> msgs(batch_size);
		std::vector<struct iovec> iovecs(batch_size);
		std::vector<std::vector<char>> buffers(batch_size);

		// Setup buffers
		for (size_t i = 0; i < batch_size; i++) {
			buffers[i].resize(MAX_PACKET_SIZE);
			iovecs[i].iov_base = buffers[i].data();
			iovecs[i].iov_len = MAX_PACKET_SIZE;
			msgs[i].msg_hdr.msg_iov = &iovecs[i];
			msgs[i].msg_hdr.msg_iovlen = 1;
			msgs[i].msg_hdr.msg_name = nullptr;
			msgs[i].msg_hdr.msg_namelen = 0;
			msgs[i].msg_hdr.msg_control = nullptr;
			msgs[i].msg_hdr.msg_controllen = 0;
			msgs[i].msg_hdr.msg_flags = 0;
		}

		// Receive batch
		struct timespec *timeout_ptr = nullptr;
		struct timespec timeout_spec;
		if (timeout_ms.has_value()) {
			timeout_spec.tv_sec = *timeout_ms / 1000;
			timeout_spec.tv_nsec = (*timeout_ms % 1000) * 1000000;
			timeout_ptr = &timeout_spec;
		}

		int n = recvmmsg(sockfd_, msgs.data(), batch_size, MSG_WAITFORONE, timeout_ptr);
		if (n < 0) {
			if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
				return 0;  // Timeout or interrupted
			throw std::runtime_error(std::string("recvmmsg failed: ") + strerror(errno));
		}

		packets_received = n;

		// Process received packets
		for (int i = 0; i < n; i++) {
			size_t len = msgs[i].msg_len;

			// Validate
			if (!type_->validate(buffers[i].data(), len)) {
				std::lock_guard<std::mutex> lock(stats_mutex_);
				stats_.invalid_packets++;
				continue;
			}

			// Update stats
			{
				std::lock_guard<std::mutex> lock(stats_mutex_);
				stats_.total_packets_received++;
				stats_.total_bytes_received += len;
			}

			// Copy buffer and process
			std::vector<char> data(buffers[i].begin(), buffers[i].begin() + len);
			process_packet(std::move(data));
		}
#else
		// Non-Linux: use select + recv
		std::vector<char> buffer(MAX_PACKET_SIZE);

		for (size_t i = 0; i < batch_size; i++) {
			// Setup select for timeout
			fd_set readfds;
			FD_ZERO(&readfds);
			FD_SET(sockfd_, &readfds);

			struct timeval *timeout_ptr = nullptr;
			struct timeval timeout_val;
			if (timeout_ms.has_value()) {
				timeout_val.tv_sec = *timeout_ms / 1000;
				timeout_val.tv_usec = (*timeout_ms % 1000) * 1000;
				timeout_ptr = &timeout_val;
			}

			int ready = select(sockfd_ + 1, &readfds, nullptr, nullptr, timeout_ptr);
			if (ready <= 0)
				break;  // Timeout or error

			int n = recv(sockfd_, buffer.data(), MAX_PACKET_SIZE, 0);
			if (n < 0) {
#ifdef _WIN32
				int err = WSAGetLastError();
				if (err == WSAEWOULDBLOCK)
					break;  // No more packets
				throw std::runtime_error("recv failed: " + std::to_string(err));
#else
				if (errno == EAGAIN || errno == EWOULDBLOCK)
					break;  // No more packets
				throw std::runtime_error(std::string("recv failed: ") + strerror(errno));
#endif
			}

			if (n == 0)
				break;  // No data

			size_t len = static_cast<size_t>(n);

			// Validate
			if (!type_->validate(buffer.data(), len)) {
				std::lock_guard<std::mutex> lock(stats_mutex_);
				stats_.invalid_packets++;
				continue;
			}

			// Update stats
			{
				std::lock_guard<std::mutex> lock(stats_mutex_);
				stats_.total_packets_received++;
				stats_.total_bytes_received += len;
			}

			packets_received++;

			// Copy and process
			std::vector<char> data(buffer.begin(), buffer.begin() + len);
			process_packet(std::move(data));
		}
#endif
		return packets_received;
	}

	std::shared_ptr<PacketQueue> PacketReceiver::get_queue(uint16_t serial, uint8_t module) {
		auto key = std::make_tuple(serial, module);
		std::lock_guard<std::mutex> lock(queues_mutex_);

		if (queues_.find(key) == queues_.end())
			queues_[key] = std::make_shared<PacketQueue>();

		return queues_[key];
	}

	std::vector<std::tuple<uint16_t, uint8_t, std::shared_ptr<PacketQueue>>>
	PacketReceiver::get_all_queues() {
		std::lock_guard<std::mutex> lock(queues_mutex_);
		std::vector<std::tuple<uint16_t, uint8_t, std::shared_ptr<PacketQueue>>> result;

		for (const auto& [key, queue] : queues_) {
			const auto& [serial, module] = key;
			result.push_back({serial, module, queue});
		}

		return result;
	}

	PacketReceiver::Stats PacketReceiver::get_stats() const {
		std::lock_guard<std::mutex> lock(stats_mutex_);
		return stats_;
	}

	void PacketReceiver::reset_stats() {
		std::lock_guard<std::mutex> lock(stats_mutex_);
		stats_ = Stats{};
	}

	void PacketReceiver::process_packet(std::vector<char>&& data) {
		Packet packet(type_.get(), std::move(data));

		uint16_t serial = packet.serial();
		uint8_t module = packet.module();
		auto key = std::make_tuple(serial, module);

		std::lock_guard<std::mutex> lock(queues_mutex_);

		reorder_buffers_[key].push(std::move(packet));

		if (reorder_buffers_[key].size() >= reorder_window_)
			flush_reorder_buffer(serial, module);
	}

	void PacketReceiver::flush_reorder_buffer(uint16_t serial, uint8_t module) {
		auto key = std::make_tuple(serial, module);

		if (queues_.find(key) == queues_.end())
			queues_[key] = std::make_shared<PacketQueue>();

		auto& reorder_buf = reorder_buffers_[key];
		auto& out_queue = *queues_[key];

		size_t to_pop = reorder_window_ / 2;
		while (to_pop > 0 && !reorder_buf.empty()) {
			// Can't move from priority_queue::top() because it's const
			// Must copy, then pop
			Packet pkt = reorder_buf.top();
			reorder_buf.pop();
			out_queue.push(std::move(pkt));
			to_pop--;
		}
	}

	// Concrete packet type implementations
	class ReadoutPacketTypeImpl : public PacketType {
	public:
		uint32_t magic() const override { return READOUT_PACKET_MAGIC; }
		int port() const override { return STREAMER_PORT; }
		size_t max_size() const override { return LONG_PACKET_SIZE; }

		size_t packet_size(const void* data, size_t available) const override {
			if (available < sizeof(readout_packet_header))
				return 0;

			const auto* hdr = static_cast<const readout_packet_header*>(data);

			switch (hdr->version) {
				case LONG_PACKET_VERSION: return LONG_PACKET_SIZE;
				case SHORT_PACKET_VERSION: return SHORT_PACKET_SIZE;
				default: throw std::runtime_error("Invalid readout packet version");
			}
		}

		bool validate(const void* data, size_t len) const override {
			if (len < sizeof(readout_packet_header))
				return false;

			const auto* hdr = static_cast<const readout_packet_header*>(data);

			if (hdr->magic != READOUT_PACKET_MAGIC)
				return false;

			try {
				size_t expected = packet_size(data, len);
				if (len != expected)
					return false;
			} catch (...) {
				return false;
			}

			return true;
		}

		uint16_t get_serial(const void* data) const override {
			return static_cast<const readout_packet_header*>(data)->serial;
		}

		uint8_t get_module(const void* data) const override {
			return static_cast<const readout_packet_header*>(data)->module;
		}

		uint32_t get_seq(const void* data) const override {
			return static_cast<const readout_packet_header*>(data)->seq;
		}

		Timestamp get_timestamp(const void* data, size_t) const override {
			const auto* hdr = static_cast<const readout_packet_header*>(data);

			int num_channels = (hdr->version == LONG_PACKET_VERSION)
							  ? LONG_PACKET_CHANNELS
							  : SHORT_PACKET_CHANNELS;

			size_t offset = sizeof(readout_packet_header) +
						   num_channels * 2 * sizeof(int32_t);

			const auto* ts = reinterpret_cast<const irigb_timestamp*>(
				static_cast<const char*>(data) + offset
			);

			return Timestamp(*ts);
		}

		py::object to_python(const void* data, size_t len) const override {
			return py::cast(ReadoutPacket::from_bytes(data, len));
		}
	};

	class PFBPacketTypeImpl : public PacketType {
	public:
		uint32_t magic() const override { return PFB_PACKET_MAGIC; }
		int port() const override { return PFB_STREAMER_PORT; }
		size_t max_size() const override { return sizeof(pfb_packet_buffer); }

		size_t packet_size(const void* data, size_t available) const override {
			if (available < sizeof(pfb_packet_header))
				return 0;

			const auto* hdr = static_cast<const pfb_packet_header*>(data);

			if (hdr->num_samples > PFBPACKET_NSAMP_MAX)
				throw std::runtime_error("PFB sample count exceeds maximum");

			return sizeof(pfb_packet_header) +
				   (hdr->num_samples * 2 * sizeof(int32_t)) +
				   sizeof(irigb_timestamp);
		}

		bool validate(const void* data, size_t len) const override {
			if (len < sizeof(pfb_packet_header))
				return false;

			const auto* hdr = static_cast<const pfb_packet_header*>(data);

			if (hdr->magic != PFB_PACKET_MAGIC)
				return false;

			if (hdr->num_samples > PFBPACKET_NSAMP_MAX)
				return false;

			try {
				size_t expected = packet_size(data, len);
				if (len != expected)
					return false;
			} catch (...) {
				return false;
			}

			return true;
		}

		uint16_t get_serial(const void* data) const override {
			return static_cast<const pfb_packet_header*>(data)->serial;
		}

		uint8_t get_module(const void* data) const override {
			return static_cast<const pfb_packet_header*>(data)->module;
		}

		uint32_t get_seq(const void* data) const override {
			return static_cast<const pfb_packet_header*>(data)->seq;
		}

		Timestamp get_timestamp(const void* data, size_t) const override {
			const auto* hdr = static_cast<const pfb_packet_header*>(data);

			size_t offset = sizeof(pfb_packet_header) +
						   hdr->num_samples * 2 * sizeof(int32_t);

			const auto* ts = reinterpret_cast<const irigb_timestamp*>(
				static_cast<const char*>(data) + offset
			);

			return Timestamp(*ts);
		}

		py::object to_python(const void* data, size_t len) const override {
			return py::cast(PFBPacket::from_bytes(data, len));
		}
	};

	ReadoutPacketReceiver::ReadoutPacketReceiver(py::object socket, size_t reorder_window)
		: PacketReceiver(std::make_shared<ReadoutPacketTypeImpl>(),
				socket, reorder_window) {}

	PFBPacketReceiver::PFBPacketReceiver(py::object socket, size_t reorder_window)
		: PacketReceiver(std::make_shared<PFBPacketTypeImpl>(), socket, reorder_window) {}
}
