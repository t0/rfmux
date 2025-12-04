#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <fmt/format.h>

#include "packets.hpp"

namespace py = pybind11;
using namespace packets;
using namespace py::literals;

PYBIND11_MODULE(_packets, m) {
	py::class_<Timestamp>(m, "Timestamp")
		.def_readonly("y", &Timestamp::y, "Year (0-99)")
		.def_readonly("d", &Timestamp::d, "Day of year (1-366)")
		.def_readonly("h", &Timestamp::h, "Hour (0-23)")
		.def_readonly("m", &Timestamp::m, "Minute (0-59)")
		.def_readonly("s", &Timestamp::s, "Second (0-59)")
		.def_readonly("ss", &Timestamp::ss, "Sub-seconds")
		.def_readonly("c", &Timestamp::c, "Control field")
		.def_readonly("sbs", &Timestamp::sbs, "Sub-block sequence")
		.def("get_source", &Timestamp::get_source, "Get timestamp source")
		.def("is_recent", &Timestamp::is_recent, "Check if timestamp is recent")
		.def("get_count", &Timestamp::get_count, "Get count field")
		.def("__repr__", [](const Timestamp& ts) {
			return fmt::format("Timestamp(y={} d={} {}:{}:{} ss={} c={} sbs={} recent={})",
					ts.y, ts.d,
					ts.h, ts.m, ts.s,
					ts.ss, ts.c, ts.sbs,
					ts.is_recent() ? "Y" : "N");
		});

	py::enum_<Timestamp::Source>(m, "TimestampSource")
		.value("BACKPLANE", Timestamp::Source::BACKPLANE)
		.value("TEST", Timestamp::Source::TEST)
		.value("SMA", Timestamp::Source::SMA)
		.value("GND", Timestamp::Source::GND);

	py::class_<ReadoutPacket>(m, "ReadoutPacket")
		.def_static("from_bytes", &ReadoutPacket::from_bytes, "data"_a, "len"_a)
		.def_readonly("magic", &ReadoutPacket::magic)
		.def_readonly("version", &ReadoutPacket::version)
		.def_readonly("serial", &ReadoutPacket::serial)
		.def_readonly("num_modules", &ReadoutPacket::num_modules)
		.def_readonly("flags", &ReadoutPacket::flags)
		.def_readonly("fir_stage", &ReadoutPacket::fir_stage)
		.def_readonly("module", &ReadoutPacket::module)
		.def_readonly("seq", &ReadoutPacket::seq)

		.def_property_readonly("samples", &ReadoutPacket::samples)
		.def_property_readonly("timestamp", &ReadoutPacket::timestamp)
		.def("get_num_channels", &ReadoutPacket::get_num_channels)
		.def("get_channel", &ReadoutPacket::get_channel, "ch"_a)
		.def("__repr__", [](const ReadoutPacket& pkt) {
			return fmt::format("ReadoutPacket(serial={} module={} seq={})",
					pkt.serial, pkt.module, pkt.seq);
		});

	py::class_<PFBPacket>(m, "PFBPacket")
		.def_static("from_bytes", &PFBPacket::from_bytes, "data"_a, "len"_a)
		.def_readonly("magic", &PFBPacket::magic)
		.def_readonly("version", &PFBPacket::version)
		.def_readonly("mode", &PFBPacket::mode)
		.def_readonly("serial", &PFBPacket::serial)
		.def_readonly("slot1", &PFBPacket::slot1)
		.def_readonly("slot2", &PFBPacket::slot2)
		.def_readonly("slot3", &PFBPacket::slot3)
		.def_readonly("slot4", &PFBPacket::slot4)
		.def_readonly("num_samples", &PFBPacket::num_samples)
		.def_readonly("module", &PFBPacket::module)
		.def_readonly("seq", &PFBPacket::seq)
		// Other packet data
		.def_property_readonly("samples", &PFBPacket::samples)
		.def_property_readonly("timestamp", &PFBPacket::timestamp)
		.def("get_num_samples", &PFBPacket::get_num_samples)
		.def("get_sample", &PFBPacket::get_sample, "idx"_a)
		.def("__repr__", [](const PFBPacket& pkt) {
			return fmt::format("PFBPacket(serial={} module={} seq={} samples={})",
					pkt.serial, pkt.module, pkt.seq, pkt.num_samples);
		});

	py::class_<Packet>(m, "Packet")
		.def("serial", &Packet::serial)
		.def("module", &Packet::module)
		.def("seq", &Packet::seq)
		.def("timestamp", &Packet::timestamp)
		.def("to_python", &Packet::to_python,
			 "Convert to typed Python object (ReadoutPacket or PFBPacket)")
		.def("size", &Packet::size)
		.def("__repr__", [](const Packet& pkt) {
			return fmt::format("Packet(serial={} module={} seq={})",
					pkt.serial(), pkt.module(), pkt.seq());
		});

	py::class_<PacketQueue::Stats>(m, "PacketQueueStats")
		.def_readonly("packets_received", &PacketQueue::Stats::packets_received)
		.def_readonly("packets_dropped", &PacketQueue::Stats::packets_dropped)
		.def_readonly("sequence_gaps", &PacketQueue::Stats::sequence_gaps)
		.def_readonly("last_seq", &PacketQueue::Stats::last_seq);

	py::class_<PacketQueue, std::shared_ptr<PacketQueue>>(m, "PacketQueue")
		.def("pop", [](PacketQueue& self, std::optional<int> timeout_ms) {
			py::gil_scoped_release release; // Release GIL while waiting
			return self.pop(timeout_ms);
		}, "timeout_ms"_a = py::none(),
		   "Pop packet from queue (blocks, releases GIL)")
		.def("try_pop", &PacketQueue::try_pop, "Try to pop packet (non-blocking)")
		.def("empty", &PacketQueue::empty)
		.def("size", &PacketQueue::size)
		.def("max_size", &PacketQueue::max_size)
		.def("get_stats", &PacketQueue::get_stats)
		.def("reset_stats", &PacketQueue::reset_stats);

	py::class_<PacketReceiver::Stats>(m, "PacketReceiverStats")
		.def_readonly("total_packets_received", &PacketReceiver::Stats::total_packets_received)
		.def_readonly("total_bytes_received", &PacketReceiver::Stats::total_bytes_received)
		.def_readonly("invalid_packets", &PacketReceiver::Stats::invalid_packets)
		.def_readonly("wrong_magic", &PacketReceiver::Stats::wrong_magic);

	py::class_<PacketReceiver>(m, "PacketReceiver",
		"Base class for packet receivers (use ReadoutPacketReceiver or PFBPacketReceiver instead).")
		.def("receive_batch", [](PacketReceiver& self, size_t batch_size,
								std::optional<int> timeout_ms) {
			py::gil_scoped_release release;
			return self.receive_batch(batch_size, timeout_ms);
		}, "batch_size"_a = 256,
		   "timeout_ms"_a = py::none(),
		   "Receive and process batch of packets (releases GIL)")
		.def("get_queue", &PacketReceiver::get_queue,
			 "serial"_a, "module"_a,
			 "Get queue for specific module")
		.def("get_all_queues", &PacketReceiver::get_all_queues,
			 "Get all active queues as (serial, module, queue) tuples")
		.def("get_stats", &PacketReceiver::get_stats)
		.def("reset_stats", &PacketReceiver::reset_stats)
		.def_property_readonly("sockfd", &PacketReceiver::sockfd)
		.def_property_readonly("type", &PacketReceiver::type, py::return_value_policy::reference);

	py::class_<ReadoutPacketReceiver, PacketReceiver>(m, "ReadoutPacketReceiver",
		"Packet receiver for readout packets.")
		.def(py::init<py::object, size_t>(),
			 "socket"_a,
			 "reorder_window"_a = 256,
			 "Create readout packet receiver");

	py::class_<PFBPacketReceiver, PacketReceiver>(m, "PFBPacketReceiver",
		"Packet receiver for PFB packets.")
		.def(py::init<py::object, size_t>(),
			 "socket"_a,
			 "reorder_window"_a = 256,
			 "Create PFB packet receiver");

	/* Cross-platform wrapper for IP_ADD_SOURCE_MEMBERSHIP structure.
	 * Linux/macOS/Windows all have struct ip_mreq_source with the same fields
	 * but in different orders - this provides a consistent constructor. */
	struct IpMreqSource : ip_mreq_source {
		IpMreqSource() = delete;
		explicit IpMreqSource(const std::string& multiaddr,
					 const std::string& sourceaddr,
					 const std::string& interface_addr) {
			inet_pton(AF_INET, multiaddr.c_str(), &imr_multiaddr);
			inet_pton(AF_INET, sourceaddr.c_str(), &imr_sourceaddr);
			inet_pton(AF_INET, interface_addr.c_str(), &imr_interface);
		}

		py::bytes to_bytes() const {
			return py::bytes(reinterpret_cast<const char*>(this), sizeof(*this));
		}
	};
	py::class_<IpMreqSource>(m, "ip_mreq_source",
		"Cross-platform IP_ADD_SOURCE_MEMBERSHIP structure.")
		.def(py::init<const std::string&, const std::string&, const std::string&>(),
			py::kw_only(), "multiaddr"_a, "sourceaddr"_a, "interface"_a)
		.def("to_bytes", &IpMreqSource::to_bytes)
		.def_property_readonly_static("IP_ADD_SOURCE_MEMBERSHIP", [](py::object) {
			return IP_ADD_SOURCE_MEMBERSHIP; }
		);

	m.attr("MULTICAST_GROUP") = "239.192.0.2";
	m.attr("READOUT_PACKET_MAGIC") = READOUT_PACKET_MAGIC;
	m.attr("PFB_PACKET_MAGIC") = PFB_PACKET_MAGIC;
	m.attr("STREAMER_PORT") = STREAMER_PORT;
	m.attr("PFB_STREAMER_PORT") = PFB_STREAMER_PORT;
	m.attr("LONG_PACKET_SIZE") = LONG_PACKET_SIZE;
	m.attr("SHORT_PACKET_SIZE") = SHORT_PACKET_SIZE;
	m.attr("LONG_PACKET_CHANNELS") = LONG_PACKET_CHANNELS;
	m.attr("SHORT_PACKET_CHANNELS") = SHORT_PACKET_CHANNELS;
}
