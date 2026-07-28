#include "tuber_support.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <fmt/format.h>

#include "packets.hpp"

namespace py = pybind11;
using namespace packets;
using namespace py::literals;

PYBIND11_MODULE(_receiver, m) {
	py::class_<Timestamp>(m, "Timestamp")
		.def(py::init<>())
		.def(py::init([](int32_t y, int32_t d, int32_t h, int32_t m, int32_t s,
		                 int32_t ss, uint32_t c, uint32_t sbs,
		                 Timestamp::Source source, bool recent) {
			Timestamp ts;
			ts.y = y;
			ts.d = d;
			ts.h = h;
			ts.m = m;
			ts.s = s;
			ts.ss = ss;
			ts.c = c;
			ts.sbs = sbs;
			ts.set_source(source);
			ts.set_recent(recent);
			return ts;
		}),
		py::kw_only(),
		"y"_a, "d"_a, "h"_a, "m"_a, "s"_a, "ss"_a, "c"_a, "sbs"_a,
		"source"_a, "recent"_a,
		"Create timestamp with specified fields")
		.def_readwrite("y", &Timestamp::y, "Year (0-99)")
		.def_readwrite("d", &Timestamp::d, "Day of year (1-366)")
		.def_readwrite("h", &Timestamp::h, "Hour (0-23)")
		.def_readwrite("m", &Timestamp::m, "Minute (0-59)")
		.def_readwrite("s", &Timestamp::s, "Second (0-59)")
		.def_readwrite("ss", &Timestamp::ss, "Sub-seconds")
		.def_property("c",
			[](const Timestamp& ts) { return ts.c & 0xfff; },
			[](Timestamp& ts, uint32_t control) {
				ts.c = (ts.c & 0xfffff000) | (control & 0xfff);
			},
			"Control field (12-bit, preserves source and recent bits)")
		.def_readwrite("sbs", &Timestamp::sbs, "Sub-block sequence")
		.def_property("source", &Timestamp::get_source, &Timestamp::set_source, "Timestamp source (read/write)")
		.def_property("recent", &Timestamp::is_recent, &Timestamp::set_recent, "Recent flag (read/write)")
		.def("renormalize", &Timestamp::renormalize, "Normalize timestamp fields (carry overflow)")
		.def("__lt__", [](const Timestamp& a, const Timestamp& b) {
			auto an = a.normalized(), bn = b.normalized();
			return std::tie(an.y, an.d, an.h, an.m, an.s, an.ss, an.c) < std::tie(bn.y, bn.d, bn.h, bn.m, bn.s, bn.ss, bn.c);
		})
		.def("__le__", [](const Timestamp& a, const Timestamp& b) {
			auto an = a.normalized(), bn = b.normalized();
			return std::tie(an.y, an.d, an.h, an.m, an.s, an.ss, an.c) <= std::tie(bn.y, bn.d, bn.h, bn.m, bn.s, bn.ss, bn.c);
		})
		.def("__gt__", [](const Timestamp& a, const Timestamp& b) {
			auto an = a.normalized(), bn = b.normalized();
			return std::tie(an.y, an.d, an.h, an.m, an.s, an.ss, an.c) > std::tie(bn.y, bn.d, bn.h, bn.m, bn.s, bn.ss, bn.c);
		})
		.def("__ge__", [](const Timestamp& a, const Timestamp& b) {
			auto an = a.normalized(), bn = b.normalized();
			return std::tie(an.y, an.d, an.h, an.m, an.s, an.ss, an.c) >= std::tie(bn.y, bn.d, bn.h, bn.m, bn.s, bn.ss, bn.c);
		})
		.def("__eq__", [](const Timestamp& a, const Timestamp& b) {
			auto an = a.normalized(), bn = b.normalized();
			return std::tie(an.y, an.d, an.h, an.m, an.s, an.ss, an.c) == std::tie(bn.y, bn.d, bn.h, bn.m, bn.s, bn.ss, bn.c);
		})
		.def("__ne__", [](const Timestamp& a, const Timestamp& b) {
			auto an = a.normalized(), bn = b.normalized();
			return std::tie(an.y, an.d, an.h, an.m, an.s, an.ss, an.c) != std::tie(bn.y, bn.d, bn.h, bn.m, bn.s, bn.ss, bn.c);
		})
		.def("keys", [](const Timestamp&) {
			return py::make_tuple("y", "d", "h", "m", "s", "ss", "c", "sbs", "source", "recent");
		}, "Return dictionary keys")
		.def("__getitem__", [](const Timestamp& ts, const std::string& key) {
			if (key == "y") return py::cast(ts.y);
			if (key == "d") return py::cast(ts.d);
			if (key == "h") return py::cast(ts.h);
			if (key == "m") return py::cast(ts.m);
			if (key == "s") return py::cast(ts.s);
			if (key == "ss") return py::cast(ts.ss);
			if (key == "c") return py::cast(ts.c & 0xfff);
			if (key == "sbs") return py::cast(ts.sbs);
			if (key == "source") return py::cast(ts.get_source());
			if (key == "recent") return py::cast(ts.is_recent());
			throw py::key_error(key);
		}, "key"_a, "Get field value by name")
		.def("__iter__", [](const Timestamp&) {
			return py::iter(py::make_tuple("y", "d", "h", "m", "s", "ss", "c", "sbs", "source", "recent"));
		}, "Iterate over field names")
		.def("__len__", [](const Timestamp&) { return 10; }, "Return number of fields")
		.def("__repr__", [](const Timestamp& ts) {
			return fmt::format("Timestamp(y={} d={} {}:{}:{} ss={} c={} sbs={} recent={})",
					ts.y, ts.d,
					ts.h, ts.m, ts.s,
					ts.ss, ts.c, ts.sbs,
					ts.is_recent() ? "Y" : "N");
		});

	pybind11::str_enum<Timestamp::Source>(m, "TimestampSource")
		.value("BACKPLANE", Timestamp::Source::BACKPLANE)
		.value("TEST", Timestamp::Source::TEST)
		.value("SMA", Timestamp::Source::SMA)
		.value("GND", Timestamp::Source::GND);

	/* Helper: convert interleaved raw int32 I/Q data to complex<double>,
	 * applying the packetizer gain correction (/ 256). */
	auto raw_to_complex = [](const int32_t* raw, py::ssize_t num_samples) {
		py::array_t<std::complex<double>> result(num_samples);
		auto out = result.mutable_unchecked<1>();
		for (size_t i = 0; i < num_samples; i++)
			out(i) = std::complex<double>(raw[2*i], raw[2*i+1]) / 256.;
		return result;
	};

	py::class_<ReadoutPacket>(m, "ReadoutPacket")
		.def(py::init<>())
		.def(py::init([](py::bytes data) {
			char* buffer;
			Py_ssize_t length;
			if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length)) {
				throw std::runtime_error("Unable to extract bytes contents");
			}
			return ReadoutPacket::from_bytes(buffer, length);
		}), "data"_a, "Deserialize packet from bytes")
		.def(py::init([](uint32_t magic, uint16_t version, uint16_t serial,
		                 uint8_t num_modules, uint8_t flags, int fir_stage,
		                 uint8_t module, uint32_t seq) {
			ReadoutPacket pkt;
			pkt.magic = magic;
			pkt.version = version;
			pkt.serial = serial;
			pkt.num_modules = num_modules;
			pkt.flags = flags;
			pkt.fir_stage = fir_stage;
			pkt.module = module;
			pkt.seq = seq;
			return pkt;
		}),
		py::kw_only(),
		"magic"_a, "version"_a, "serial"_a, "num_modules"_a, "flags"_a,
		"fir_stage"_a, "module"_a, "seq"_a,
		"Create readout packet with specified scalar fields")
		.def("__bytes__", &ReadoutPacket::to_bytes, "Serialize packet to bytes")
		.def_readwrite("magic", &ReadoutPacket::magic)
		.def_readwrite("version", &ReadoutPacket::version)
		.def_readwrite("serial", &ReadoutPacket::serial)
		.def_readwrite("num_modules", &ReadoutPacket::num_modules)
		.def_readwrite("flags", &ReadoutPacket::flags)
		.def_readwrite("fir_stage", &ReadoutPacket::fir_stage)
		.def_readwrite("module", &ReadoutPacket::module)
		.def_readwrite("seq", &ReadoutPacket::seq)

		// raw_samples: zero-copy view of interleaved int32 I/Q pairs
		.def_property("raw_samples",
			[](ReadoutPacket& self) {
				auto& vec = self.raw_samples();
				return py::array_t<int32_t>(
					{static_cast<py::ssize_t>(vec.size())},
					{sizeof(int32_t)},
					vec.data(),
					py::cast(self)
				);
			},
			[](ReadoutPacket& self, py::array_t<int32_t> arr) {
				auto buf = arr.request();
				auto* ptr = static_cast<int32_t*>(buf.ptr);
				self.raw_samples() = std::vector<int32_t>(ptr, ptr + buf.size);
			})

		.def_property("ts",
			[](const ReadoutPacket& self) { return self.timestamp(); },
			[](ReadoutPacket& self, const Timestamp& ts) { self.timestamp() = ts; })
		// __len__: number of channels
		.def("__len__", &ReadoutPacket::get_num_channels)

		// __getitem__: lazy scaled access (single index or slice)
		.def("__getitem__", [](ReadoutPacket& self, py::object key) -> py::object {
			auto& raw = self.raw_samples();
			int num_channels = self.get_num_channels();

			if (py::isinstance<py::int_>(key)) {
				int idx = key.cast<int>();
				if (idx < 0) idx += num_channels;
				if (idx < 0 || idx >= num_channels)
					throw py::index_error("channel index out of range");
				return py::cast(
					std::complex<double>(raw[2*idx], raw[2*idx+1]) / 256.
				);
			}

			if (py::isinstance<py::slice>(key)) {
				auto slice = key.cast<py::slice>();
				py::ssize_t start, stop, step, length;
				if (!slice.compute(num_channels, &start, &stop, &step, &length))
					throw py::error_already_set();

				py::array_t<std::complex<double>> result(length);
				auto out = result.mutable_unchecked<1>();
				for (py::ssize_t i = 0; i < length; i++) {
					py::ssize_t ch = start + i * step;
					out(i) = std::complex<double>(raw[2*ch], raw[2*ch+1]) / 256.;
				}
				return result;
			}

			// Fall back to numpy fancy indexing: convert key to index array
			auto indices = py::array_t<int32_t>::ensure(key);
			if (!indices)
				throw py::type_error("indices must be integers, slices, or integer arrays");
			auto idx_buf = indices.request();
			auto* idx_ptr = static_cast<int32_t*>(idx_buf.ptr);
			py::ssize_t n = idx_buf.size;

			py::array_t<std::complex<double>> result(n);
			auto out = result.mutable_unchecked<1>();
			for (py::ssize_t i = 0; i < n; i++) {
				int ch = idx_ptr[i];
				if (ch < 0) ch += num_channels;
				if (ch < 0 || ch >= num_channels)
					throw py::index_error("channel index out of range");
				out(i) = std::complex<double>(raw[2*ch], raw[2*ch+1]) / 256.;
			}
			return result;
		})

		// __setitem__: write complex values (×256) into raw int32 storage
		.def("__setitem__", [](ReadoutPacket& self, py::object key, py::object value) {
			auto& raw = self.raw_samples();
			int num_channels = self.get_num_channels();

			if (py::isinstance<py::int_>(key)) {
				int idx = key.cast<int>();
				if (idx < 0) idx += num_channels;
				if (idx < 0 || idx >= num_channels)
					throw py::index_error("channel index out of range");
				auto v = value.cast<std::complex<double>>();
				raw[2*idx] = static_cast<int32_t>(v.real() * 256.);
				raw[2*idx+1] = static_cast<int32_t>(v.imag() * 256.);
				return;
			}

			// Slice or full assignment — coerce value to complex array
			auto arr = py::array_t<std::complex<double>>::ensure(value);
			if (!arr)
				throw std::runtime_error("value must be array-like of complex values");
			auto buf = arr.request();
			auto* ptr = static_cast<std::complex<double>*>(buf.ptr);

			if (py::isinstance<py::slice>(key)) {
				auto slice = key.cast<py::slice>();
				py::ssize_t start, stop, step, length;
				if (!slice.compute(num_channels, &start, &stop, &step, &length))
					throw py::error_already_set();
				if (buf.size != length)
					throw std::runtime_error("value length does not match slice length");
				// Resize raw storage if empty (initial assignment)
				if (raw.empty()) raw.resize(2 * num_channels, 0);
				for (py::ssize_t i = 0; i < length; i++) {
					py::ssize_t ch = start + i * step;
					raw[2*ch] = static_cast<int32_t>(ptr[i].real() * 256.);
					raw[2*ch+1] = static_cast<int32_t>(ptr[i].imag() * 256.);
				}
				return;
			}

			throw py::type_error("index must be an integer or slice");
		})

		// __array__: numpy protocol — returns all channels as complex<double>/256
		.def("__array__", [raw_to_complex](ReadoutPacket& self, py::object /* dtype */, py::object /* copy */) {
			auto& raw = self.raw_samples();
			return raw_to_complex(raw.data(), raw.size() / 2);
		}, "dtype"_a = py::none(), "copy"_a = py::none())

		.def("__repr__", [](const ReadoutPacket& pkt) {
			return fmt::format("ReadoutPacket(serial={} module={} seq={})",
					pkt.serial, pkt.module, pkt.seq);
		});

	py::class_<PFBPacket>(m, "PFBPacket")
		.def(py::init<>())
		.def(py::init([](py::bytes data) {
			char* buffer;
			Py_ssize_t length;
			if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length)) {
				throw std::runtime_error("Unable to extract bytes contents");
			}
			return PFBPacket::from_bytes(buffer, length);
		}), "data"_a, "Deserialize packet from bytes")
		.def("__bytes__", &PFBPacket::to_bytes, "Serialize packet to bytes")
		.def_readwrite("magic", &PFBPacket::magic)
		.def_readwrite("version", &PFBPacket::version)
		.def_readwrite("mode", &PFBPacket::mode)
		.def_readwrite("serial", &PFBPacket::serial)
		.def_readwrite("slot1", &PFBPacket::slot1)
		.def_readwrite("slot2", &PFBPacket::slot2)
		.def_readwrite("slot3", &PFBPacket::slot3)
		.def_readwrite("slot4", &PFBPacket::slot4)
		.def_readwrite("num_samples", &PFBPacket::num_samples)
		.def_readwrite("module", &PFBPacket::module)
		.def_readwrite("seq", &PFBPacket::seq)

		// raw_samples: zero-copy view of interleaved int32 I/Q pairs
		.def_property("raw_samples",
			[](PFBPacket& self) {
				auto& vec = self.raw_samples();
				return py::array_t<int32_t>(
					{static_cast<py::ssize_t>(vec.size())},
					{sizeof(int32_t)},
					vec.data(),
					py::cast(self)
				);
			},
			[](PFBPacket& self, py::array_t<int32_t> arr) {
				auto buf = arr.request();
				auto* ptr = static_cast<int32_t*>(buf.ptr);
				self.raw_samples() = std::vector<int32_t>(ptr, ptr + buf.size);
			})

		.def_property("ts",
			[](const PFBPacket& self) { return self.timestamp(); },
			[](PFBPacket& self, const Timestamp& ts) { self.timestamp() = ts; })
		// __len__: number of samples
		.def("__len__", &PFBPacket::get_num_samples)

		// __getitem__: lazy scaled access (single index or slice)
		.def("__getitem__", [](PFBPacket& self, py::object key) -> py::object {
			auto& raw = self.raw_samples();
			int num_samples = self.get_num_samples();

			if (py::isinstance<py::int_>(key)) {
				int idx = key.cast<int>();
				if (idx < 0) idx += num_samples;
				if (idx < 0 || idx >= num_samples)
					throw py::index_error("sample index out of range");
				return py::cast(
					std::complex<double>(raw[2*idx], raw[2*idx+1]) / 256.
				);
			}

			if (py::isinstance<py::slice>(key)) {
				auto slice = key.cast<py::slice>();
				py::ssize_t start, stop, step, length;
				if (!slice.compute(num_samples, &start, &stop, &step, &length))
					throw py::error_already_set();

				py::array_t<std::complex<double>> result(length);
				auto out = result.mutable_unchecked<1>();
				for (py::ssize_t i = 0; i < length; i++) {
					py::ssize_t s = start + i * step;
					out(i) = std::complex<double>(raw[2*s], raw[2*s+1]) / 256.;
				}
				return result;
			}

			auto indices = py::array_t<int32_t>::ensure(key);
			if (!indices)
				throw py::type_error("indices must be integers, slices, or integer arrays");
			auto idx_buf = indices.request();
			auto* idx_ptr = static_cast<int32_t*>(idx_buf.ptr);
			py::ssize_t n = idx_buf.size;

			py::array_t<std::complex<double>> result(n);
			auto out = result.mutable_unchecked<1>();
			for (py::ssize_t i = 0; i < n; i++) {
				int s = idx_ptr[i];
				if (s < 0) s += num_samples;
				if (s < 0 || s >= num_samples)
					throw py::index_error("sample index out of range");
				out(i) = std::complex<double>(raw[2*s], raw[2*s+1]) / 256.;
			}
			return result;
		})

		// __setitem__: write complex values (×256) into raw int32 storage
		.def("__setitem__", [](PFBPacket& self, py::object key, py::object value) {
			auto& raw = self.raw_samples();
			int num_samples = self.get_num_samples();

			if (py::isinstance<py::int_>(key)) {
				int idx = key.cast<int>();
				if (idx < 0) idx += num_samples;
				if (idx < 0 || idx >= num_samples)
					throw py::index_error("sample index out of range");
				auto v = value.cast<std::complex<double>>();
				raw[2*idx] = static_cast<int32_t>(v.real() * 256.);
				raw[2*idx+1] = static_cast<int32_t>(v.imag() * 256.);
				return;
			}

			auto arr = py::array_t<std::complex<double>>::ensure(value);
			if (!arr)
				throw std::runtime_error("value must be array-like of complex values");
			auto buf = arr.request();
			auto* ptr = static_cast<std::complex<double>*>(buf.ptr);

			if (py::isinstance<py::slice>(key)) {
				auto slice = key.cast<py::slice>();
				py::ssize_t start, stop, step, length;
				if (!slice.compute(num_samples, &start, &stop, &step, &length))
					throw py::error_already_set();
				if (buf.size != length)
					throw std::runtime_error("value length does not match slice length");
				if (raw.empty()) raw.resize(2 * num_samples, 0);
				for (py::ssize_t i = 0; i < length; i++) {
					py::ssize_t s = start + i * step;
					raw[2*s] = static_cast<int32_t>(ptr[i].real() * 256.);
					raw[2*s+1] = static_cast<int32_t>(ptr[i].imag() * 256.);
				}
				return;
			}

			throw py::type_error("index must be an integer or slice");
		})

		// __array__: numpy protocol — returns all samples as complex<double>/256
		.def("__array__", [raw_to_complex](PFBPacket& self, py::object /* dtype */, py::object /* copy */) {
			auto& raw = self.raw_samples();
			return raw_to_complex(raw.data(), raw.size() / 2);
		}, "dtype"_a = py::none(), "copy"_a = py::none())

		.def("__repr__", [](const PFBPacket& pkt) {
			return fmt::format("PFBPacket(serial={} module={} seq={} samples={})",
					pkt.serial, pkt.module, pkt.seq, pkt.num_samples);
		});

	py::class_<Packet>(m, "Packet")
		.def("serial", &Packet::serial)
		.def("module", &Packet::module)
		.def("seq", &Packet::seq)
		.def("ts", &Packet::timestamp)
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
		.def("clear", &PacketQueue::clear, "Clear all packets from queue")
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
		.def(py::init<py::object, size_t, size_t, size_t>(),
			 "socket"_a,
			 "reorder_window"_a = 256,
			 "queue_max_size"_a = 4096,
			 "flush_threshold"_a = 128,
			 "Create readout packet receiver with reorder_window (min buffer) and flush_threshold (flush every N additional packets)");

	py::class_<PFBPacketReceiver, PacketReceiver>(m, "PFBPacketReceiver",
		"Packet receiver for PFB packets.")
		.def(py::init<py::object, size_t, size_t, size_t>(),
			 "socket"_a,
			 "reorder_window"_a = 256,
			 "queue_max_size"_a = 4096,
			 "flush_threshold"_a = 128,
			 "Create PFB packet receiver with reorder_window (min buffer) and flush_threshold (flush every N additional packets)");

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

	m.attr("MULTICAST_GROUP") = MULTICAST_GROUP;
	m.attr("READOUT_PACKET_MAGIC") = READOUT_PACKET_MAGIC;
	m.attr("PFB_PACKET_MAGIC") = PFB_PACKET_MAGIC;
	m.attr("STREAMER_PORT") = STREAMER_PORT;
	m.attr("PFB_STREAMER_PORT") = PFB_STREAMER_PORT;
	m.attr("LONG_PACKET_SIZE") = LONG_PACKET_SIZE;
	m.attr("SHORT_PACKET_SIZE") = SHORT_PACKET_SIZE;
	m.attr("PFB_PACKET_SIZE") = PFB_PACKET_SIZE(PFBPACKET_NSAMP_MAX);
	m.attr("LONG_PACKET_CHANNELS") = LONG_PACKET_CHANNELS;
	m.attr("SHORT_PACKET_CHANNELS") = SHORT_PACKET_CHANNELS;
	m.attr("LONG_PACKET_VERSION") = LONG_PACKET_VERSION;
	m.attr("SHORT_PACKET_VERSION") = SHORT_PACKET_VERSION;
	m.attr("PFBPACKET_NSAMP_MAX") = PFBPACKET_NSAMP_MAX;
	m.attr("SS_PER_SECOND") = SS_PER_SECOND;

	/* Must match PY_API_VERSION in rfmux/streamer/__init__.py */
	m.attr("_SO_API_VERSION") = 1;
}
