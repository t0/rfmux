#pragma once

#include <vector>
#include <list>
#include <string>
#include <stdint.h>
#include <optional>
#include <netdb.h>
#include <cassert>
#include <queue>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include <fmt/format.h>

namespace py = pybind11;
using namespace pybind11::literals;

/* Multicast address for streaming */
#define STREAMER_HOST		"239.192.0.2"

static const inline uint32_t PACKET_MAGIC=0x5344494b;
static const inline uint32_t FAST_PACKET_MAGIC = 0xfa57b175;
static const uint16_t STREAMER_PORT=9876;

static const size_t LONG_PACKET_CHANNELS = 1024;
static const uint32_t LONG_PACKET_SIZE = (LONG_PACKET_CHANNELS + 2 + 4) * 8;
static const uint16_t LONG_PACKET_VERSION = 5;

static const size_t SHORT_PACKET_CHANNELS = 128;
static const uint32_t SHORT_PACKET_VERSION = 6;

enum class TimestampPort {
	BACKPLANE,
	TEST,
	SMA,
	GND,
};

struct Timestamp {
	/* SIGNED types are used here to permit negative numbers during
	 * renormalization and comparison. THE ORDER IS IMPORTANT, since
	 * this matches the VHDL's field packing. */
	int32_t y,d,h,m,s;
	int32_t ss;
	int32_t c, sbs;

	int32_t get_c_field(void) const { return c & 0x1fffffff; };
	TimestampPort get_source(void) const {
		switch((c >> 29) & 0x3) {
			case 0: return TimestampPort::BACKPLANE;
			case 1: return TimestampPort::TEST;
			case 2: return TimestampPort::SMA;
			case 3: return TimestampPort::GND;
		}
		throw std::runtime_error("Unexpected TimestampPort");
	};
	bool is_recent(void) const { return c & 0x80000000; };
};

static const uint8_t PACKET_FLAGS_OVERRANGE = 0x1;
static const uint8_t PACKET_FLAGS_OVERVOLTAGE = 0x2;

struct ReadoutPacketHeader {
	uint32_t magic;

	uint16_t version;
	uint16_t serial; /* IceBoard serial */

	uint8_t num_modules;
	uint8_t flags; /* formerly block index */
	uint8_t decimation;
	uint8_t module; /* linear; 0-num_modules-1 */

	uint32_t seq; /* incrementing sequence number */
};

/* When we receive a multicast packet, we don't yet know what kind it will be -
 * we only know that it will have a maximum length and a common header. */
struct RawFrame {
	union {
		ReadoutPacketHeader header;
		char buf[LONG_PACKET_SIZE];
	};
};

/* This structure describes on-the-wire readout packets */
struct iq32 { int32_t i, q; };
template<int nsamples>
struct StaticReadoutPacket {
	struct ReadoutPacketHeader h;
	struct iq32 samples[nsamples];
	struct Timestamp ts;
};

/* Here's what's exposed in Python */
struct ReadoutFrame {
	uint32_t seq;
	std::complex<double> samples[LONG_PACKET_CHANNELS];
	struct Timestamp ts;
};
