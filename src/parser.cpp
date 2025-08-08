#include <sys/socket.h>
#include <netinet/ip.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <poll.h>

#include "packet.hpp"

/* The reorder buffer (see below) needs a comparison operator
 * that sorts packets by sequence number. This needs to work
 * even when seq wraps around (that is, from 0xffffffff to 0).
 * Even though this struct is essentially an internal detail
 * (hence properly "private", it's public so it can be tested
 * via static_assert in packet.cpp. */
struct ReadoutPacketComparison {
	static constexpr bool ascends(uint32_t sa, uint32_t sb) {
		/* Ensure correct behaviour when seq wraps around to 0 */
		return static_cast<uint32_t>(sb - sa) < 0x80000000u;
	}

	bool operator()(const RawFrame &a, const RawFrame &b) const {
		auto sa = a.header.seq;
		auto sb = b.header.seq;
		return !ascends(sa, sb);
	}
};

/* Packets may be reordered during transmission or in network
 * hardware. To work around this, we use a std::priority_queue
 * as a reorder buffer. */
using reorder_queue_type = std::priority_queue<
	RawFrame,
	std::vector<RawFrame>,
	ReadoutPacketComparison>;

/* Because different modules/boards will have wildly different
 * sequence numbers, a single priority queue isn't good enough
 * on its own - the module with the lowest sequence number will
 * zip through the queue before anyone else. So, we need to keep
 * one priority queue for each (board serial, module) tuple. */
using module_uid_type = std::tuple<uint16_t, uint8_t>;

/* We store these in a std::map */
using reorder_queues_type = std::map<
	module_uid_type,
	reorder_queue_type>;

struct Parser {
	Parser(std::string ifname) {
		if(!(sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)))
			throw std::runtime_error("Failed to create socket!");

		/* Permit re-binding */
		const int so_reuseaddr = 1;
		setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &so_reuseaddr, sizeof(so_reuseaddr));

		/* Use large socket buffer */
		const int so_rcvbuf = 16777216;
		if(setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &so_rcvbuf, sizeof(so_rcvbuf)) == -1)
		throw std::runtime_error(fmt::format("Can't set SO_RCVBUF to {}!", so_rcvbuf));

		/* Don't deliver all packets - just the ones we've asked for */
		const int ip_multicast_all = 0;
		setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_ALL, &ip_multicast_all, sizeof(ip_multicast_all));

		/* bind() */
		struct sockaddr_in addr = {
			.sin_family = AF_INET,
			.sin_port = ntohs(STREAMER_PORT),
			.sin_addr = { ntohl(INADDR_ANY) },
			.sin_zero = 0,
		};
		if(bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) == -1)
			throw std::runtime_error("Can't bind() to this IP/port. Hint: is someone else already camping here for samples?");

		/* Now work towards IP_ADD_MEMBERSHIP.  Initialize "addr" with the specified
		 * interface's IPv4 address */
		struct ifaddrs *ifas, *ifa = NULL;
		if(getifaddrs(&ifas) != 0)
			throw std::runtime_error("Failure to retrieve Ethernet interface addresses!");

		memset(&addr, 0, sizeof(addr));
		for(ifa = ifas; ifa; ifa=ifa->ifa_next) {
			if(strcmp(ifa->ifa_name, ifname.c_str()) == 0 &&
					ifa->ifa_addr->sa_family == AF_INET) {
				memcpy(&addr, ifa->ifa_addr, sizeof(struct sockaddr_in));
				break;
			}
		}
		if(!ifa)
			throw std::runtime_error(fmt::format("Failure to find interface {}", ifname));
		freeifaddrs(ifas);

		/* Add multicast membership. This creates an IGMP subscription
		 * request that ducts all CRS boards' traffic into this network
		 * endpoint. Hope it's enough bandwidth... */
		struct ip_mreqn mreqn;

		memset(&mreqn, 0, sizeof(mreqn));
		inet_aton(STREAMER_HOST, &mreqn.imr_multiaddr);
		memcpy(&mreqn.imr_address, &addr.sin_addr, sizeof(struct in_addr));
		mreqn.imr_ifindex = if_nametoindex(ifname.c_str());

		if(setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreqn, sizeof(mreqn)) == -1)
			throw std::runtime_error(fmt::format("Multicast: couldn't add membership. Does your "
					"network configuration (DHCP or local) specify where to send multicast "
					"packets? ({})", strerror(errno)));

		/* Capture reference to Python socket. (This also takes
		 * care of close()ing the socket on destruction.) */
		auto socket_mod = py::module_::import("socket");
		socket_obj = socket_mod.attr("socket")("fileno"_a=sockfd);
	}

	auto sniff(double timeout_sec) {
		const int num_packets = 10000;
		std::set<module_uid_type> received;

		std::vector<mmsghdr> msgs(num_packets);
		std::vector<iovec> iovs(num_packets);
		std::vector<ReadoutPacketHeader> incoming(num_packets);

		/* Initialize these structures to point to each other, and contain 1 packet each */
		for(unsigned int n=0; n<num_packets; n++) {
			iovs[n].iov_base = &incoming[n];
			iovs[n].iov_len = sizeof(ReadoutPacketHeader); /* just the header */
			msgs[n].msg_hdr.msg_iov = &iovs[n];
			msgs[n].msg_hdr.msg_iovlen = 1;
		};

		struct pollfd pfd = { .fd=sockfd, .events=POLLIN, .revents=0 };

		struct timespec start;
		clock_gettime(CLOCK_MONOTONIC, &start);
		auto ms_elapsed = [&start]() -> long {
			struct timespec now;
			clock_gettime(CLOCK_MONOTONIC, &now);
			return (now.tv_sec - start.tv_sec)*1000 + (now.tv_nsec - start.tv_nsec)/1000000;
		};

		while(1) {
			long elapsed = ms_elapsed();
			if(elapsed > timeout_sec*1e3)
				break;

			int rv = poll(&pfd, 1, timeout_sec*1e3 - elapsed);
			if(rv < 0)
				throw std::runtime_error(fmt::format("poll() failed: {}", strerror(errno)));
			if(rv == 0)
				break;

			rv = recvmmsg(sockfd, &msgs[0], num_packets, MSG_DONTWAIT, NULL);
			if(rv <= 0)
				throw std::runtime_error(fmt::format("Failed to receive packets! Got {} ({})", rv, strerror(errno)));

			/* Validate packets, and re-format short packets as long ones
			 * (so we're only dealing with a single format downstream) */
			for(int i=0; i<rv; i++) {
				if(incoming[i].magic != PACKET_MAGIC)
					throw std::runtime_error(fmt::format("Packet didn't have correct magic! (0x{:08x})",
							(uint32_t)incoming[i].magic));

				/* Move the packets into a priority queue by sequence number. */
				received.emplace(incoming[i].serial, incoming[i].module+1);
			}
		}
		return received;
	}

	void receive(size_t num_packets, std::optional<double> timeout_sec, int flags) {
		/* Receive "num_packets" frames and sort them into per-module reorder queues. */

		if(num_packets <= 0 || num_packets > 65536)
			throw std::out_of_range("Insane number of packets requested.");

		/* Timeout, maybe */
		struct timespec ts = {
			.tv_sec = (time_t)timeout_sec.value_or(0),
			.tv_nsec = lrint(1e9 * (timeout_sec.value_or(0)-(time_t)timeout_sec.value_or(0))),
		};

		std::vector<mmsghdr> msgs(num_packets);
		std::vector<iovec> iovs(num_packets);
		std::vector<RawFrame> incoming(num_packets);

		/* Initialize these structures to point to each other, and contain 1 packet each */
		for(unsigned int n=0; n<num_packets; n++) {
			iovs[n].iov_base = &incoming[n];
			iovs[n].iov_len = LONG_PACKET_SIZE; /* worst case */
			msgs[n].msg_hdr.msg_iov = &iovs[n];
			msgs[n].msg_hdr.msg_iovlen = 1;
		};

		int rv = recvmmsg(sockfd, &msgs[0], num_packets, flags, timeout_sec ? &ts : NULL);
		if(rv <= 0)
			throw std::runtime_error(fmt::format("Failed to receive packets! Got {} ({})", rv, strerror(errno)));

		if(rv != (int)num_packets) {
			/* Received some packets, but less than requested -
			 * perhaps control-C or similar. Do the best we can. */
			msgs.resize(rv);
			iovs.resize(rv);
			incoming.resize(rv);
		}

		/* Validate packets, and re-format short packets as long ones
		 * (so we're only dealing with a single format downstream) */
		for(auto &p : incoming) {
			auto h = reinterpret_cast<ReadoutPacketHeader*>(&p);
			if(h->magic != PACKET_MAGIC)
				throw std::runtime_error(fmt::format("Packet didn't have correct magic! (0x{:08x})",
						(uint32_t)h->magic));

			/* Move the packets into a priority queue by sequence number. */
			reorder_queues[{h->serial, h->module+1}].push(std::move(p));
		}
	}

	py::object drain(module_uid_type qid, size_t minimum_occupancy=0) {
		auto it = reorder_queues.find(qid);
		if(it == reorder_queues.end())
			return py::none();

		auto &q = it->second;

		int nsamples = q.size() - minimum_occupancy;
		if(nsamples <= 0)
			return py::none();

		auto ret = py::array_t<ReadoutFrame>(nsamples);
		auto view = ret.mutable_unchecked<1>();
		ReadoutFrame f;

		for(int n=0; n<nsamples; n++) {
			auto p = q.top();
			q.pop();

			auto h = reinterpret_cast<ReadoutPacketHeader*>(&p);
			auto pl = reinterpret_cast<StaticReadoutPacket<LONG_PACKET_CHANNELS>*>(&p);
			auto ps = reinterpret_cast<StaticReadoutPacket<SHORT_PACKET_CHANNELS>*>(&p);

			/* Zero ReadoutFrame */
			f = {};

			/* Fill in whatever we can */
			switch(h->version) {
				case LONG_PACKET_VERSION:
					f.seq = pl->h.seq;
					std::transform(std::begin(pl->samples), std::end(pl->samples),
							std::begin(f.samples),
							[](struct iq32 x) {
						return std::complex<double>(x.i, x.q) / 256.;
					});
					f.ts = pl->ts;
					break;
				case SHORT_PACKET_VERSION:
					f.seq = ps->h.seq;
					std::transform(std::begin(ps->samples), std::end(ps->samples),
							std::begin(f.samples),
							[](struct iq32 x) {
						return std::complex<double>(x.i, x.q) / 256.;
					});
					f.ts = ps->ts;
					break;
				default:
					throw std::runtime_error(fmt::format("Invalid packet version! (0x{:04x})",
							(uint32_t)h->version));
			}

			view[n] = f;
		}

		return ret;
	}

	int sockfd;
	py::object socket_obj; /* Python representation of the socket */

	reorder_queues_type reorder_queues;
};

PYBIND11_MODULE(_parser, m) {
	m.doc() = "Parser helper";

	auto parser = py::class_<Parser>(m, "Parser")
		.def(py::init<std::string>(), "ifname"_a)
		.def("receive", &Parser::receive,
				"num_packets"_a,
				py::kw_only(),
				"timeout_sec"_a=std::nullopt,
				"flags"_a=0,
				"Receive up to num_packets readout packets into a reorder buffer.\n"
				"\n"
				"This data can subsequently be retrieved using 'drain'.\n"
				"A reorder buffer is necessary because UDP data carries no\n"
				"ordering guarantees, and additionally, multicore streaming\n"
				"from the CRS may result in out-of-sequence packets even on\n"
				"the CRS.")
		.def("get_socket",
				[](Parser &p){ return py::handle(p.socket_obj); },
				"Returns a Python socket associated with the parser.\n"
				"\n"
				"Because packet reception using this socket bypasses\n"
				"the reorder queues, use of this socket for anything\n"
				"other than investigation is discouraged.\n")
		.def("get_queues", [](Parser &p) {
				std::vector<module_uid_type> keys;
				keys.reserve(p.reorder_queues.size());
				for(auto &kv : p.reorder_queues)
					if(!kv.second.empty())
						keys.emplace_back(kv.first);
				return keys;
				})

		.def("sniff", &Parser::sniff, "timeout"_a=1.0,
				"Determine which CRS serial/modules are streaming.\n"
				"\n"
				"Operates by collecting multicast traffic for a\n"
				"limited amount of time. Boards that aren't active\n"
				"(obviously) can't be detected this way.\n"
				"\n"
				"This function exists because hdf5 doesn't allow\n"
				"metadata changes in Single-Writer, Multiple-Reader\n"
				"(SWMR) mode, so we have to anticipate which CRSes\n"
				"will be present before the parser starts writing\n"
				"data to disk. It's hoped this will eventually be\n"
				"resolved with an updated HDF5 release.)\n"
				"\n"
				"Returns a set of (serial, module) tuples.\n")

		.def("drain", &Parser::drain,
				"qid"_a,
				"minimum_occupancy"_a=0)
		;

	PYBIND11_NUMPY_DTYPE(Timestamp, y, d, h, m, s, ss, c, sbs);
	PYBIND11_NUMPY_DTYPE(ReadoutFrame, seq, samples, ts);

	/* Expose the dtype to Python (for consistent in-memory representation) */
	m.attr("ReadoutFrame") = py::dtype::of<ReadoutFrame>();
}
