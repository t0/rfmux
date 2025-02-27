# rfmux

`rfmux` is the python API for t0.technology Control and Readout System (CRS).
It includes python bindings to methods and properties for the on-board C++ and firmware operations; a framework asynchronous operation of large arrays; and a common set of algorithms for based operation of Kinetic Inductance Detectors.


Distributed within `rfmux` is the binary files for the firmware itself (`/firmware`) as well as a C++ implementation of a data acquisition system (`parser`) for the continuously streamed downsampled data.

`rfmux` is designed to run locally on your control computer and is also embedded on CRS boards and useable via the Jupyter Hub that is hosted on each CRS and accessible on the network at `rfmux<serial>.local` when the board is booted.

## Installation

Clone rfmux from github, and then install via pip (will also install the dependencies):

```bash
cd rfmux
pip install .
```

Note that rfmux requires Python 3.10 or later

## Firmware & Git LFS

The repository contains a `firmware` directory with large binary files managed via Git LFS. To enable and retrieve these files:

1. **Install Git LFS:** Follow the installation instructions from [git-lfs](https://git-lfs.github.com/).
2. **Pull LFS files:**

   ```bash
   git lfs install
   git lfs pull --exclude=
   ```

For instructions on making flashcards from the firmware, please see the [Firmware README](./firmware/README.md).

## Network Tuning for UDP Buffers

To improve long captures of data you may need to increase your system's receive buffer size. For example:

```bash
sudo sysctl net.core.rmem_max=67108864
```

To make this change persistent across reboots, add the following line to `/etc/sysctl.conf` or a file in `/etc/sysctl.d/`:

```bash
net.core.rmem_max=67108864
```

## Platform Compatibility

- **Linux:** Primary platform for testing and deployment.
- **Windows:** Supported but less tested. Specific windows README is available in the [Windows README](./README.Windows).

## Contribution & Feedback

We actively encourage contributions and feedback. Understanding operator code and needs is how we determine what to add to rfmux.
- **Pull Requests:** Your pull requests are welcome.
- **Issues:** Please submit ticket issues for bugs or enhancement suggestions.

## Usage & Operator API

rfmux is designed to be the API for operator code, rather than the end-to-end deployment control software itself.
Nevertheless, common algorithms and operations are gradually being included, such as:
- **Network Analysis**
- **Transferfunction-corrected Spectra Collection**

If you have suggestions for algorithms or operations to be included in the API, please let us know.


The rfmux API itself is currently compatible with two different end-to-end KIDs control software stacks:
- **hidfmux:** End-to-end KIDs characterization and astrophysics deployment developed by McGill University.
- **citkid:** Developed at JPL/Caltech. More details can be found at the [citkid GitHub repository](https://github.com/loganfoote/citkid).

## Documentation Philosophy

Our documentation lives in executable Jupyter notebooks, which offer interactive walkthroughs and examples. Currently, these notebooks reside in the `/home` directory, which also serves as the homepage for the rfmux Jupyter Hub.

---

For more details, visit our documentation or submit an issue for any specific inquiries.
