# rfmux

`rfmux` is the python API for t0.technology Control and Readout System (CRS).
rfmux includes python bindings to methods and properties for the on-board C++ and firmware operations; a framework asynchronous operation of large arrays; and a common set of algorithms for based operation of Kinetic Inductance Detectors.
The CRS + MKIDs firmware is described in [this conference proceedings](https://arxiv.org/abs/2406.16266).

Distributed within `rfmux` is the binary files for the firmware itself (`/firmware`) as well as a C++ implementation of a data acquisition system (`parser`) for the continuously streamed downsampled data.

`rfmux` is designed to run locally on your control computer and is also embedded on CRS boards and useable via the Jupyter Hub that is hosted on each CRS and accessible on the network at `rfmux<serial>.local` when the board is booted.

## Installation

Clone rfmux from github, and then install via pip (will also install the dependencies):

```bash
cd rfmux
pip install -e .
```

Note that rfmux requires Python 3.12 or later

## Periscope

https://github.com/user-attachments/assets/fa8eb68c-8593-43b1-af96-285720ed0d5f

An additional system library that cannot be automatically installed with pip is xcb-cursor.
This is required for the real-time GUI visualization tool `periscope`. It can be installed via:
```bash
$ sudo apt-get install libxcb-cursor0
```
The visualizer can be called directly from the commandline using the CRS serial number:
```bash
$ periscope 0022 --module 2 --channels "1,3&5"
```
or directly from within a Python / IPython / Jupyter environment as a method of a CRS object:
```python
>>> crs.raise_periscope(module=2, channels="1,3&5")
```

### Emulation Mode

`periscope` can also be run in a local emulation mode that simulates both the CRS and a KID array by using "mock" as the CRS serial number.
When launched this way, you will be prompted to customize the parameters of the KID array to be emulated.

```bash
$ periscope mock --module 1 --channels "1,3&5"
```

Emulation mode is under active development, and not all CRS functions are instrumented, but it does emulate KID non-linear inductance, all of the tuning algorithms currently used in rfmux, fitters, and the real-time visualization.
Instantiating a hardware map using mock mode also allows for offline algorithm development, as most of the signal processing functions are properly emulated at baseband.

```python
s = load_session("""
!HardwareMap
- !flavour "rfmux.core.mock"
- !CRS { serial: "MOCK0001", hostname: "127.0.0.1" }
""")
```

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
- **Windows:** Supported but less tested. Specific windows README is available in the [Windows README](./README.Windows.md).

## Contribution & Feedback

We actively encourage contributions and feedback. Understanding operator code and needs is how we determine what to add to rfmux.
- **Pull Requests:** Your pull requests are welcome.
- **Issues:** Please submit ticket issues for bugs or enhancement suggestions.

### Collaborator Slack Channel
Anybody working with a CRS board is welcome to join the #crs-collaboration slack channel.
Just email Joshua@t0.technology with your name, affiliation, and what project you are working on.


## Usage & Operator API

rfmux is designed to be the API for operator code, rather than the end-to-end deployment control software itself.
Nevertheless, common algorithms and operations are gradually being included, such as:
- **Network Analysis**
- **Transferfunction-corrected Spectra Collection**

If you have suggestions for algorithms or operations to be included in the API, please let us know.


The rfmux API itself is currently compatible with two different end-to-end KIDs control software stacks:
- **hidfmux:** End-to-end KIDs characterization and astrophysics deployment developed by researchers at McGill University.
- **citkid:** Developed at JPL/Caltech. More details can be found at the [citkid GitHub repository](https://github.com/loganfoote/citkid).

## Documentation Philosophy

Our documentation lives in executable Jupyter notebooks, which offer interactive walkthroughs and examples. Currently, these notebooks reside in the `/home` directory, which also serves as the homepage for the rfmux Jupyter Hub.

---

For more details, visit our documentation or submit an issue for any specific inquiries.
