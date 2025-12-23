Windows Instructions
====================

Windows is not a primary development or test platform for rfmux. So, while
we're not aware of any fundamental reason why it shouldn't work, you may be the
first to discover an accidental platform incompatibility.

Native (non-WSL)
----------------

The following is what worked for me (Feb. 2025).  You may prefer another method
of installing Python or may already have Python and/or Git installed. Feel free
to skip these steps as needed.

First, a few pre-requisites:

1. Install git (https://git-scm.com/downloads/win)
2. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
3. Install Visual Studio Community edition (if building from source)

Now, you should be able to open up a DOS box (cmd.exe), and run:

```
> cd \path\to\rfmux
rfmux> uv pip install -e .
rfmux> uv run ipython
```

This will launch an interactive Python session. You should be able to run:

```
In [1]: import rfmux
In [2]: s = rfmux.load_session('!HardwareMap [ !CRS { serial: "0033" } ]')
In [3]: d = s.query(rfmux.CRS).one()
In [4]: await d.resolve()
```

You will need to substitute your board's serial number in the transcript above.
You should now be able to interact with the board as in a Linux environment:

```
In [5]: x = await d.get_samples(10, channel=1, module=1)
```

It's also worth trying `py_get_samples`, because this function interacts with
the Windows network stack directly and may cause Windows to ask for permission
to access the network:

```
In [6]: x = await d.py_get_samples(10, channel=1, module=1)
```

If you've gotten this far, it's likely that the rest of rfmux should work fine.
Again, Windows is not our primary development platform - which means you might
discover problems we haven't seen in our release testing. If you discover a
problem, please let us know and we'll try to fix it.

## Increase Buffer Size Windows ##
To set parameters like `DefaultReceiveWindow`, `MaximumBufferSize` and `MaximumDynamicBufferSize`
- Press Win+R, type regedit, hit Enter.
- Navigate to:
`HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\Afd\Parameters`
- Right-click → New → DWORD (32-bit) Value → name it `DefaultReceiveWindow`.
- Double-click it, set Base to Decimal, then set Value data to 67108864 (64 MB).
- Repeat for `MaximumBufferSize` (and `MaximumDynamicBufferSize`).
- Close Registry Editor and reboot for changes to take effect.
