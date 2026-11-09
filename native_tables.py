"""Generate scattering tables for an alloy with CHIMERA's own ab initio engine.

The native counterpart of feff8l_tables.py: same arguments, same cluster,
same output format (feffNNNN.dat + meta.json in data/<name>/), but no FEFF
binaries and no larch. See exafs_gpu/abinitio/ for the physics.

    python native_tables.py --elements Cr Co Ni --d-nn 2.517 --name crconi_fcc2517_native
    python native_tables.py --elements Cr Co Ni --d-nn 2.517 \\
        --name crconi_fcc2517_native_ms --rpath 5.3 --nleg 4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exafs_gpu.abinitio.tables import main  # noqa: E402

if __name__ == "__main__":
    main()
