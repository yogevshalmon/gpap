# Third-party material

> **Reader:** anyone using or redistributing GPAP — which files here are not GPAP's own, their
> licences, and what to keep with them.

GPAP's own code and documentation are under the MIT licence in [LICENSE](LICENSE). This
repository also contains benchmark files and a few quoted lines of code written by others, and
its scripts can download further tools. Those keep their own licences; the MIT licence does not
cover them.

The `gpap` Python package and the `libgpap` library contain no third-party code under a licence;
the public-domain algorithms they re-implement are listed under Dependencies.

## Included in this repository

These files are unmodified copies of their source, except that `b15.bench` lacks the final empty
line of the I99T original.

| files | what they are | source | licence |
|---|---|---|---|
| `benchmarks/circuits/bench/seq/b01.bench` to `b15.bench` | ITC'99 benchmark circuits (I99T), CAD Group, Politecnico di Torino | [cad-polito-it/I99T](https://github.com/cad-polito-it/I99T), commit `8a2c3b5` | EUPL-1.2, text in [LICENSE-ITC99.EUPL-1.2.txt](benchmarks/circuits/LICENSE-ITC99.EUPL-1.2.txt) |
| `benchmarks/circuits/epfl/` | EPFL Combinational Benchmark Suite, 20 circuits | [lsils/benchmarks](https://github.com/lsils/benchmarks), commit `82d8cc6` | MIT, notice in [LICENSE](benchmarks/circuits/epfl/LICENSE) |
| `benchmarks/sampling/` | 29 instances of the standard SAT-sampling suite, from Sketch program-synthesis problems | the `Benchmarks/` folder of [QuickSampler](https://github.com/RafaelTupynamba/quicksampler), commit `f9a3e09` | BSD-3-Clause, the licence of that repository, notice in [LICENSE-QuickSampler.BSD-3-Clause.txt](benchmarks/sampling/LICENSE-QuickSampler.BSD-3-Clause.txt) |
| `benchmarks/circuits/bench/comb/`, `benchmarks/circuits/bench/seq/s*.bench` | ISCAS-85 and ISCAS-89 benchmark circuits | the classic `.bench` distribution | none stated |
| `benchmarks/circuits/hwmcc17/` | 7 circuits from the HWMCC 2017 single-property set | [fmv.jku.at/hwmcc17](https://fmv.jku.at/hwmcc17/) | none stated |

"None stated" means that neither the files nor their distributors give a licence. These are
standard research benchmarks, included so that published experiments can be reproduced. If you
hold rights in any of them and want them removed, please open an issue.

The ITC'99 circuits are under the EUPL, a copyleft licence. If you redistribute them, keep the
licence text and the notices inside the files with them, and keep them under the EUPL. Their
source is the I99T repository above.

Two patch files quote a few lines of the code they change, so that they can be applied:

- `scripts/cmsgen_gpap_eval.patch` quotes CMSGen's `src/main.cpp` and `src/main.h`, which are
  under the MIT licence; the notice is at the end of this file.
- `eval/rivals/recurrent_gpap_eval.patch` quotes about 30 lines of `src/pytorch/run.py` from
  [Recurrent-CircuitSAT-Sampler](https://github.com/arashardakani/Recurrent-CircuitSAT-Sampler).
  That project states no licence, and its authors keep all rights in those lines.

## Downloaded by scripts, not included

The scripts fetch these from upstream at a fixed commit (the HWMCC'17 archive by checksum). This
repository does not redistribute them.

| component | script | upstream | licence |
|---|---|---|---|
| ITC'99 `b17` to `b22` | `scripts/fetch_benchmarks.sh` | I99T, commit `8a2c3b5` | EUPL-1.2 |
| the 20 EPFL circuits (the same as committed) and the whole HWMCC'17 set | `scripts/fetch_benchmarks.sh` | as above | MIT; none stated |
| ABC, the optional circuit encoder | `scripts/build_abc.sh` | [berkeley-abc/abc](https://github.com/berkeley-abc/abc), commit `bcfdf59` | the University of California's permissive licence, in its `copyright.txt` |
| CMSGen, the CPU sampler used for comparison | `scripts/build_cmsgen.sh` | [meelgroup/cmsgen](https://github.com/meelgroup/cmsgen), commit `8210aa9` | MIT |
| DEMOTIC, Recurrent-CircuitSAT-Sampler and High-Throughput-SAT-Sampler, the GPU samplers used for comparison | `eval/rivals/rivals_setup.sh` | [github.com/arashardakani](https://github.com/arashardakani), commits `f4e8ede`, `b9b048a`, `5cf5888` | none stated |
| Barbarik, the uniformity test, with SPUR, the reference sampler it ships as a binary | by hand, as described in `eval/uniformity/barbarik_gpap.py` | [meelgroup/barbarik](https://github.com/meelgroup/barbarik), commit `2c0a669`; [ZaydH/spur](https://github.com/ZaydH/spur) | MIT (both); Barbarik's other bundled binaries have their own licences |

The reproduction archive published with the paper contains this repository, the paper's campaign
data and ITC'99 `b17` to `b22`, unmodified and beside the EUPL text. It contains no third-party
tools.

## Dependencies

Installed by you, not redistributed here:

- Python packages: PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), python-sat (MIT), matplotlib
  (the Matplotlib licence), pytest (MIT); for the comparison samplers also pycryptosat (MIT),
  pandas (BSD-3-Clause) and tqdm (MPL-2.0 and MIT).
- To build: Triton (MIT), which compiles the GPU kernels and includes NVIDIA's `ptxas` and CUDA
  headers under NVIDIA's licence; CMake and a C++17 compiler.
- To run: the NVIDIA driver (`libcuda`), under NVIDIA's licence.

The random-number generators in `cpp/src/sample.cpp` and `cpp/kernels/bcp.py` re-implement
xoshiro256** and splitmix64 by David Blackman and Sebastiano Vigna, whose reference code is in
the public domain.

If you use a benchmark suite, please cite it:
[benchmarks/circuits/README.md](benchmarks/circuits/README.md) and
[benchmarks/sampling/README.md](benchmarks/sampling/README.md) list the papers.

## CMSGen notice

For the lines of CMSGen quoted in `scripts/cmsgen_gpap_eval.patch`:

    src/main.cpp: Copyright (c) 2010-2015 Mate Soos
    src/main.h:   Copyright (c) 2016, Mate Soos

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
