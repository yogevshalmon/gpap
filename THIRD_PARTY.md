# Third-party material

> **Reader:** anyone using or redistributing GPAP — which files here are not GPAP's own, their
> licences, and what to keep with them.

GPAP's own code and documentation are MIT ([LICENSE](LICENSE)). The `gpap` package and `libgpap`
contain no third-party code. Everything below keeps its own licence, which the MIT licence does
not cover.

## In this repository

Unmodified copies of their source (`b15.bench` lacks the original's final empty line).

| files | what | source | licence |
|---|---|---|---|
| `benchmarks/circuits/bench/seq/b01`–`b15.bench` | ITC'99 circuits (I99T, Politecnico di Torino) | [cad-polito-it/I99T](https://github.com/cad-polito-it/I99T) `8a2c3b5` | [EUPL-1.2](benchmarks/circuits/LICENSE-ITC99.EUPL-1.2.txt) |
| `benchmarks/circuits/epfl/` | EPFL combinational suite, 20 circuits | [lsils/benchmarks](https://github.com/lsils/benchmarks) `82d8cc6` | [MIT](benchmarks/circuits/epfl/LICENSE) |
| `benchmarks/sampling/` | 29 SAT-sampling instances (Sketch problems) | `Benchmarks/` of [QuickSampler](https://github.com/RafaelTupynamba/quicksampler) `f9a3e09` | [BSD-3-Clause](benchmarks/sampling/LICENSE-QuickSampler.BSD-3-Clause.txt) |
| `benchmarks/circuits/bench/comb/`, `benchmarks/circuits/bench/seq/s*.bench` | ISCAS-85 and ISCAS-89 circuits | the classic `.bench` distribution | none stated |
| `benchmarks/circuits/hwmcc17/` | 7 HWMCC 2017 single-property circuits | [fmv.jku.at/hwmcc17](https://fmv.jku.at/hwmcc17/) | none stated |

- **None stated:** neither the files nor their distributors give a licence. They are standard
  research benchmarks, kept so published experiments can be reproduced; if you hold rights in one
  and want it removed, open an issue.
- **ITC'99 is copyleft (EUPL).** Redistribute those circuits under the EUPL, with its text and the
  notices inside the files.
- **Two patches quote the code they change.** `scripts/cmsgen_gpap_eval.patch` quotes CMSGen's
  `src/main.cpp` and `src/main.h` (MIT, [notice below](#cmsgen-notice)).
  `eval/rivals/recurrent_gpap_eval.patch` quotes about 30 lines of `src/pytorch/run.py` from
  [Recurrent-CircuitSAT-Sampler](https://github.com/arashardakani/Recurrent-CircuitSAT-Sampler),
  which states no licence; its authors keep all rights in them.

## Fetched by scripts, not redistributed

Each is pinned to a commit (the HWMCC'17 archive to a checksum).

| component | script | upstream | licence |
|---|---|---|---|
| ITC'99 `b17`–`b22`; the EPFL and full HWMCC'17 sets | `scripts/fetch_benchmarks.sh` | as above | as above |
| ABC, the optional circuit encoder | `scripts/build_abc.sh` | [berkeley-abc/abc](https://github.com/berkeley-abc/abc) `bcfdf59` | University of California, permissive (`copyright.txt`) |
| CMSGen, the CPU sampler compared with | `scripts/build_cmsgen.sh` | [meelgroup/cmsgen](https://github.com/meelgroup/cmsgen) `8210aa9` | MIT |
| DEMOTIC, Recurrent-CircuitSAT-Sampler, High-Throughput-SAT-Sampler | `eval/rivals/rivals_setup.sh` | [arashardakani](https://github.com/arashardakani) `f4e8ede`, `b9b048a`, `5cf5888` | none stated |
| Barbarik (uniformity test) and SPUR, the sampler it ships as a binary | by hand, see `eval/uniformity/barbarik_gpap.py` | [meelgroup/barbarik](https://github.com/meelgroup/barbarik) `2c0a669`; [ZaydH/spur](https://github.com/ZaydH/spur) | MIT; Barbarik's other binaries have their own |

The paper's reproduction archive holds this repository, the campaign data and ITC'99 `b17`–`b22`
(unmodified, beside the EUPL text), and no third-party tools.

## In the Docker images

The images built from `Dockerfile` and `eval/artifact/Dockerfile`, the artifact's among them,
redistribute what they install, each under its own licence:

| component | image | licence |
|---|---|---|
| Ubuntu 24.04 packages (Python 3.12, g++, cmake, git, …) | both | each package's, under `/usr/share/doc/<package>/copyright` |
| PyTorch, Triton and the other Python packages, with the NVIDIA CUDA libraries in PyTorch's wheels | both | PyTorch BSD-3-Clause, Triton MIT, CUDA NVIDIA's (in each package); `/opt/gpap-image/pip-freeze.txt` lists them all |
| ABC, from the commit above | both | as above |
| CMSGen, from the commit above with `scripts/cmsgen_gpap_eval.patch` | artifact | MIT |

The GPU samplers state no licence, so no image contains them: the artifact's `setup` step clones
them.

## Dependencies you install

- **Python:** PyTorch and NumPy (BSD-3-Clause), python-sat and pytest (MIT), matplotlib (its own
  licence); for the comparison samplers, pycryptosat (MIT), pandas (BSD-3-Clause), tqdm (MPL-2.0
  and MIT).
- **To build:** Triton (MIT), which bundles NVIDIA's `ptxas` and CUDA headers under NVIDIA's
  licence; CMake; a C++17 compiler.
- **To run:** the NVIDIA driver (`libcuda`), under NVIDIA's licence.

The random-number generators in `cpp/src/sample.cpp` and `cpp/kernels/bcp.py` re-implement
xoshiro256** and splitmix64 (Blackman and Vigna, public domain).

Citing a benchmark suite: [benchmarks/circuits/README.md](benchmarks/circuits/README.md) and
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
