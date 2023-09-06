# What is DROPStomo?
DROPStomo is a Python package for performing state and process tomography in the context of finite-dimensional Wigner representations on near-term quantum devices. This is a phase-space tomography approach to recovering the finite-dimensional Wigner type representations of quantum states and processes, with a particular focus on the [DROPS](https://spindrops.org/) (Discrete Representation of OPeratorS) representation. The package is based on the paper: [Wigner State and Process Tomography on Near-Term Quantum Devices](https://arxiv.org/abs/2302.12725).

This git repository contains the source code and examples for the package. 

This package is based on the Qiskit framework and can be easily adapted for other frameworks. The package can be plugged directly into quantum simulators or quantum devices.  

## Installation
```bash
pip install DROPStomo
```
## How to use
Currently, Wigner state tomography is available for single and two qubit systems, whereas Wigner process tomography is available for single qubit systems. For detailed examples on how to use Wigner tomography, please see the [Examples](/Examples) folder.  

## Citation
If you use the DROPS tomography tool in your work, cite it as follows:
```bash
@misc{https://doi.org/10.48550/arxiv.2302.12725,
  doi = {10.48550/ARXIV.2302.12725},
  url = {https://arxiv.org/abs/2302.12725},
  author = {Devra, Amit and Glaser, Niklas J. and Huber, Dennis and Glaser, Steffen J.},
  keywords = {Quantum Physics (quant-ph), FOS: Physical sciences, FOS: Physical sciences},
  title = {Wigner State and Process Tomography on Near-Term Quantum Devices},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
