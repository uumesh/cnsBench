# cnsbench
External cnsbench app for performance benchmarking

This is a benchmarking application to investigate core kernels used in CFD solvers using DG methods. Particularly, it is built using the libParanumal library that provides the implementations and methods to setup the DG solver. All the compute kernels and setup are based on the CNS solver within libParanumal.

The libParanumal library can be obtained here : https://github.com/paranumal/libparanumal

References:
1. Chalmers, N., Karakus, A., Austin, A. P., Swirydowicz, K., & Warburton, T. (2020). libParanumal: a performance portable high-order finite element library. Release 0.4. 0.
