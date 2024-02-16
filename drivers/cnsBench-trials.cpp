/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "cnsBench.hpp"
#include "bench.hpp"

#include<iostream>
#include<chrono>
#include<random>
using namespace libp;

int main(int argc, char **argv)
{
  const char* benchmark_kernel[20] = {"SurfaceHex3D","VolumeHex3D", "GradSurfaceHex3D"};

  // start up MPI
  Comm::Init(argc, argv);

  comm_t comm(Comm::World().Dup());

  //if(argc!=2)
  //  LIBP_ABORT(string("Usage: ./cnsBench benchmark"));

  int iopt = std::atoi(argv[1]);
  std::string NX(argv[2]);
  std::string Norder(argv[3]);

  //create default settings
  platformSettings_t platformSettings(comm);
  meshSettings_t meshSettings(comm);
  cnsSettings_t cnsSettings(comm);

  // set up platform
  platformSettings.changeSetting("THREAD MODEL","CUDA");
  platformSettings.changeSetting("CACHE DIR", DCNS "/cache");
  platform_t platform(platformSettings);

  // set up mesh
  meshSettings.changeSetting("MESH DIMENSION","3");	// 3D elements
  meshSettings.changeSetting("ELEMENT TYPE","12");	// Hex elements
  meshSettings.changeSetting("BOX BOUNDARY FLAG","-1");	// Periodic
  meshSettings.changeSetting("POLYNOMIAL DEGREE",Norder); //argv[2]);
  meshSettings.changeSetting("BOX GLOBAL NX",NX);
  meshSettings.changeSetting("BOX GLOBAL NY",NX);
  meshSettings.changeSetting("BOX GLOBAL NZ",NX);
  mesh_t mesh(platform, meshSettings, comm);
  
  // Setup cns solver
  cnsSettings.changeSetting("DATA FILE","data/cnsGaussian3D.h");
  cns_t cns = cns_t(platform, mesh, cnsSettings);

  //Specify solver settings -> cnsSetup
  cns.mu = 0.01;
  cns.gamma = 1.4;
  cns.cubature = 0;	// 1 = Cubature, 0 = Others (Collocation)
  cns.isothermal = 0;  // 1 = True, 0 = False

  cns.Nfields = 4;	// 4 = 3D, 3 = 2D
  cns.Ngrads = 3*3;

  if (cns.cubature) {
    mesh.CubatureSetup();
    mesh.CubaturePhysicalNodes();
  }

  if(!cns.isothermal) cns.Nfields++;
  
  // From CNS Setup
  int Nelements = mesh.Nelements;
  int Np = mesh.Np;
  int Nfields = cns.Nfields;
  dlong NlocalFields = mesh.Nelements*mesh.Np*cns.Nfields;
  dlong NhaloFields  = mesh.totalHaloPairs*mesh.Np*cns.Nfields;
  dlong NlocalGrads = mesh.Nelements*mesh.Np*cns.Ngrads;
  dlong NhaloGrads  = mesh.totalHaloPairs*mesh.Np*cns.Ngrads;

  // Initialize required arrays
  std::random_device rd;   // Used to obtain a seed for the random number engine
  std::mt19937 gen(rd());  // Standard mersenne_twister engine seeded with rd
  std::uniform_real_distribution<dfloat> distribution(-1.0,1.0);

  // cns.q = (dfloat*) calloc(NlocalFields+NhaloFields, sizeof(dfloat));
  // cns.gradq = (dfloat*) calloc(NlocalGrads+NhaloGrads, sizeof(dfloat));
  cns.q.calloc(NlocalFields+NhaloFields);
  cns.gradq.calloc(NlocalGrads+NhaloGrads);
  
  for(int e=0;e<Nelements;++e) {
    for(int n=0;n<Np;++n) {
      dlong id = e*Np*Nfields + n;
      cns.q[id+0*Np] = 1.0 + distribution(gen); // rho
      cns.q[id+1*Np] = distribution(gen); // rho*u
      cns.q[id+2*Np] = distribution(gen); // rho*v
      cns.q[id+3*Np] = distribution(gen); // rho*w
      cns.q[id+4*Np] = 1.0 + distribution(gen); // rho*etotal
    }
  }

  cns.o_q = platform.malloc<dfloat>((NlocalFields+NhaloFields),cns.q);
  cns.o_gradq = platform.malloc<dfloat>((NlocalGrads+NhaloGrads),cns.gradq);

  occa::properties kernelInfo = mesh.props;
  std::string dataFileName = "data/cnsGaussian3D.h";
  kernelInfo["includes"] += dataFileName;
  kernelInfo["defines/" "p_Nfields"] = cns.Nfields;
  kernelInfo["defines/" "p_Ngrads"]  = cns.Ngrads;

  // Work-block parameters
  int blockMax = 512;
  int NblockV = std::max(1, blockMax/mesh.Np);
  kernelInfo["defines/" "p_NblockV"]= NblockV;

  int maxNodes = std::max(mesh.Np, (mesh.Nfp*mesh.Nfaces));
  int NblockS = std::max(1, blockMax/maxNodes);
  kernelInfo["defines/" "p_NblockS"]= NblockS;

  // Kernel setup
  memory<dfloat> rhsq;
  deviceMemory<dfloat> o_rhsq = platform.malloc<dfloat>(NlocalFields+NhaloFields);
  rhsq.calloc(NlocalFields+NhaloFields);

  char fileName[BUFSIZ], kernelName[BUFSIZ];
  sprintf(fileName, DCNS "/okl/cnsVolumeHex3D.okl");
  sprintf(kernelName, "cnsVolumeHex3D");
  cns.volumeKernel = platform.buildKernel(fileName, kernelName, kernelInfo);
  sprintf(fileName, DCNS "/okl/cnsGradVolumeHex3D.okl");
  sprintf(kernelName, "cnsGradVolumeHex3D");
  cns.gradVolumeKernel = platform.buildKernel(fileName, kernelName, kernelInfo);
  std::cout<<" Baseline kernel(s) built ...\n";

  sprintf(fileName, DCNS "/okl/cnsVolumeHex3D.okl");
  sprintf(kernelName, "cnsVolumeHex3D");
  kernel_t test_kernel = platform.buildKernel(fileName, kernelName, kernelInfo);
  std::cout<<" Test kernel built ...\n";

  dfloat simulation_time = 0.0;

  // =================================================================================>
  // Validate test kernel results
  memory<dfloat> baseline_result, optimized_result;
  baseline_result.calloc(NlocalFields+NhaloFields);
  optimized_result.calloc(NlocalFields+NhaloFields);
  // dfloat *optimized_result = (dfloat*) calloc(NlocalFields+NhaloFields,sizeof(dfloat));

  for(int i=0;i<NlocalFields+NhaloFields;++i) baseline_result[i] = 0.0;
  o_rhsq.copyFrom(baseline_result);
  platform.device.finish();
  cns.gradVolumeKernel(mesh.Nelements, mesh.o_vgeo, mesh.o_D, cns.o_q, cns.o_gradq);
  platform.device.finish();
  cns.volumeKernel(mesh.Nelements, mesh.o_vgeo, mesh.o_D, mesh.o_x, mesh.o_y, mesh.o_z,
                   simulation_time, cns.mu, cns.gamma, cns.o_q, cns.o_gradq, o_rhsq);
  platform.device.finish();
  o_rhsq.copyTo(baseline_result);
  platform.device.finish();
  std::cout<<" Baseline kernel run ...\n";

  for(int i=0;i<NlocalFields+NhaloFields;++i) optimized_result[i] = 0.0;
  o_rhsq.copyFrom(optimized_result);
  platform.device.finish();
  cns.gradVolumeKernel(mesh.Nelements, mesh.o_vgeo, mesh.o_D, cns.o_q, cns.o_gradq);
  platform.device.finish();
  // cns.volumeKernel(mesh.Nelements, mesh.o_vgeo, mesh.o_D, mesh.o_x, mesh.o_y, mesh.o_z,
  //                  simulation_time, cns.mu, cns.gamma, cns.o_q, cns.o_gradq, o_rhsq);
  // platform.device.finish();
  test_kernel(mesh.Nelements, mesh.o_vgeo, mesh.o_D, mesh.o_x, mesh.o_y, mesh.o_z,
              simulation_time, cns.mu, cns.gamma, cns.o_q, cns.o_gradq, o_rhsq);
  o_rhsq.copyTo(optimized_result);
  platform.device.finish();
  std::cout<<" Test kernel run ...\n";

  if(benchmark::validate(baseline_result.ptr(),optimized_result.ptr(),NlocalFields+NhaloFields))
       std::cout<<" Validation check passed...\n";

  // =================================================================================>
  // Run kernels and measure runtimes
  size_t ntrials = iopt;
  std::vector<double> walltimes(ntrials);
  // Baseline kernel(s)
  for(size_t trial{}; trial < ntrials; ++trial) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // cns.gradVolumeKernel(mesh.Nelements, mesh.o_vgeo, mesh.o_D, cns.o_q, cns.o_gradq);
    cns.volumeKernel(mesh.Nelements, mesh.o_vgeo, mesh.o_D, mesh.o_x, mesh.o_y, mesh.o_z,
                    simulation_time, cns.mu, cns.gamma, cns.o_q, cns.o_gradq, o_rhsq);
    platform.device.finish();

    auto finish_time = std::chrono::high_resolution_clock::now();
    walltimes[trial] = std::chrono::duration<double,std::milli>(finish_time-start_time).count();
  }

  auto baseline_stats = benchmark::calculateStatistics(walltimes);

  // Modified kernels
  for(size_t trial{}; trial < ntrials; ++trial) {
    auto start_time = std::chrono::high_resolution_clock::now();

    test_kernel(mesh.Nelements, mesh.o_vgeo, mesh.o_D, mesh.o_x, mesh.o_y, mesh.o_z,
              simulation_time, cns.mu, cns.gamma, cns.o_q, cns.o_gradq, o_rhsq);
    platform.device.finish();

    auto finish_time = std::chrono::high_resolution_clock::now();
    walltimes[trial] = std::chrono::duration<double,std::milli>(finish_time-start_time).count();
  }

  auto optimized_stats = benchmark::calculateStatistics(walltimes);

  // =================================================================================>
  // Print results
  std::cout<<" BENCHMARK:\n";
  std::cout<<" - Kernel Name : "<<std::string(benchmark_kernel[1])<<std::endl;
  std::cout<<" - Backend API : "<<platform.device.mode()<<std::endl;
  std::cout<<" PARAMETERS :\n";
  std::cout<<" - Number of elements : "<<mesh.Nelements<<std::endl;
  std::cout<<" - Polynomial degree  : "<<mesh.N<<std::endl;
  std::cout<<" - Number of trials   : "<<ntrials<<std::endl;
  std::cout<<" RUNTIME STATISTICS:\n";
  std::cout<<" - SPEEDUP : "<<baseline_stats.mean/optimized_stats.mean<<"\n";
  std::cout<<" - Mean : "<<std::scientific<<baseline_stats.mean   <<"   "<<optimized_stats.mean   <<" ms\n";
  std::cout<<" - Min  : "<<std::scientific<<baseline_stats.min    <<"   "<<optimized_stats.min    <<" ms\n";
  std::cout<<" - Max  : "<<std::scientific<<baseline_stats.max    <<"   "<<optimized_stats.max    <<" ms\n";
  std::cout<<" - Stdv : "<<std::scientific<<baseline_stats.stddev <<"   "<<optimized_stats.stddev <<" ms\n";
  std::cout<<std::endl;

  // close down MPI
  // Comm::Finalize();
  return LIBP_SUCCESS;
}
