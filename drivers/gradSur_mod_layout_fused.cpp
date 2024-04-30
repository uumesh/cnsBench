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
  dlong NsurfaceNodes  = mesh.Nelements * mesh.Nfaces * mesh.Nfp;
  dlong NsurfaceFields = NsurfaceNodes * cns.Nfields;
  dlong NsurfaceGrads  = NsurfaceNodes * cns.Ngrads;

  memory<dfloat> q_packed, gradq_test;
  deviceMemory<dfloat> o_qpacked, o_gradq_packed, o_gradq_test;
  o_qpacked = platform.malloc<dfloat>(NsurfaceFields);
  o_gradq_packed = platform.malloc<dfloat>(NsurfaceGrads);
  o_gradq_test   = platform.malloc<dfloat>(NlocalGrads+NhaloGrads);
  q_packed.calloc(NsurfaceFields);
  gradq_test.calloc(NlocalGrads+NhaloGrads);

  kernelInfo["defines/" "p_Nfaces"] = mesh.Nfaces;
  kernelInfo["defines/" "p_Nfp"] = mesh.Nfp;
  kernelInfo["defines/" "p_Nq"]  = mesh.Nq;
  kernelInfo["defines/" "p_Nsp"] = mesh.Nfaces*mesh.Nfp;

  char fileName[BUFSIZ], kernelName[BUFSIZ];
  sprintf(fileName, DCNS "/okl/cnsGradSurfaceHex3D.okl");
  sprintf(kernelName, "cnsGradSurfaceHex3D");
  cns.gradSurfaceKernel = platform.buildKernel(fileName, kernelName, kernelInfo);
  std::cout<<" Baseline kernel(s) built ...\n";

  sprintf(fileName, DCNS "/experimental/gradSurface_mod_layout_fused.okl");
  kernel_t surfaceGrad   = platform.buildKernel(fileName, "surfaceGrad", kernelInfo);
  kernel_t surfacePack   = platform.buildKernel(fileName, "surfacePack", kernelInfo);
  kernel_t surfaceUnpack = platform.buildKernel(fileName, "surfaceUnpack", kernelInfo);
  // kernel_t surfaceSyncEdges   = platform.buildKernel(fileName,"surfaceSyncEdges",kernelInfo);
  // kernel_t surfaceSyncCorners = platform.buildKernel(fileName,"surfaceSyncCorners",kernelInfo);
  std::cout<<" Test kernel(s) built ...\n";

  // =================================================================================>
  // Kernel setup

  // Edges and corner synchronization
  memory<int> faceIndices, mapEdgeIndices, mapCornerIndices;
  faceIndices.calloc(mesh.Nfaces*mesh.Nfp);
  mapEdgeIndices.calloc(mesh.Nfaces*mesh.Nfp);
  mapCornerIndices.calloc(8*3);
  deviceMemory<int> o_mapEdgeIndices = platform.malloc<int>(mesh.Nfaces*mesh.Nfp);
  deviceMemory<int> o_mapCornerIndices = platform.malloc<int>(8*3);

  // Set volume indices for each surface node
  for(int f=0;f<mesh.Nfaces;f++) {
    //  for(int j=0;j<mesh.Nq;j++) {
    for(int i=0;i<mesh.Nfp;i++) {
      const int sid = f*mesh.Nfp + i; //j*mesh.Nq + i; // surface index
      const int vid = mesh.vmapM[sid]; // volume index
      faceIndices[sid] = vid;
      mapEdgeIndices[sid] = -100; // default value for non-edge nodes
    }
    //  }
  }

  // map surface node indices among shared nodes (intersecting faces)
  int corner = 0;
  for(int i=0;i<mesh.Nfaces*mesh.Nfp;i++) {
    for(int j=i+1;j<mesh.Nfaces*mesh.Nfp;j++) {
      if(faceIndices[i]==faceIndices[j]) {
        if(mapEdgeIndices[i]!=-1000)
          mapEdgeIndices[i] = j;
        for(int k=j+1;k<mesh.Nfaces*mesh.Nfp;k++) {
          if(faceIndices[k]==faceIndices[j]) {
            mapEdgeIndices[i] = -1*(corner+1);
            mapEdgeIndices[j] = -1000;
            mapEdgeIndices[k] = -1000;
            mapCornerIndices[corner]    = i;
            mapCornerIndices[corner+8]  = j;
            mapCornerIndices[corner+16] = k;
            corner++;
            break;
          }
        }
        break;
      }
    }
  }

  // std::cout<<"Number of corners mapped = "<<corner<<"\n";
  // for (int i=0;i<8;i++) {
  //   std::cout<<i<<" "<<mapCornerIndices[i]<<" "<<mapCornerIndices[i+8]<<" "<<mapCornerIndices[i+16]<<"\n";
  // }

  o_mapEdgeIndices.copyFrom(mapEdgeIndices); 
  o_mapCornerIndices.copyFrom(mapCornerIndices);

  dfloat simulation_time = 0.0;

  // =================================================================================>
  // Validate test kernel results
  memory<dfloat> baseline_result, modified_result;
  baseline_result.calloc(NlocalGrads+NhaloGrads);
  modified_result.calloc(NlocalGrads+NhaloGrads);

  // Baseline kernel
  for(int i=0;i<NlocalGrads+NhaloGrads;++i) baseline_result[i] = 0.0;
  cns.o_gradq.copyFrom(baseline_result);
  platform.device.finish();
  cns.gradSurfaceKernel(mesh.Nelements, mesh.o_sgeo, mesh.o_LIFT,	mesh.o_vmapM,	mesh.o_vmapP,	mesh.o_EToB,
			                  mesh.o_x,	mesh.o_y,	mesh.o_z,	simulation_time, cns.mu, cns.gamma,	cns.o_q, cns.o_gradq);
  platform.device.finish();
  cns.o_gradq.copyTo(baseline_result);
  platform.device.finish();
  std::cout<<" Baseline kernel run ...\n";

  // Modified kernel
  for(int i=0;i<NlocalGrads+NhaloGrads;++i) modified_result[i] = 0.0;
  // o_qpacked.copyFrom(q_packed);
  o_gradq_test.copyFrom(modified_result);
  platform.device.finish();

  // Pack
  surfacePack(mesh.Nelements, mesh.o_vmapM, mesh.o_vmapP, cns.o_q, o_qpacked);

  // Compute gradients
  surfaceGrad(mesh.Nelements, mesh.o_sgeo, mesh.o_LIFT, mesh.o_vmapM,	mesh.o_mapP,	mesh.o_EToB,
			        mesh.o_x,	mesh.o_y,	mesh.o_z,	simulation_time, cns.mu, cns.gamma,	o_qpacked, o_gradq_packed, o_mapEdgeIndices, o_mapCornerIndices);
  // Sync edges and corners data
  // surfaceSyncEdges(mesh.Nelements, o_mapEdgeIndices, o_gradq_packed);
  // surfaceSyncCorners(mesh.Nelements, o_mapCornerIndices, o_gradq_packed);

  // Transfer back to original layout for comparison
  surfaceUnpack(mesh.Nelements, mesh.o_vmapM, o_gradq_packed, o_gradq_test);
  platform.device.finish();

  o_gradq_test.copyTo(modified_result);
  platform.device.finish();
  std::cout<<" Test kernel run ...\n";

  if(benchmark::validate(baseline_result.ptr(),modified_result.ptr(),NlocalGrads+NhaloGrads))
       std::cout<<" Validation check passed...\n";

  // =================================================================================>
  // Run kernels and measure runtimes
  size_t ntrials = 10000;
  std::vector<double> walltimes(ntrials);
  // Baseline kernel(s)
  for(size_t trial{}; trial < ntrials; ++trial) {
    auto start_time = std::chrono::high_resolution_clock::now();

    cns.gradSurfaceKernel(mesh.Nelements, mesh.o_sgeo, mesh.o_LIFT,	mesh.o_vmapM,	mesh.o_vmapP,	mesh.o_EToB,
			                  mesh.o_x,	mesh.o_y,	mesh.o_z,	simulation_time, cns.mu, cns.gamma,	cns.o_q, cns.o_gradq);
    platform.device.finish();

    auto finish_time = std::chrono::high_resolution_clock::now();
    walltimes[trial] = std::chrono::duration<double,std::milli>(finish_time-start_time).count();
  }

  auto baseline_stats = benchmark::calculateStatistics(walltimes);

  // Modified kernels
  for(size_t trial{}; trial < ntrials; ++trial) {
    auto start_time = std::chrono::high_resolution_clock::now();

    surfaceGrad(mesh.Nelements, mesh.o_sgeo, mesh.o_LIFT, mesh.o_vmapM,	mesh.o_mapP,	mesh.o_EToB,
			        mesh.o_x,	mesh.o_y,	mesh.o_z,	simulation_time, cns.mu, cns.gamma,	o_qpacked, o_gradq_packed, o_mapEdgeIndices, o_mapCornerIndices);
    // Sync edges and corners data
    // surfaceSyncEdges(mesh.Nelements, o_mapEdgeIndices, o_gradq_packed);
    // surfaceSyncCorners(mesh.Nelements, o_mapCornerIndices, o_gradq_packed);
    platform.device.finish();

    auto finish_time = std::chrono::high_resolution_clock::now();
    walltimes[trial] = std::chrono::duration<double,std::milli>(finish_time-start_time).count();
  }

  auto optimized_stats = benchmark::calculateStatistics(walltimes);

  // =================================================================================>
  // Print results
  std::cout<<" BENCHMARK:\n";
  std::cout<<" - Kernel Name : "<<std::string(benchmark_kernel[iopt-1])<<std::endl;
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
