# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-8.3.1/cmake-3.23.1-ij35dzv4x2ql3uxn2n63ei4qr2uutjtu/bin/cmake

# The command to remove a file.
RM = /autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-8.3.1/cmake-3.23.1-ij35dzv4x2ql3uxn2n63ei4qr2uutjtu/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/build

# Include any dependencies generated for this target.
include CMakeFiles/cnsbench.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cnsbench.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cnsbench.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cnsbench.dir/flags.make

CMakeFiles/cnsbench.dir/cnsBench.cpp.o: CMakeFiles/cnsbench.dir/flags.make
CMakeFiles/cnsbench.dir/cnsBench.cpp.o: ../cnsBench.cpp
CMakeFiles/cnsbench.dir/cnsBench.cpp.o: CMakeFiles/cnsbench.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cnsbench.dir/cnsBench.cpp.o"
	/sw/summit/gcc/9.1.0-alpha+20190716/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnsbench.dir/cnsBench.cpp.o -MF CMakeFiles/cnsbench.dir/cnsBench.cpp.o.d -o CMakeFiles/cnsbench.dir/cnsBench.cpp.o -c /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/cnsBench.cpp

CMakeFiles/cnsbench.dir/cnsBench.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnsbench.dir/cnsBench.cpp.i"
	/sw/summit/gcc/9.1.0-alpha+20190716/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/cnsBench.cpp > CMakeFiles/cnsbench.dir/cnsBench.cpp.i

CMakeFiles/cnsbench.dir/cnsBench.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnsbench.dir/cnsBench.cpp.s"
	/sw/summit/gcc/9.1.0-alpha+20190716/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/cnsBench.cpp -o CMakeFiles/cnsbench.dir/cnsBench.cpp.s

CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.o: CMakeFiles/cnsbench.dir/flags.make
CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.o: ../src/cnsSettings.cpp
CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.o: CMakeFiles/cnsbench.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.o"
	/sw/summit/gcc/9.1.0-alpha+20190716/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.o -MF CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.o.d -o CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.o -c /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/src/cnsSettings.cpp

CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.i"
	/sw/summit/gcc/9.1.0-alpha+20190716/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/src/cnsSettings.cpp > CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.i

CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.s"
	/sw/summit/gcc/9.1.0-alpha+20190716/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/src/cnsSettings.cpp -o CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.s

# Object files for target cnsbench
cnsbench_OBJECTS = \
"CMakeFiles/cnsbench.dir/cnsBench.cpp.o" \
"CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.o"

# External object files for target cnsbench
cnsbench_EXTERNAL_OBJECTS =

cnsbench: CMakeFiles/cnsbench.dir/cnsBench.cpp.o
cnsbench: CMakeFiles/cnsbench.dir/src/cnsSettings.cpp.o
cnsbench: CMakeFiles/cnsbench.dir/build.make
cnsbench: /gpfs/alpine/scratch/umeshu/cfd144/occa/install/lib/libocca.so
cnsbench: /sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-9.1.0/spectrum-mpi-10.4.0.3-20210112-6jbupg3thjwhsabgevk6xmwhd2bbyxdc/lib/libmpiprofilesupport.so
cnsbench: /sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-9.1.0/spectrum-mpi-10.4.0.3-20210112-6jbupg3thjwhsabgevk6xmwhd2bbyxdc/lib/libmpi_ibm.so
cnsbench: CMakeFiles/cnsbench.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable cnsbench"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cnsbench.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cnsbench.dir/build: cnsbench
.PHONY : CMakeFiles/cnsbench.dir/build

CMakeFiles/cnsbench.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cnsbench.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cnsbench.dir/clean

CMakeFiles/cnsbench.dir/depend:
	cd /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/build /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/build /gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/cns-bench/build/CMakeFiles/cnsbench.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cnsbench.dir/depend

