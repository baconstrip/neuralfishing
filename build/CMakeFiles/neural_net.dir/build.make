# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/breeki/src/neuralfishing

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/breeki/src/neuralfishing/build

# Include any dependencies generated for this target.
include CMakeFiles/neural_net.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/neural_net.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/neural_net.dir/flags.make

CMakeFiles/neural_net.dir/neural/real_layer.cpp.o: CMakeFiles/neural_net.dir/flags.make
CMakeFiles/neural_net.dir/neural/real_layer.cpp.o: ../neural/real_layer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/breeki/src/neuralfishing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/neural_net.dir/neural/real_layer.cpp.o"
	/usr/bin/clang++-10  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/neural_net.dir/neural/real_layer.cpp.o -c /home/breeki/src/neuralfishing/neural/real_layer.cpp

CMakeFiles/neural_net.dir/neural/real_layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neural_net.dir/neural/real_layer.cpp.i"
	/usr/bin/clang++-10 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/breeki/src/neuralfishing/neural/real_layer.cpp > CMakeFiles/neural_net.dir/neural/real_layer.cpp.i

CMakeFiles/neural_net.dir/neural/real_layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neural_net.dir/neural/real_layer.cpp.s"
	/usr/bin/clang++-10 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/breeki/src/neuralfishing/neural/real_layer.cpp -o CMakeFiles/neural_net.dir/neural/real_layer.cpp.s

neural_net: CMakeFiles/neural_net.dir/neural/real_layer.cpp.o
neural_net: CMakeFiles/neural_net.dir/build.make

.PHONY : neural_net

# Rule to build all files generated by this target.
CMakeFiles/neural_net.dir/build: neural_net

.PHONY : CMakeFiles/neural_net.dir/build

CMakeFiles/neural_net.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/neural_net.dir/cmake_clean.cmake
.PHONY : CMakeFiles/neural_net.dir/clean

CMakeFiles/neural_net.dir/depend:
	cd /home/breeki/src/neuralfishing/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/breeki/src/neuralfishing /home/breeki/src/neuralfishing /home/breeki/src/neuralfishing/build /home/breeki/src/neuralfishing/build /home/breeki/src/neuralfishing/build/CMakeFiles/neural_net.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/neural_net.dir/depend
