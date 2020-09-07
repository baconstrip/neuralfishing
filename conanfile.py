from conans import ConanFile, CMake

class Fishy(ConanFile):
    requires = "eigen/3.3.7"
    generators = "cmake"
    