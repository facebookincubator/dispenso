# Dispenso
Dispenso is a library for working with sets of tasks.  It provides mechanisms for TaskSets, parallel for loops, etc...

## TODO

Find a more streamlined approach to obtaining and including dependencies.

## Setup build files
1. If you're building with benchmarks and tests, run setup scripts found in the setupScripts folder.
1. mkdir build
1. cd build
1. To create project build files run the following (by default it only builds the library):
   1. build library only: **cmake ..**
      * To build with tests add the argument: **-DBUILD_TESTS=1**
      * To build with tests add the argument: **-DBUILD_BENCHMARKS=1**

## Known issues
* On some platforms an error with folly stating "Folly::folly" includes non-existent path "//include", if this is the case navigate to installed folly-targets.cmake (typically found in /usr/local/lib/cmake/folly) and remove that entry. You may preemptively address this issue by changing the file from the cloned git files of Folly before running cmake.
* On Windows this project builds with MT flag; if MD is desired, changed to cmakelists and setupscripts will be needed.

## License

The library is released under the MIT license, but also relies on the moodycamel concurrentqueue library, which is released under the Simplified BSD and Zlib licenses.  See the top of the source at `dispenso/ext/moodycamel/*.h` for details.
