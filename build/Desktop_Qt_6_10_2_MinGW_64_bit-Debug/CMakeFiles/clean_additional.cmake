# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles\\ProyectoQt_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\ProyectoQt_autogen.dir\\ParseCache.txt"
  "ProyectoQt_autogen"
  )
endif()
