project( block_match )
find_package( OpenCV REQUIRED)
find_package( OpenCL REQUIRED )
add_executable( block_match Main.cpp)
target_link_libraries( block_match ${OpenCV_LIBS})
target_link_libraries( block_match ${OPENCL_LIBRARIES} )

