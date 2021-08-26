# Specify the parent image from which we build
FROM stereolabs/zed:3.5-gl-devel-cuda11.1-ubuntu20.04

# Set the working directory
WORKDIR /app

COPY install-packages.sh .

# Build the application with cmake
RUN ./install-packages.sh


mkdir /app/hello_zed_src/build && cd /app/hello_zed_src/build && \
    cmake -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
      -DCMAKE_CXX_FLAGS="-Wl,--allow-shlib-undefined" .. && \
    make

# Run the application
CMD ["/app/hello_zed_src/build/ZED_Tutorial_1"]