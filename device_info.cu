#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <sstream>

// Host function to fetch and return GPU device info
std::string get_device_info() {
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    std::ostringstream info;
    info << "Device: " << props.name << "\n";
    info << "maxThreadsPerBlock: " << props.maxThreadsPerBlock << "\n";
    info << "maxThreadsDim:      ["
         << props.maxThreadsDim[0] << ", "
         << props.maxThreadsDim[1] << ", "
         << props.maxThreadsDim[2] << "]\n";
    info << "maxGridSize:        ["
         << props.maxGridSize[0] << ", "
         << props.maxGridSize[1] << ", "
         << props.maxGridSize[2] << "]\n";
    info << "multiProcessorCount: " << props.multiProcessorCount << "\n";

    return info.str();
}

int main() {
    std::string info = get_device_info();
    std::cout << info;
    return 0;
}