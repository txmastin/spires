cd ../..
make clean && make USE_CUDA=1
cd examples/spoken_digit_recognition
make clean && make USE_CUDA=1