echo "Compiling PSA kernels by nvcc..."

nvcc -c -o ./src/PSAkernel.o ./src/PSAkernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_61

echo "Compiling PSAWapper..."

python build.py