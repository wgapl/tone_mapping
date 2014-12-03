/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"

#define NUM_THREADS  1024

__global__ void reduce_min_kernel(const float* g_in, float* g_out, long int size)
{

    extern __shared__ float sdata_min[];

    int tid = threadIdx.x;
    int i = blockDim.x*blockIdx.x + tid;
    if(i>=size)
        return;
    sdata_min[tid] = g_in[i];
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){
        if(tid < s){
            sdata_min[tid] = min(sdata_min[tid], sdata_min[tid+s]);
        }
    __syncthreads();
    }

    if(tid == 0)
        g_out[blockIdx.x] = sdata_min[0];
}

__global__ void reduce_max_kernel(const float* g_in, float* g_out, long int size)
{

    extern __shared__ float sdata_min[];

    int tid = threadIdx.x;
    int i = blockDim.x*blockIdx.x + tid;
    if(i>=size)
        return;
    sdata_min[tid] = g_in[i];
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){
        if(tid < s){
            sdata_min[tid] = max(sdata_min[tid], sdata_min[tid+s]);
        }
    __syncthreads();
    }

    if(tid == 0)
        g_out[blockIdx.x] = sdata_min[0];
}

__global__ void histogram_kernel(const float* const d_logLuminance,
                          unsigned int* d_histogram,
                          float lumRange,
                          float lumMin,
                          int numBins)
{
    int myId = threadIdx.x + blockIdx.x*blockDim.x;
    int bin = (d_logLuminance[myId] - lumMin)/lumRange * numBins;
    if (bin == numBins){
        bin = bin-1; }
    atomicAdd(&(d_histogram[bin]),1);
}

__global__ void scan_kernel(unsigned int* out, 
                        unsigned int* in, 
                        const int n)
{
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    temp[tid] = (tid>0) ? in[tid-1] : 0;
    __syncthreads();
    for(int offset = 1; offset < n; offset *=2)
    {
        if(tid>=offset)
            temp[tid] += temp[tid - offset];
        else
            temp[tid] = temp[tid];
    }
    __syncthreads();
    out[tid] = temp[tid];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum*/
    float* d_inter_min;
    float* d_out_min;
    float* d_inter_max;
    float* d_out_max;

    const int maxThreads = 1024;
    int numBlocks = ((numCols*numRows)+maxThreads-1)/maxThreads;
    
    cudaMalloc((void**) &d_inter_min, numBlocks*sizeof(float));
    cudaMalloc((void**) &d_out_min, sizeof(float));

    cudaMalloc((void**) &d_inter_max, numBlocks*sizeof(float));
    cudaMalloc((void**) &d_out_max, sizeof(float));

    const dim3 gridSize = numBlocks;
    const dim3 blockSize = maxThreads;

    long int size1 = numRows*numCols;
    long int size2 = numBlocks;

    reduce_min_kernel<<<gridSize, blockSize, maxThreads*sizeof(float)>>>(d_logLuminance, d_inter_min, size1);
    reduce_min_kernel<<<1, gridSize, numBlocks*sizeof(float)>>>(d_inter_min, d_out_min, size2);

    cudaMemcpy(&min_logLum, d_out_min, sizeof(float), cudaMemcpyDeviceToHost);

    reduce_max_kernel<<<gridSize, blockSize, maxThreads*sizeof(float)>>>(d_logLuminance, d_inter_max, size1);
    reduce_max_kernel<<<1, gridSize, numBlocks*sizeof(float)>>>(d_inter_max, d_out_max, size2);

    cudaMemcpy(&max_logLum, d_out_max, sizeof(float), cudaMemcpyDeviceToHost);

    float lumRange;
    lumRange = max_logLum - min_logLum;
    
    /*
    2) subtract them to find the range*/
    //float lumRange = max_logLum-min_logLum;
    std::cout << "Max, Min" << std::endl;
    std::cout << max_logLum << ", " << min_logLum << std::endl;
    std::cout << "The range" << std::endl;
    std::cout << lumRange << std::endl;
    /*
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins*/ 
    
    const dim3 blockSizeHistogram(NUM_THREADS,1,1);
    const dim3 gridSizeHistogram( (numCols*numRows + blockSizeHistogram.x -1)/blockSizeHistogram.x,1,1);
    
    unsigned int *d_histogram;
    
    /* Move lumRange over to GPU
    checkCudaErrors(cudaMalloc((void**) &d_lumRange, sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_lumRange, h_lumRange,sizeof(float), cudaMemcpyHostToDevice));*/
    
    // Allocate memory for d_histogram and initialize to 0.
    checkCudaErrors(cudaMalloc((void**)&d_histogram,sizeof(unsigned int)*numBins));
    checkCudaErrors(cudaMemset(d_histogram,0,sizeof(unsigned int)*numBins));
    
    // Launch histogram_kernel on the Device
    histogram_kernel<<<gridSizeHistogram,blockSizeHistogram>>>(d_logLuminance,
                                                               d_histogram,
                                                               lumRange,
                                                               min_logLum,
                                                               numBins);
    
    /*
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    
    const dim3 blockSizeScan(numBins,1,1);
    const dim3 gridSizeScan(1,1,1);
    
    scan_kernel<<<gridSizeScan,blockSizeScan,sizeof(float)*blockSizeScan.x>>>(d_cdf, 
                                                                              d_histogram,
                                                                              numBins);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    /*
    checkCudaErrors(cudaFree(d_histogram));
    checkCudaErrors(cudaFree(d_max_inter));
    checkCudaErrors(cudaFree(d_max));
    checkCudaErrors(cudaFree(d_min_inter));
    checkCudaErrors(cudaFree(d_min));*/
}

