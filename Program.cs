using System;
using System.Linq;
using System.Runtime.InteropServices;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;


/*
 * Example: Using ONNX Runtime with CUDA for Inference with Pre-Allocated GPU Memory
 * 
 * This example demonstrates how to use ONNX Runtime with CUDA for running an inference session, where the input is provided
 * as a raw float array (CPU memory), and the output tensor is pre-allocated in GPU memory using `cudaMalloc`.
 * 
 * Key Highlights:
 * 1. **GPU Memory Allocation**: The program uses `cudaMalloc` to allocate memory for the output tensor directly on the GPU.
 * 2. **Input Tensor**: The input tensor is passed from CPU memory using a float array, wrapped as an `OrtValue` using the method `OrtValue.CreateTensorValueFromMemory`.
 * 3. **Output Tensor**: The output tensor is created from pre-allocated GPU memory using `OrtValue.CreateTensorValueWithData`, which allows ONNX Runtime to write the inference results directly to GPU memory.
 * 4. **Inference**: The ONNX Runtime session runs using the `Run` method, which takes the input and output `OrtValue` collections and processes the inference on the model.
 * 5. **Copying Output from GPU to CPU**: After inference, `cudaMemcpy` is used to copy the output tensor from GPU memory back to CPU memory for display or further processing.
 * 
 * This example is ideal for scenarios where you need to pre-allocate memory for performance reasons or have custom memory handling (e.g., in high-performance GPU-based applications).
 */

class Program
{
    public enum cudaError : int
    {
        cudaSuccess = 0,
        // ... other error codes can go here ...
    }

    public enum cudaMemcpyKind : int
    {
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4
    }

    [DllImport("cudart", EntryPoint = "cudaMalloc")]
    public static extern cudaError cudaMalloc(out IntPtr devPtr, int size);
    [DllImport("cudart", EntryPoint = "cudaMemcpy")]
    public static extern cudaError cudaMemcpy(float[] dst, IntPtr src, int size, cudaMemcpyKind kind);
    [DllImport("cudart", EntryPoint = "cudaFree")]
    public static extern int cudaFree(IntPtr devPtr);

    static void Main(string[] args)
    {
        // Initialize session with CUDA execution provider
        var sessionOptions = new SessionOptions();
        sessionOptions.AppendExecutionProvider_CUDA(0);  // Use GPU 0

        // Pointer to GPU memory for output
        IntPtr devicePtr; 
        cudaError cudaResult = cudaMalloc(out devicePtr, 4 * sizeof(float));
        if (cudaResult != cudaError.cudaSuccess)
        {
            Console.WriteLine("Failed to allocate device memory: {0}", cudaResult);
            return;
        }

        using var session = new InferenceSession("model.onnx", sessionOptions);

        // Create input data (for example, a float array)
        var input = new float[4] { 1.0f, 2.0f, 3.0f, 4.0f };

        // Bind input data
        var inputTensor = new DenseTensor<float>(input, new int[] { 4 });
        OrtValue inputValue = OrtValue.CreateTensorValueFromMemory<float>(
            OrtMemoryInfo.DefaultInstance, inputTensor.Buffer, new long[] { 4 });
 
        // Create OrtMemoryInfo for CUDA (GPU) memory
        using var ortMemoryInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);

        // Create an OrtValue from the GPU memory (output tensor) using CreateTensorValueWithData
        using var outputValue = OrtValue.CreateTensorValueWithData(
            ortMemoryInfo,                           // Memory info for GPU
            TensorElementType.Float,                 // Tensor element type
            new long[] { 4 },                        // Tensor shape (4 elements)
            devicePtr,                               // Pointer to the GPU memory
            sizeof(float) * 4                        // Total buffer size in bytes (4 floats)
        );


        var inputNames = session.InputMetadata.Keys.ToArray();
        var outputNames = session.OutputMetadata.Keys.ToArray();

        // Run the inference session, using the pre-allocated GPU memory for output
        session.Run(null, inputNames, new List<OrtValue> { inputValue }, outputNames, new List<OrtValue> { outputValue });

        // At this point, the inference has completed, and the results will be in `devicePtr`
        // If needed, copy the output back to the host (CPU)
        float[] outputData = new float[4];  // Allocate a buffer to hold the output
        cudaMemcpy(outputData, devicePtr, 4 * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost);
        
        // Print the output values
        Console.WriteLine("Inference output:");
        foreach (var value in outputData)
        {
            Console.WriteLine(value);
        }
        
        cudaFree(devicePtr);
    }
}