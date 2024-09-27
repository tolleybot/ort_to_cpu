using System;
using System.Linq;
using System.Runtime.InteropServices;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

class Program
{
    // Define the required data types
    public enum cudaError : int
    {
        cudaSuccess = 0,
        // ... other error codes ...
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
        }

        using var session = new InferenceSession("model.onnx", sessionOptions);

        // Create input data (for example, a float array)
        var input = new float[4] { 1.0f, 2.0f, 3.0f, 4.0f };
        
        // Bind input data
        var inputTensor = new DenseTensor<float>(input, new int[] { 4 });
        var inputValue = OrtValue.CreateTensorValueFromMemory<float>(
            OrtMemoryInfo.DefaultInstance, inputTensor.Buffer, new long[] { 4 });

        // Create input/output bindings
        var binding = session.CreateIoBinding();

        // Bind input
        binding.BindInput("input", inputValue);

        long[] outputShape = { 4 };
        int outputSizeInBytes = 4 * sizeof(float);
        OrtMemoryInfo memInfo = new OrtMemoryInfo
        ("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
        OrtExternalAllocation externalAllocation = new OrtExternalAllocation(memInfo, outputShape, TensorElementType.Float, devicePtr, outputSizeInBytes);
  
        // Allocate output on GPU and bind it
        // var alloc = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA,
        //     OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);

        // long[] ouptutShape = { 4 }; 
        // OrtValue tensorValue = OrtValue.CreateTensorValueWithData(memInfo,
        //                                                         TensorElementType.Float,
        //                                                         ouptutShape, devicePtr,
        //                                                         xLength * sizeof(float));

        binding.BindOutputToDevice("output", externalAllocation);

        // Run inference
        session.RunWithBinding(new RunOptions(), binding);

        // Retrieve output
        var outputs = binding.GetOutputValues();

        using OrtValue result = outputs.First();

        var outputData = result.GetTensorDataAsSpan<float>();

        // Assuming the first output is the one we need
        using OrtValue outputValue = outputs[0];

        // Now we need to copy the data from GPU to CPU using ILGPU
     //   float[] cpuData = CopyFromGpuToCpuWithILGPU(outputValue, 4);

        // Print the copied data
       // Console.WriteLine("Data copied to CPU: " + string.Join(", ", cpuData));
    }

    // ILGPU function to copy data from GPU to CPU

    // public static float[] CopyFromGpuToCpuWithILGPU(OrtValue outputValue, int dataSize)
    // {
    //     // Retrieve GPU memory pointer
    //     IntPtr gpuPointer = outputValue.TensorDataPointer;

    //     // Initialize ILGPU context and CUDA accelerator
    //     using var context = Context.CreateDefault();
    //     using var accelerator = new CudaAccelerator(context);

    //     // Allocate a buffer on the CPU
    //     float[] cpuData = new float[dataSize];

    //     // Wrap the GPU pointer in a MemoryBuffer using ILGPU interop
    //     var deviceBuffer = accelerator.LoadFromPointer<float>(gpuPointer, dataSize);

    //     // Copy data from the GPU to CPU
    //     deviceBuffer.CopyTo(cpuData, 0, 0, dataSize);

    //     // Return the data copied to CPU
    //     return cpuData;
    // }
}