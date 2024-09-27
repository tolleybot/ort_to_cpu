import onnx
import onnx.helper as helper
from onnx import TensorProto
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

# Define the input tensor (with shape [4])
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4])

# Define the output tensor (with shape [4])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4])

# Define the constant multiplier
constant_value = 2.0  # Example: multiply by 2
constant_node = helper.make_node(
    'Constant',
    inputs=[],
    outputs=['const'],
    value=helper.make_tensor(
        name='const_tensor',
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[constant_value]
    )
)

# Define the node that multiplies input by the constant
multiply_node = helper.make_node(
    'Mul',  # Operation: multiply
    inputs=['input', 'const'],  # Input and constant
    outputs=['output']  # Output
)

# Create the graph
graph_def = helper.make_graph(
    [constant_node, multiply_node],  # Nodes in the graph
    'SimpleMultiplyModel',  # Model name
    [input_tensor],  # Inputs
    [output_tensor]  # Outputs
)

# Create the model
model_def = helper.make_model(graph_def, producer_name='onnx-example')

# Check if the model is valid
onnx.checker.check_model(model_def)

# Save the model to a file
onnx.save(model_def, 'model.onnx')

print("model.onnx has been created successfully.")