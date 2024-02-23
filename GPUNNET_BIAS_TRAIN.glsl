#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer CommonData {
	float learning_rate;
	uint  last_layer;
	uint layers_size;
	uint neurons_max;
	uint layer;
	float bias;
	uint mode;
} data;

layout(set = 0, binding = 1, std430) restrict writeonly buffer BiasData {
	float data[];
} biases;

layout(set = 0, binding = 2, std430) restrict readonly buffer DeltasData {
	float data[];
} deltas;

layout(set = 0, binding = 4, std430) restrict readonly buffer StructureData {
	uint data[];
} layers;

uint d2(uint x, uint y)
{
	return x * data.neurons_max + y;
}

void main() {
	uint neuron = gl_WorkGroupID.x * 64 + gl_LocalInvocationID.x * 8 + gl_LocalInvocationID.y;
	if( neuron < layers.data[data.layer + 1] )
		biases.data[d2(data.layer, neuron)] -= data.learning_rate * deltas.data[d2(data.layer, neuron)];
}

