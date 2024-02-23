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

layout(set = 0, binding = 2, std430) restrict readonly buffer DeltasData {
	float data[];
} deltas;

layout(set = 0, binding = 3, std430) restrict writeonly buffer WeightsData {
	float data[];
} weights;

layout(set = 0, binding = 4, std430) restrict readonly buffer StructureData {
	uint data[];
} layers;

layout(set = 0, binding = 5, std430) restrict readonly buffer OutputNeurons {
	float data[];
} neurons;

uint d2(uint x, uint y)
{
	return x * data.neurons_max + y;
}

uint d3(uint x, uint y, uint z)
{
	return x * data.neurons_max * data.neurons_max + y * data.neurons_max + z;
}

void main() {
	uint neuron = gl_WorkGroupID.x * 64 + gl_LocalInvocationID.x * 8 + gl_LocalInvocationID.y;
	if( neuron < layers.data[data.layer] )
	{
		for(uint i = 0; i < layers.data[data.layer + 1]; ++i)
			weights.data[d3(data.layer, neuron, i)] -= data.learning_rate * neurons.data[d2(data.layer, neuron)] * deltas.data[d2(data.layer, i)];
	}
}

