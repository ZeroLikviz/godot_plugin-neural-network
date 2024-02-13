#[compute]
#version 450

const int RUN = 0;
const int FIND_DELTAS = 1;
const int TRAIN = 2;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
const mediump uint max_x = gl_WorkGroupSize.x + 1;

layout(set = 0, binding = 0, std430) restrict buffer CommonData {
	float learning_rate;
	uint  last_layer;
	uint layers_size;
	bool bias;
	uint mode;
} data;

layout(set = 0, binding = 1, std430) restrict buffer BiasData {
	float data[gl_WorkGroupSize.x][gl_WorkGroupSize.y];
} biases;

layout(set = 0, binding = 2, std430) restrict buffer DeltasData {
	float data[gl_WorkGroupSize.x][gl_WorkGroupSize.y];
} deltas;

layout(set = 0, binding = 3, std430) restrict buffer WeightsData {
	float data[max_x][gl_WorkGroupSize.y][gl_WorkGroupSize.y];
} weights;

layout(set = 0, binding = 4, std430) restrict buffer StructureData {
	int data[];
} layers;

layout(set = 0, binding = 5, std430) restrict buffer OutputNeurons {
	float data[max_x][gl_WorkGroupSize.y];
} output_neurons;

layout(set = 0, binding = 6, std430) restrict buffer DesiredOutput {
	float data[gl_WorkGroupSize.y];
} desired_output;

float f(float x)
{
	return 1.0 / pow(2.71828, -x) + 1.0;
}

float fd(float x)
{
	return f(x) * (1.0 - f(x));
}

void main() {
	mediump uint x = gl_WorkGroupID.x;
	mediump uint y = gl_WorkGroupID.y;
	switch( data.mode )
	{
		case RUN:
			x += 1;
			if(y < layers.data[x])
			{
				output_neurons.data[x][y] = biases.data[gl_WorkGroupID.x][y] * float(data.bias);
				for(mediump uint i = 0; i < layers.data[x - 1]; ++i)
				{
					output_neurons.data[x][y] += output_neurons.data[x - 1][i] * weights.data[x - 1][i][y];
				}
				output_neurons.data[x][y] = f(output_neurons.data[x][y]);
			}
			break;
		case FIND_DELTAS:
			x = gl_WorkGroupSize.x - x;
			if(y < layers.data[x])
			{
				if(x == gl_WorkGroupSize.x)
				{
					deltas.data[x][y] = (output_neurons.data[x][y] - desired_output.data[y]) * fd(output_neurons.data[x][y]);
					return;
				}
				for(mediump uint i = 0; i < layers.data[x + 1]; ++i)
				{
					deltas.data[x][y] += deltas.data[x + 1][i] * weights.data[x][y][i];
				}
				deltas.data[x][y] = fd(deltas.data[x][y]);
			}
			
			break;
		case TRAIN:
			if(y < layers.data[x])
			{
				for(mediump uint i = 0; i < layers.data[x + 1]; ++i)
				{
					weights.data[x][y][i] -= data.learning_rate * output_neurons.data[x][y] * deltas.data[x + 1][i];
				}
			}
			break;
	}
}
