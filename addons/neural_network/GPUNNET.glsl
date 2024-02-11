#[compute]
#version 450

const int RUN = 0;
const int FIND_DELTAS = 1;
const int TRAIN = 2;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict buffer CommonData {
	float learning_rate;
	int  last_layer;
	int layers_size;
	int mode;
} data;

layout(set = 0, binding = 1, std430) restrict buffer BiasData {
	float data[];
} biases;

layout(set = 0, binding = 2, std430) restrict buffer DeltasData {
	float data[gl_WorkGroupSize.x][gl_WorkGroupSize.y];
} deltas;

layout(set = 0, binding = 3, std430) restrict buffer WeightsData {
	float data[gl_WorkGroupSize.x][gl_WorkGroupSize.y][gl_WorkGroupSize.y];
} weights;

layout(set = 0, binding = 4, std430) restrict buffer StructureData {
	int data[];
} layers;

layout(set = 0, binding = 5, std430) restrict buffer OutputNeurons {
	float data[gl_WorkGroupSize.x][gl_WorkGroupSize.y];
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
	mediump uint max_x = gl_WorkGroupSize.x;
	if( gl_WorkGroupID.y < layers.data[x] && gl_WorkGroupID.x != 0 )
	{
		switch( data.mode )
		{
			case RUN:
				for(mediump uint i = 0; i < layers.data[x - 1]; ++i)
				{
					output_neurons.data[x][y] += output_neurons.data[x - 1][i] * weights.data[x - 1][i][y];
				}
				output_neurons.data[x][y] = f(output_neurons.data[x][y]);
				break;
			case FIND_DELTAS:
				x = max_x - x;
				if( x == max_x - 1)
				{
					deltas.data[x][y] = (output_neurons.data[x][y] - desired_output.data[y]) * fd(output_neurons.data[x][y]);
				}
				else
				{
					for(mediump uint i = 0; i < layers.data[x + 1]; ++i)
					{
						deltas.data[x][y] += output_neurons.data[x + 1][i] * weights.data[x][y][i];
					}
					deltas.data[x][y] *= fd(output_neurons.data[x][y]);
				}
				break;
			case TRAIN:
				x -= 1;
				for(mediump uint i = 0; i < layers.data[x + 1]; ++i)
				{
					weights.data[x][y][i] -= data.learning_rate * output_neurons.data[x][y] * deltas.data[x + 1][i];
				}
				break;
		}
	}
}
