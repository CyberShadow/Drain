import std.algorithm.mutation;
import std.algorithm.searching;
import std.math;
import std.random;
import std.range.primitives;
import std.stdio;
import std.traits;

import ae.utils.array;

private:

struct Shape
{
	size_t[] dims;

	@property size_t count()
	{
		size_t result = 1;
		foreach (dim; dims)
			result *= dim;
		return result;
	}

	// Shape cartesianProduct(Shape other) const
	// {
	// 	return Shape(dims ~ other.dims);
	// }
}

// template StaticArray(T, Shape shape)
// {
// 	static if (shape.dims.length == 0)
// 		alias StaticArray = T;
// 	else
// 		alias StaticArray = StaticArray!(T, Shape(shape.dims[1 .. $]))[shape.dims[0]];
// }

struct Index(Shape _shape)
{
	enum shape = _shape;
	size_t[shape.dims.length] indices;
	alias indices this;

	auto opBinary(string op : "~", Shape otherShape)(Index!otherShape otherIndex) @nogc
	{
		enum Shape newShape = Shape(shape.dims ~ otherShape.dims);
		size_t[newShape.dims.length] newIndices;
		newIndices[0 .. indices.length] = indices;
		newIndices[indices.length .. $] = otherIndex.indices;
		return Index!newShape(newIndices);
	}
}

struct ShapeIterator(Shape shape)
{
	Index!shape front;
	bool empty;

	void popFront()
	{
		foreach_reverse (dimIndex; 0 .. shape.dims.length)
		{
			if (++front.indices[dimIndex] == shape.dims[dimIndex])
				front.indices[dimIndex] = 0;
			else
				return;
		}
		empty = true;
	}
}

// -------------

struct Variable(T)
{
	T value;
	T gradientAccumulator = 0, gradientTotal = 0;
	// Optimizer parameters here?

	void accumulateGradient(T value, T weight = 1)
	{
		gradientAccumulator += value;
		gradientTotal += weight;
	}

	@property float gradient() const
	{
		if (gradientAccumulator == 0 || gradientTotal == 0)
			return 0;
		T result = gradientAccumulator / gradientTotal;
		if (result != result)
			result = 0;
		return result;
	}

	// Used to start backpropagation.
	void setGradient(T label, T learningRate)
	{
		gradientAccumulator = (label - value) * learningRate;
		gradientTotal = 1;
	}

	void applyGradient()
	{
		value += gradient();
		gradientAccumulator = gradientTotal = 0;
	}
}
Variable!T variable(T)(T value) { return Variable!T(value); }
T value(T)(Variable!T variable) { return variable.value; }

template isTensor(Tensor, T)
{
	enum isTensor = is(typeof((*cast(Tensor*)null).valueIterator.front) == Variable!T);
}

// Tensor
struct DenseArray(T, Shape _shape)
{
	enum shape = _shape;

	// StaticArray!(Variable!T, shape) array;
	static if (shape.dims.length == 0)
		Variable!T value;
	else
		DenseArray!(T, Shape(shape.dims[1 .. $]))[shape.dims[0]] value;

	enum count = shape.count;

	@property ref Variable!T[count] valueIterator() inout
	{
		return *cast(Variable!T[count]*)&value;
	}

	auto indexIterator() const
	{
		return ShapeIterator!shape();
	}

	ref auto opIndex(Shape indexShape)(Index!indexShape index) inout
	if (shape.dims.startsWith(indexShape.dims))
	{
		static if (indexShape.dims.length == 0)
			return value;
		else
			return value[index.indices[0]][Index!(Shape(indexShape.dims[1 .. $]))(index.indices[1 .. $])];
	}
}
static assert(isTensor!(DenseArray!(float, Shape([1])), float));

struct LinearDense(T, Shape inputShape, Shape outputShape)
{
	DenseArray!(T, Shape(outputShape.dims ~ inputShape.dims)) weights;
	DenseArray!(T, outputShape) biases;

	void visit(void delegate(ref Variable!T p) cb)
	{
		foreach (ref v; weights.valueIterator)
			cb(v);
		foreach (ref v; biases.valueIterator)
			cb(v);
	}

	void forward(InputTensor, OutputTensor)(ref const InputTensor inputs, ref OutputTensor outputs)
	if (isTensor!(InputTensor , T) && InputTensor .shape == inputShape &&
		isTensor!(OutputTensor, T) && OutputTensor.shape == outputShape)
	{
		foreach (i; outputs.indexIterator)
			outputs[i].value = biases[i].value;
		foreach (i; inputs.indexIterator)
			foreach (o; outputs.indexIterator)
				outputs[o].value += inputs[i].value * weights[o~i].value;
	}

	void backward(InputTensor, OutputTensor)(ref InputTensor inputs, ref const OutputTensor outputs)
	if (isTensor!(InputTensor , T) && InputTensor .shape == inputShape &&
		isTensor!(OutputTensor, T) && OutputTensor.shape == outputShape)
	{
		foreach (o; outputs.indexIterator)
		{
			auto gradient = outputs[o].gradient;
			biases[o].accumulateGradient(gradient);
			foreach (i; inputs.indexIterator)
				weights[o~i].accumulateGradient(gradient * inputs[i].value);
		}

		foreach (o; outputs.indexIterator)
			foreach (i; inputs.indexIterator)
				inputs[i].accumulateGradient(outputs[o].gradient * weights[o~i].value);
	}
}

struct Identity(T)
{
	void visit(void delegate(ref Variable!T p) cb)
	{
	}

	void forward(InputTensor, OutputTensor)(ref const InputTensor inputs, ref OutputTensor outputs)
	if (isTensor!(InputTensor, T) && isTensor!(OutputTensor, T) && InputTensor.shape == OutputTensor.shape)
	{
		foreach (i; inputs.indexIterator)
			outputs[i].value = inputs[i].value;
	}

	void backward(InputTensor, OutputTensor)(ref InputTensor inputs, ref const OutputTensor outputs)
	if (isTensor!(InputTensor, T) && isTensor!(OutputTensor, T) && InputTensor.shape == OutputTensor.shape)
	{
		foreach (i; inputs.indexIterator)
			inputs[i].accumulateGradient(outputs[i].gradient);
	}
}

struct ReLU(T)
{
	void visit(void delegate(ref Variable!T p) cb)
	{
	}

	void forward(InputTensor, OutputTensor)(ref const InputTensor inputs, ref OutputTensor outputs)
	if (isTensor!(InputTensor, T) && isTensor!(OutputTensor, T) && InputTensor.shape == OutputTensor.shape)
	{
		foreach (i; inputs.indexIterator)
			outputs[i].value = inputs[i].value < 0 ? 0 : inputs[i].value;
	}

	void backward(InputTensor, OutputTensor)(ref InputTensor inputs, ref const OutputTensor outputs)
	if (isTensor!(InputTensor, T) && isTensor!(OutputTensor, T) && InputTensor.shape == OutputTensor.shape)
	{
		foreach (i; inputs.indexIterator)
			inputs[i].accumulateGradient(inputs[i].value < 0 ? 0 : outputs[i].gradient);
	}
}

// note: this is the tanh-based sigmoid function, not the one used by Keras
struct Sigmoid(T)
{
	void visit(void delegate(ref Variable!T p) cb)
	{
	}

	void forward(InputTensor, OutputTensor)(ref const InputTensor inputs, ref OutputTensor outputs)
	if (isTensor!(InputTensor, T) && isTensor!(OutputTensor, T) && InputTensor.shape == OutputTensor.shape)
	{
		enum z = T(0.5);
		foreach (i; inputs.indexIterator)
			outputs[i].value = tanh(inputs[i].value * z) * z + z;
	}

	void backward(InputTensor, OutputTensor)(ref InputTensor inputs, ref const OutputTensor outputs)
	if (isTensor!(InputTensor, T) && isTensor!(OutputTensor, T) && InputTensor.shape == OutputTensor.shape)
	{
		foreach (i; inputs.indexIterator)
			inputs[i].accumulateGradient(outputs[i].value * (T(1) - outputs[i].value) * outputs[i].gradient);
	}
}

void initialize(Layer)(ref Layer layer)
{
	alias LayerParameter = std.traits.Parameters!(std.traits.Parameters!(Layer.visit)[0])[0];
	alias T = float; // TODO

	void visitor(ref Variable!T p)
	{
		p.value = uniform01!T * 2 - 1;
	}
	layer.visit(&visitor);
}

void applyGradients(Layer)(ref Layer layer)
{
	alias T = float; // TODO
	void visitor(ref Variable!T p)
	{
		p.applyGradient();
	}
	layer.visit(&visitor);
}

void main()
{
	rndGen.seed(0);

	enum numSamples = 4;

	float[2][numSamples] inputs;
	float[1][numSamples] labels;
	foreach (i; 0 .. numSamples)
	{
		inputs[i][0] = uniform01!float();
		inputs[i][1] = uniform01!float();

		// labels[i][0] = inputs[i][0] * 2 + inputs[i][1] * 3 + 5;
		// labels[i][0] = inputs[i][0] < inputs[i][1] ? 0 : 1;
		inputs[i][0] = i % 2;
		inputs[i][1] = i / 2 % 2;
		labels[i][0] = inputs[i][0] != inputs[i][1];
	}

	struct Vars
	{
		DenseArray!(float, Shape([2])) v0;
		DenseArray!(float, Shape([4])) v1;
		DenseArray!(float, Shape([4])) v2;
		DenseArray!(float, Shape([1])) v3;
	}
	struct Model
	{
		LinearDense!(float, Vars.v0.shape, Vars.v1.shape) l1;
		ReLU       !(float                              ) l2;
		LinearDense!(float, Vars.v2.shape, Vars.v3.shape) l3;
	}

	Model m;
	foreach (ref layer; m.tupleof)
		layer.initialize();

	foreach (i; 0 .. inputs[0].length)
	{
		foreach (s; 0 .. numSamples)
			writef("%1.4f\t", inputs[s][i]);
		writeln;
	}
	foreach (s; 0 .. numSamples)
		writef("%1.4f\t", labels[s][0]);
	writeln;
	writeln("---------------------------------------------------");

	enum numEpochs = 1000;
	foreach (epoch; 0 .. numEpochs)
	{
		// auto learningRate = (2.0 / numEpochs) * (numEpochs - epoch) / numEpochs;
		// learningRate *= 100;
		// auto learningRate = 1f;
		auto learningRate = (numEpochs - epoch) / float(numEpochs);

		foreach (s; 0 .. numSamples)
		{
			Vars vars;

			vars.tupleof[0].valueIterator = inputs[s].amap!variable;
			static foreach (i; 0 .. Model.tupleof.length)
				m.tupleof[i].forward(vars.tupleof[i], vars.tupleof[i + 1]);

			// writef("%1.4f\t", hidden[0].value);
			foreach (o; vars.tupleof[$-1].indexIterator)
				vars.tupleof[$-1][o].setGradient(labels[s][o[0]], learningRate);

			static foreach_reverse (i; 0 .. Model.tupleof.length)
				m.tupleof[i].backward(vars.tupleof[i], vars.tupleof[i + 1]);
		}
		// writefln("\t%s\t%s", m.d.weights, m.d.biases);

		foreach (ref layer; m.tupleof)
			layer.applyGradients();
	}

	writeln(m.l1.weights);
	writeln(m.l1.biases);
	// d.weights[0][0] = 3;
	// d.biases[0] = 4;

	foreach (s; 0 .. numSamples)
	{
		Vars vars;
		vars.tupleof[0].valueIterator = inputs[s].amap!variable;
		static foreach (i; 0 .. Model.tupleof.length)
			m.tupleof[i].forward(vars.tupleof[i], vars.tupleof[i + 1]);

		writeln(vars.tupleof[0].valueIterator.amap!value, " => ", vars.tupleof[$-1].valueIterator.amap!value, " / ", labels[s]);
	}

	// d.weights
}
