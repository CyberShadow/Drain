import std.algorithm.mutation;
import std.math;
import std.random;
import std.stdio;
import std.traits;

import ae.utils.array;

private:

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

struct LinearDense(T, size_t numInputs, size_t numOutputs)
{
	Variable!T[numInputs][numOutputs] weights;
	Variable!T[numOutputs] biases;

	void visit(void delegate(ref Variable!T p) cb)
	{
		foreach (ref r; weights)
			foreach (ref v; r)
				cb(v);
		foreach (ref v; biases)
			cb(v);
	}

	void forward(ref const Variable!T[numInputs] inputs, ref Variable!T[numOutputs] outputs)
	{
		foreach (i; 0 .. numOutputs)
			outputs[i].value = biases[i].value;
		foreach (i; 0 .. numInputs)
			foreach (o; 0 .. numOutputs)
				outputs[o].value += inputs[i].value * weights[o][i].value;
	}

	void backward(ref Variable!T[numInputs] inputs, ref const Variable!T[numOutputs] outputs)
	{
		foreach (o; 0 .. numOutputs)
		{
			auto gradient = outputs[o].gradient;
			biases[o].accumulateGradient(gradient);
			foreach (i; 0 .. numInputs)
				weights[o][i].accumulateGradient(gradient * inputs[i].value);
		}

		foreach (o; 0 .. numOutputs)
			foreach (i; 0 .. numInputs)
				inputs[i].accumulateGradient(outputs[o].gradient * weights[o][i].value);
	}
}

struct Identity(T, size_t numInputs)
{
	alias numOutputs = numInputs;

	void visit(void delegate(ref Variable!T p) cb)
	{
	}

	void forward(ref const Variable!T[numInputs] inputs, ref Variable!T[numOutputs] outputs)
	{
		foreach (i; 0 .. numInputs)
			outputs[i].value = inputs[i].value;
	}

	void backward(ref Variable!T[numInputs] inputs, ref const Variable!T[numOutputs] outputs)
	{
		foreach (i; 0 .. numInputs)
			inputs[i].accumulateGradient(outputs[i].gradient);
	}
}

struct ReLU(T, size_t numInputs)
{
	alias numOutputs = numInputs;

	void visit(void delegate(ref Variable!T p) cb)
	{
	}

	void forward(ref const Variable!T[numInputs] inputs, ref Variable!T[numOutputs] outputs)
	{
		foreach (i; 0 .. numInputs)
			outputs[i].value = inputs[i].value < 0 ? 0 : inputs[i].value;
	}

	void backward(ref Variable!T[numInputs] inputs, ref const Variable!T[numOutputs] outputs)
	{
		foreach (i; 0 .. numInputs)
			inputs[i].accumulateGradient(inputs[i].value < 0 ? 0 : outputs[i].gradient);
	}
}

// note: this is the tanh-based sigmoid function, not the one used by Keras
struct Sigmoid(T, size_t numInputs)
{
	alias numOutputs = numInputs;

	void visit(void delegate(ref Variable!T p) cb)
	{
	}

	void forward(ref const Variable!T[numInputs] inputs, ref Variable!T[numOutputs] outputs)
	{
		enum z = T(0.5);
		foreach (i; 0 .. numInputs)
			outputs[i].value = tanh(inputs[i].value * z) * z + z;
	}

	void backward(ref Variable!T[numInputs] inputs, ref const Variable!T[numOutputs] outputs)
	{
		foreach (i; 0 .. numInputs)
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
		Variable!float[2] v0;
		Variable!float[4] v1;
		Variable!float[4] v2;
		Variable!float[1] v3;
	}
	struct Model
	{
		LinearDense!(float, Vars.tupleof[0].length, Vars.tupleof[1].length) l1;
		ReLU       !(float, Vars.tupleof[1].length                        ) l2;
		LinearDense!(float, Vars.tupleof[2].length, Vars.tupleof[3].length) l3;
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

			vars.tupleof[0] = inputs[s].amap!variable;
			static foreach (i; 0 .. Model.tupleof.length)
				m.tupleof[i].forward(vars.tupleof[i], vars.tupleof[i + 1]);

			// writef("%1.4f\t", hidden[0].value);
			foreach (o; 0 .. vars.tupleof[$-1].length)
				vars.tupleof[$-1][o].setGradient(labels[s][o], learningRate);

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
		vars.tupleof[0] = inputs[s].amap!variable;
		static foreach (i; 0 .. Model.tupleof.length)
			m.tupleof[i].forward(vars.tupleof[i], vars.tupleof[i + 1]);

		writeln(vars.tupleof[0].amap!value, " => ", vars.tupleof[$-1].amap!value, " / ", labels[s]);
	}

	// d.weights
}
