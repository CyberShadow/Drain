import std.random;
import std.stdio;
import std.traits;

private:

struct Variable(T)
{
	T value;
	T gradientAccumulator, gradientTotal;
	// Optimizer parameters here?

	void accumulateGradient(T value, T weight = 1)
	{
		gradientAccumulator += value;
		gradientTotal += weight;
	}
}

struct Dense(T, size_t numInputs, size_t numOutputs)
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

	void forward(ref const T[numInputs] inputs, ref T[numOutputs] outputs)
	{
		foreach (i; 0 .. numOutputs)
			outputs[i] = biases[i].value;
		foreach (i; 0 .. numInputs)
			foreach (o; 0 .. numOutputs)
				outputs[o] += inputs[i] * weights[o][i].value;
	}

	void backward(ref const T[numInputs] inputs, ref const T[numOutputs] gradients, ref Variable!T[numInputs] backGradients)
	{
		foreach (o; 0 .. numOutputs)
		{
			auto gradient = gradients[o];
			biases[o].accumulateGradient(gradient);
			foreach (i; 0 .. numInputs)
				weights[o][i].accumulateGradient(gradient * inputs[i]);
		}

		foreach (o; 0 .. numOutputs)
			foreach (i; 0 .. numInputs)
				backGradients[i].accumulateGradient(gradients[o] * weights[o][i].value);
	}
}

void initialize(Layer)(ref Layer layer)
{
	alias LayerParameter = std.traits.Parameters!(std.traits.Parameters!(Layer.visit)[0])[0];
	alias T = float; // TODO

	void visitor(ref Variable!T p)
	{
		p.value = uniform01!T;
	}
	layer.visit(&visitor);
}

void beginLearn(Layer)(ref Layer layer)
{
	alias T = float; // TODO
	void visitor(ref Variable!T p)
	{
		p.gradientAccumulator = p.gradientTotal = 0;
	}
	layer.visit(&visitor);
}

void endLearn(Layer)(ref Layer layer)
{
	alias T = float; // TODO
	void visitor(ref Variable!T p)
	{
		if (p.gradientAccumulator != 0)
			p.value += p.gradientAccumulator / p.gradientTotal;
	}
	layer.visit(&visitor);
}

void main()
{
	rndGen.seed(1);

	enum numSamples = 16;

	float[2][numSamples] inputs;
	float[1][numSamples] labels;
	foreach (i; 0 .. numSamples)
	{
		inputs[i][0] = uniform01!float();
		inputs[i][1] = uniform01!float();

		labels[i][0] = inputs[i][0] * 2 + inputs[i][1] * 3 + 5;
	}

	struct Model
	{
		Dense!(float, 2, 1) d;
	}
	Model m;
	foreach (ref layer; m.tupleof)
		layer.initialize();

	foreach (i; 0 .. numSamples)
		writef("%1.4f\t", inputs[i][0]);
	writeln;
	foreach (i; 0 .. numSamples)
		writef("%1.4f\t", labels[i][0]);
	writeln;
	writeln("---------------------------------------------------");

	enum numEpochs = 100;
	foreach (epoch; 0 .. numEpochs)
	{
		// auto learningRate = (2.0 / numEpochs) * (numEpochs - epoch) / numEpochs;
		// learningRate *= 100;
		// auto learningRate = 1f;
		auto learningRate = (numEpochs - epoch) / float(numEpochs);

		foreach (ref layer; m.tupleof)
			layer.beginLearn();

		foreach (i; 0 .. numSamples)
		{
			float[1] output, gradient;
			Variable!float[2] inputGradient;
			m.d.forward(inputs[i], output);
			// writef("%1.4f\t", output[0]);
			gradient[] = (labels[i][] - output[]) * learningRate;
			m.d.backward(inputs[i], gradient, inputGradient);
		}
		// writefln("\t%s\t%s", m.d.weights, m.d.biases);

		foreach (ref layer; m.tupleof)
			layer.endLearn();
	}

	writeln(m.d.weights);
	writeln(m.d.biases);
	// d.weights[0][0] = 3;
	// d.biases[0] = 4;

	float[1][numSamples] outputs;
	foreach (i; 0 .. numSamples)
	{
		m.d.forward(inputs[i], outputs[i]);
		writeln(inputs[i], " => ", outputs[i], " / ", labels[i]);
	}

	// d.weights
}
