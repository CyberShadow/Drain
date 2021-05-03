import ae.utils.array;

import std.range;
import std.stdio;
import std.algorithm;
import std.typecons;

import vectorflow;
import vectorflow.math : fabs, round;

void main(string[] args)
{
    writeln("Hello world!");

	Linear[1] lls;

    auto nn = NeuralNet()
        .stack(DenseData(1))
        .stack(lls[0]=Linear(1))
		;
    nn.initialize(0);

	float[] inputs = [1, 2, 3];
	float[] labels = inputs.amap!(n => n * 2 + 1);

	enum scale = 1e3f;
	inputs[] *= scale;
	labels[] *= scale;

	nn.learn(
		inputs.length.iota.map!(i => Tuple!(float[1], "features", float, "label")([inputs[i]], labels[i])),
		"square",
        new ADAM(
            10000, // number of passes
            0.1, // learning rate
            inputs.length, // mini-batch size
		),
		true,
		1,
	);
	foreach (input; inputs)
		writeln(nn.predict([input].staticArray));
}
