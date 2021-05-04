@nogc:

import std.array : staticArray;
import std.math;
import std.meta;
import std.random;
import std.range;
import std.range.primitives;
import std.traits;
debug import std.format;

import shapes;
private import shapes : constant, Constant, repeat, Repeat, swapAxes, SwapAxes; // Introduce overloads

// ----------------------------------------------------------------------------

/// Tensor definition.
/// A tensor is like a box, with the following adjustments:
/// - Calculation is explicitly step-wise
///   (to support non-linear computation graphs).
/// - Parents are explicitly declared for the same reason.
/// - In addition to normal forward evaluation,
///   tensors may support backpropagation,
///   in which a gradient is used to apply
///   changes to parameters (i.e. fitting).
enum isTensor(Tensor) = true

	/// This tensor's parents. Cycles are not allowed (nor are possible), but forks/merges are.
	&& __traits(hasMember, Tensor, q{Parents})

	/// Function which populates `value`.
	&& __traits(hasMember, Tensor, q{forward})

	/// Output value (Box).
	&& isBox!(typeof(Tensor.value))
;


/// Whether a tensor supports backpropagation.
template isTrainable(Tensor)
if (isTensor!Tensor)
{
	enum isTrainable = true

		/// Function which applies and further propagates the gradient.
		/// The gradient should be reset (to zeroes) after the call.
		&& __traits(hasMember, Tensor, q{backward})

		/// The counterpart of `value`.
		/// Populated by child tensors' `backward` functions.
		/// Read by this tensor`s `backward` function.
		&& isBox!(typeof(Tensor.gradient))
		&& Tensor.gradient.shape == Tensor.value.shape
	;
}


/// Whether optimizers should adjust this tensor's values.
template isOptimizable(Tensor)
if (isTensor!Tensor)
{
	enum isOptimizable = true
		&& isTrainable!Tensor

		&& __traits(hasMember, Tensor, q{isOptimizable})
		&& Tensor.isOptimizable
	;
}


// ----------------------------------------------------------------------------


struct NullOptimizer(LearningRate = Constant!(float, 0.1f))
{
	LearningRate learningRate;

	struct Instance(Tensors...)
	{
		LearningRate learningRate;

		void run(Tensor)(ref Tensor tensor)
		{
			foreach (i; tensor.value.indexIterator)
				tensor.value[i] -= tensor.gradient[i] * learningRate.value;
		}
	}

	Instance!Tensors initialize(Tensors...)(ref Tensors tensors)
	{
		Instance!Tensors instance;
		instance.learningRate = learningRate;
		return instance;
	}
}


struct AdaGrad(
	LearningRate = Constant!(float, 0.1f),
	Eps = Constant!(float, 1e-8f),
)
{
	LearningRate learningRate;
	Eps eps;

	struct StorageFor(Tensor)
	if (isTensor!Tensor)
	{
		static if (isOptimizable!Tensor)
		{
			typeof(Tensor.value) m; // memory
		}
	}

	struct Instance(Tensors...)
	{
		LearningRate learningRate;
		Eps eps;

		staticMap!(StorageFor, Tensors) storage;
		alias storageFor(Tensor) = storage[staticIndexOf!(StorageFor!Tensor, typeof(storage))];

		void run(Tensor)(ref Tensor tensor)
		{
			foreach (i; tensor.value.indexIterator)
			{
				auto m = storageFor!Tensor.m[i];
				auto g = tensor.gradient[i];
				auto mn = m + g * g;
				auto diff = g / sqrt(mn + eps.value);
				storageFor!Tensor.m[i] = mn;

				// debug { import std.stdio; writefln("AdaGrad: Adjusting value at %s from %s to %s (for %s)",
				// 		i,
				// 		tensor.value[i],
				// 		tensor.value[i] + diff * -learningRate.value,
				// 		Tensor.stringof
				// 	); }
				tensor.value[i] += diff * -learningRate.value;
			}
		}
	}

	Instance!Tensors initialize(Tensors...)(ref Tensors tensors)
	{
		Instance!Tensors instance;
		instance.learningRate = learningRate;
		instance.eps = eps;
		foreach (i, ref s; instance.storage)
			static if (isOptimizable!(Tensors[i]))
				foreach (ref v; s.m.valueIterator)
					v = 0;
		return instance;
	}
}


struct ADAM(
	LearningRate = Constant!(float, 0.1f),
	Beta1 = Constant!(float, 0.9),
	Beta2 = Constant!(float, 0.999),
	Eps   = Constant!(float, 1e-8f),
)
{
	LearningRate learningRate;
	Beta1 beta1;
	Beta2 beta2;
	Eps eps;

	struct StorageFor(Tensor)
	if (isTensor!Tensor)
	{
		static if (isOptimizable!Tensor)
		{
			typeof(Tensor.value) m1, m2; // memory
		}
	}

	struct Instance(Tensors...)
	{
		LearningRate learningRate;
		Beta1 beta1;
		Beta2 beta2;
		Eps eps;

		staticMap!(StorageFor, Tensors) storage;
		alias storageFor(Tensor) = storage[staticIndexOf!(StorageFor!Tensor, typeof(storage))];

		void run(Tensor)(ref Tensor tensor)
		{
			foreach (i; tensor.value.indexIterator)
			{
				auto m1 = storageFor!Tensor.m1[i];
				auto m2 = storageFor!Tensor.m2[i];
				auto g = tensor.gradient[i];
				auto nextM1 = (1.0 - beta1.value) * (g     - m1) + m1;
				auto nextM2 = (1.0 - beta2.value) * (g * g - m2) + m2;
				auto diff = nextM1 / sqrt(nextM2 + eps.value);
				storageFor!Tensor.m1[i] = nextM1;
				storageFor!Tensor.m2[i] = nextM2;

				// debug { import std.stdio; writefln("ADAM: Adjusting value at %s from %s to %s (for %s)",
				// 		i,
				// 		tensor.value[i],
				// 		tensor.value[i] + diff * -learningRate.value,
				// 		Tensor.stringof
				// 	); }
				tensor.value[i] += diff * -learningRate.value;
			}
		}
	}

	Instance!Tensors initialize(Tensors...)(ref Tensors tensors)
	{
		Instance!Tensors instance;
		instance.learningRate = learningRate;
		instance.eps = eps;
		instance.beta1 = beta1;
		instance.beta2 = beta2;
		foreach (i, ref s; instance.storage)
			static if (isOptimizable!(Tensors[i]))
			{
				foreach (ref v; s.m1.valueIterator) v = 0;
				foreach (ref v; s.m2.valueIterator) v = 0;
			}
		return instance;
	}
}


// ----------------------------------------------------------------------------


// Sort `Tensor` and its parents in topological order.
private template SortTensor(Tensor)
{
	static assert(isTensor!Tensor);

	alias SortTensor = AliasSeq!(SortTensors!(Tensor.Parents), Tensor);
}

// Sort `Tensors` and their parents in topological order.
private template SortTensors(Tensors...)
{
	static assert(allSatisfy!(isTensor, Tensors));

	static if (Tensors.length == 0)
		alias SortTensors = AliasSeq!();
	else
		alias SortTensors = NoDuplicates!(SortTensor!(Tensors[0]), SortTensors!(Tensors[1 .. $]));
}


/// A computation graph, supporting
/// both forward and backpropagation.
struct Graph(Optimizer, Outputs...)
{
	alias Tensors = SortTensors!Outputs;

	/// All tensors forming this computational graph,
	/// in topological order (inputs first, outputs last).
	/// Each type in `tensors` is unique and statically identifies the tensor.
	Tensors tensors;

	// private enum isInputTensor(Tensor) = is(Tensor == Input!Box, Box);
	// alias InputTensors = Filter!(isInputTensor, Tensors);

	private enum isInputTensor(alias tensor) = __traits(hasMember, typeof(tensor), q{isInput}) && typeof(tensor).isInput;
	private alias inputTensors = Filter!(isInputTensor, tensors);

	private alias TensorValue(Tensor) = typeof(Tensor.value);

	private enum isTrainable = allSatisfy!(.isTrainable, typeof(outputTensors));

	private template tensorInstances(SoughtTensors...)
	{
		static if (SoughtTensors.length == 0)
			alias tensorInstances = AliasSeq!();
		else
			alias tensorInstances = AliasSeq!(
				tensors[staticIndexOf!(SoughtTensors[0], Tensors)],
				tensorInstances!(SoughtTensors[1 .. $])
			);
	}

	private alias outputTensors = tensorInstances!Outputs;

	private alias OptimizerInstance = typeof(Optimizer.init.initialize(tensors));
	OptimizerInstance optimizer;

	this(ref Optimizer optimizer)
	{
		// Clear the initial gradients (as they are probably NaN by default).
		// After this one-time initialization, they should be cleared
		// by the tensors' individual `backward` methods.
		foreach_reverse (ti, ref tensor; tensors)
		{
			static if (.isTrainable!(typeof(tensor)))
				foreach (ref g; tensor.gradient.valueIterator)
					g = 0;
		}

		this.optimizer = optimizer.initialize(tensors);
	}

	/// Calculate output from the given input.
	void forward(staticMap!(TensorValue, typeof(inputTensors)) input)
	{
		static foreach (i; 0 .. inputTensors.length)
			inputTensors[i].value = input[i];

		foreach (i, ref tensor; tensors)
			tensor.forward(tensorInstances!(typeof(tensor).Parents));
	}

	/// Fit the graph to the given labels.
	static if (this.isTrainable)
	void backward(staticMap!(TensorValue, typeof(outputTensors)) output)
	{
		static foreach (ti; 0 .. outputTensors.length)
			foreach (i; output[ti].indexIterator)
				outputTensors[ti].gradient[i] = outputTensors[ti].value[i] - output[ti][i];

		foreach_reverse (i, ref tensor; tensors)
		{
			static if (.isOptimizable!(typeof(tensor)))
				optimizer.run(tensor);
			static if (.isTrainable!(typeof(tensor)))
				tensor.backward(tensorInstances!(typeof(tensor).Parents));
		}
	}

	// /// Backpropagate the given labels, and then do a forward pass.
	// /// Assert that the result of the forward pass matches label.
	// /// Used to test differentiation.
	// static if (this.isTrainable)
	// void testGradient(staticMap!(TensorValue, typeof(outputTensors)) output)
	// {
	// 	// Clear gradients
	// 	foreach_reverse (ti, ref tensor; tensors)
	// 	{
	// 		static if (.isTrainable!(typeof(tensor)))
	// 			foreach (ref g; tensor.gradient.valueIterator)
	// 				g = 0;
	// 	}

	// 	static foreach (ti; 0 .. outputTensors.length)
	// 		foreach (i; output[ti].indexIterator)
	// 			outputTensors[ti].gradient[i] = output[ti][i] - outputTensors[ti].value[i];

	// 	foreach_reverse (i, ref tensor; tensors)
	// 		static if (.isTrainable!(typeof(tensor)))
	// 			tensor.backward(tensorInstances!(typeof(tensor).Parents));

	// 	foreach (i, ref tensor; tensors)
	// 		tensor.forward(tensorInstances!(typeof(tensor).Parents));

	// 	static foreach (ti; 0 .. outputTensors.length)
	// 		foreach (i; output[ti].indexIterator)
	// 			debug assert(approxEqual(output[ti][i], outputTensors[ti].value[i]),
	// 				format("Wrong output after fitting. Expected: %s, got: %s",
	// 					output[ti][i], outputTensors[ti].value[i],
	// 				),
	// 			);
	// }
}


/// Build a `Graph` starting from the given `outputs`.
/// The computation graph is logically a DAG,
/// with splits (forks) and merges (joins).
/// This is why simple recursive calculation
/// (i.e. where a tensor holds a pointer to
/// a parent or child) is not sufficient.
/// Instead, we order the tensors topologically
/// and invoke them in that order.
auto graph(Optimizer, Outputs...)(Optimizer optimizer, Outputs outputs)
{
	return Graph!(Optimizer, Outputs)(optimizer);
}


// ----------------------------------------------------------------------------


/// Nullary tensor wrapping a `Box`.
/// Holds some values.
/// May or may not be trainable, depending on whether `Box` is writable.
/// Can be used for inputs, weights, biases...
struct Value(Box, bool _isInput, bool _isTrainable)
if (isBox!Box)
{
	alias Parents = AliasSeq!(); /// No parents.

	Box value; /// Value is fed in by graph methods.

	/// No-op.
	void forward(ref Parents parents) {}

	/// Tells `Graph` whether populate `value`.
	enum isInput = _isInput;

	/// Tells optimizers whether to adjust `value`.
	enum isOptimizable = _isTrainable;

	static if (_isTrainable)
	{
		Box gradient; /// Gradient input.

		void backward(ref Parents parents)
		{
			foreach (i; gradient.indexIterator)
				gradient[i] = 0;
		}
	}

	static assert(isTensor!(typeof(this)));
	static assert(isTrainable!(typeof(this)) == _isTrainable);
	static assert(.isOptimizable!(typeof(this)) == _isTrainable);
}

/// A non-trainable non-input value.
alias Constant      (Box) = Value!(Box, false, false);
auto constant      (R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return Constant      !(ElementType!R)(); } /// ditto

/// A trainable non-input value.
alias Variable      (Box) = Value!(Box, false, true );
auto variable      (R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return Variable      !(ElementType!R)(); } /// ditto

/// A non-trainable input value.
alias Input         (Box) = Value!(Box, true , false); 
auto input         (R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return Input         !(ElementType!R)(); } /// ditto

/// A trainable input value.
alias TrainableInput(Box) = Value!(Box, true , true ); 
auto trainableInput(R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return TrainableInput!(ElementType!R)(); } /// ditto


// ----------------------------------------------------------------------------


/// Adds values in a box along an axis.
struct Add(Parent, size_t axis)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	alias T = typeof(Parent.value).T;

	DenseBox!(T, Parent.value.shape.dropAxis(axis)) value;
	static if (isTrainable!Parent)
		typeof(value) gradient;

	void forward(ref Parents parents)
	{
		foreach (ref v; value.valueIterator)
			v = 0;
		foreach (i; parents[0].value.indexIterator)
			value[i.dropAxis!axis] += parents[0].value[i];
		// debug
		// {
		// 	(ref Parents parents){
		// 		import std.stdio, std.algorithm;
		// 		foreach (j; value.indexIterator)
		// 			writefln("%(%s + %) = %s",
		// 				parents[0].value.indexIterator.filter!(i => i.dropAxis!axis == j).map!(i => parents[0].value[i]),
		// 				value[j],
		// 			);
		// 	}(parents);
		// }
	}

	static if (isTrainable!Parent)
	void backward(ref Parents parents)
	{
		foreach (i; parents[0].gradient.indexIterator)
		{
			auto j = i.dropAxis!axis;
			parents[0].gradient[i] += this.gradient[j];
		}
		foreach (ref g; this.gradient.valueIterator)
			g = 0;
	}

	static assert(isTensor!(typeof(this)));
}

Add!(Parent, axis) add(size_t axis = 0, Parent)(Parent parent)
if (isTensor!Parent)
{
	return Add!(Parent, axis)();
} /// ditto

unittest
{
	float[2][1] inputData = [[1f, 2f]];
	auto graph = graph(NullOptimizer!()(),
		inputData[].boxes
		.trainableInput
		.add
	);

	graph.forward(inputData[0].box);
	assert(graph.tensors[$-1].value.valueIterator.front == 3f);

	// float label = 5f;
	// graph.testGradient(label.box);
	// assert(graph.tensors[0].value.valueIterator == [2f, 3f]);
}


// ----------------------------------------------------------------------------


/// Multiplies values in a box along an axis.
/// Note: if the sign is wrong, the backpropagation step fixes this
/// by inverting the sign of the first multiplicand,
/// so that probably should probably be the weight in a NN layer.
struct Multiply(Parent, size_t axis)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	alias T = typeof(Parent.value).T;

	DenseBox!(T, Parent.value.shape.dropAxis(axis)) value;
	static if (isTrainable!Parent)
		typeof(value) gradient;

	void forward(ref Parents parents)
	{
		foreach (ref v; value.valueIterator)
			v = 1;
		foreach (i; parents[0].value.indexIterator)
			value[i.dropAxis!axis] *= parents[0].value[i];
		// debug
		// {
		// 	(ref Parents parents){
		// 		import std.stdio, std.algorithm;
		// 		foreach (j; value.indexIterator)
		// 			writefln("%(%s * %) = %s",
		// 				parents[0].value.indexIterator.filter!(i => i.dropAxis!axis == j).map!(i => parents[0].value[i]),
		// 				value[j],
		// 			);
		// 	}(parents);
		// }
	}

	static if (isTrainable!Parent)
	void backward(ref Parents parents)
	{
		foreach (i; parents[0].gradient.indexIterator)
		{
			auto j = i.dropAxis!axis;

			T otherProduct = 1;
			foreach (row; 0 .. Parent.value.shape.dims[axis])
				if (row != i.indices[axis])
				{
					auto i2 = i;
					i2.indices[axis] = row;
					otherProduct *= parents[0].value[i2];
				}

			parents[0].gradient[i] += this.gradient[j] * otherProduct;
		}

		foreach (ref g; this.gradient.valueIterator)
			g = 0;
	}

	static assert(isTensor!(typeof(this)));
}

Multiply!(Parent, axis) multiply(size_t axis = 0, Parent)(Parent parent)
if (isTensor!Parent)
{
	return Multiply!(Parent, axis)();
} /// ditto

unittest
{
	float[2][1] inputData = [[2f, 3f]];
	auto graph = graph(NullOptimizer!()(),
		inputData[].boxes
		.trainableInput
		.multiply
	);

	graph.forward(inputData[0].box);
	assert(graph.tensors[$-1].value.valueIterator.front == 6f);

	// float label = 24f;
	// graph.testGradient(label.box);
	// assert(graph.tensors[0].value.valueIterator == [4f, 6f]);

	// label = -24f;
	// graph.testGradient(label.box);
	// assert(graph.tensors[0].value.valueIterator == [-4f, 6f]);
}


// ----------------------------------------------------------------------------


struct Concatenate(size_t axis, _Parents...)
if (allSatisfy!(isTensor, _Parents))
{
	alias Parents = _Parents;

	private alias TensorType(Tensor) = typeof(Tensor.value).T;
	alias T = CommonType!(staticMap!(TensorType, Parents));

	private enum tensorShape(Tensor) = Tensor.value.shape;
	private enum outputShape = Shape.concatenate(axis, staticMap!(tensorShape, Parents));

	DenseBox!(T, outputShape) value;
	static if (anySatisfy!(isTrainable, Parents))
		typeof(value) gradient;

	void forward(ref Parents parents)
	{
		size_t offset; // along `axis`
		foreach (ref parent; parents)
		{
			foreach (i; parent.value.indexIterator)
			{
				auto j = Index!outputShape(i.indices);
				j[axis] += offset;
				value[j] = parent.value[i];
			}
			offset += parent.value.shape.dims[axis];
		}
	}

	static if (anySatisfy!(isTrainable, Parents))
	void backward(ref Parents parents)
	{
		size_t offset; // along `axis`
		foreach (ref parent; parents)
		{
			foreach (i; parent.value.indexIterator)
			{
				auto j = Index!outputShape(i.indices);
				j[axis] += offset;
				static if (isTrainable!(typeof(parent)))
					parent.gradient[i] += gradient[j];
				gradient[j] = 0;
			}
			offset += parent.value.shape.dims[axis];
		}
	}

	static assert(isTensor!(typeof(this)));
}

Concatenate!(axis, Parents) concatenate(size_t axis = 0, Parents...)(Parents parents)
if (allSatisfy!(isTensor, Parents))
{
	return Concatenate!(axis, Parents)();
} /// ditto

unittest
{
	float[2][1] input1 = [[1f, 2f]];
	float[1][1] input2 = [[3f]];
	auto graph = graph(NullOptimizer!()(),
		concatenate!0
		(
			input1[].boxes.input,
			input2[].boxes.input,
		)
	);

	graph.forward(input1[0].box, input2[0].box);
	assert(graph.tensors[$-1].value.valueIterator == [1, 2, 3]);
}

unittest
{
	float[2][1] input1 = [[1f, 2f]];
	float[1][1] input2 = [[3f]];
	auto graph = graph(NullOptimizer!()(),
		concatenate!0
		(
			input1[].boxes.trainableInput,
			input2[].boxes.trainableInput,
		)
		.add!0
	);

	graph.forward(input1[0].box, input2[0].box);
	assert(graph.tensors[$-1].value.valueIterator == [6]);

	// float label = 9f;
	// graph.testGradient(label.box);
	// assert(graph.tensors[0].value.valueIterator == [2f, 3f]);
	// assert(graph.tensors[1].value.valueIterator == [4f]);
}


// ----------------------------------------------------------------------------


/// Adds a dimension to the front of `Parent` with length `n`.
struct Repeat(Parent, size_t n)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;

	shapes.Repeat!(typeof(Parent.value), n) value;

	void forward(ref Parents parents)
	{
		value.value = parents[0].value;
	}

	static if (isTrainable!Parent)
	{
		typeof(value) gradient;

		void backward(ref Parents parents)
		{
			foreach (i; parents[0].gradient.indexIterator)
				parents[0].gradient[i] += this.gradient.value[i];
			foreach (ref g; this.gradient.value.valueIterator)
				g = 0;
		}
	}

	static assert(isTensor!(typeof(this)));
	static assert(isTrainable!(typeof(this)) == isTrainable!Parent);
}

Repeat!(Parent, n) repeat(size_t n, Parent)(Parent parent)
if (isTensor!Parent)
{
	return Repeat!(Parent, n)();
} /// ditto


// ----------------------------------------------------------------------------


struct SwapAxes(Parent, size_t axis1, size_t axis2)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;

	shapes.SwapAxes!(typeof(Parent.value), axis1, axis2) value;

	void forward(ref Parents parents)
	{
		value.value = parents[0].value;
	}

	static if (isTrainable!Parent)
	{
		typeof(value) gradient;

		void backward(ref Parents parents)
		{
			foreach (i; parents[0].gradient.indexIterator)
				parents[0].gradient[i] += this.gradient[i];
			foreach (ref g; this.gradient.value.valueIterator)
				g = 0;
		}
	}

	static assert(isTensor!(typeof(this)));
	static assert(isTrainable!(typeof(this)) == isTrainable!Parent);
}

SwapAxes!(Parent, axis1, axis2) swapAxes(size_t axis1, size_t axis2, Parent)(Parent parent)
if (isTensor!Parent)
{
	return SwapAxes!(Parent, axis1, axis2)();
} /// ditto


// ----------------------------------------------------------------------------


auto linearDense(size_t numOutputs, Parent)(Parent parent)
{
	// Assume first axis is the batch size
	enum batchSize = Parent.value.shape.dims[0];
	enum inputShape = Parent.value.shape.dropAxis(0);

	auto weights = Variable!(DenseBox!(Parent.value.T, Shape(numOutputs ~ inputShape.dims)))();
	auto biases  = Variable!(DenseBox!(Parent.value.T, inputShape))();
	return
		concatenate(
			concatenate(
				weights
				.repeat!batchSize
				.swapAxes!(0, 1)
				.repeat!1
				,
				parent
				.repeat!numOutputs
				.repeat!1
				,
			)
			.multiply,
			biases
			.repeat!batchSize
			.repeat!1,
		)
		.add;
}

unittest
{
	import std.random;
	rndGen.seed(0);

	auto inputData = [[1f].staticArray, [2f].staticArray, [3f].staticArray].staticArray;
	auto labelData = [[3f].staticArray, [5f].staticArray, [7f].staticArray].staticArray;
	auto graph = graph(ADAM!()(),
		inputData[].boxes
		.input
		.linearDense!1
	);

	foreach (ref tensor; graph.tensors)
		foreach (ref v; tensor.value.valueIterator)
			v = 0.5;

	foreach (epoch; 0 .. 1000)
	{
		// debug { import std.stdio; writefln("\n=== Epoch %d ===", epoch); }
		foreach (i; inputData.length.iota/*.randomCover*/)
		{
			// debug { import std.stdio; writefln("--- %s -> %s :", inputData[i], labelData[i]); }
			graph.forward (inputData[i].box);
			graph.backward(labelData[i].box);
		}
	}

	foreach (i; 0 .. inputData.length)
	{
		graph.forward(inputData[i].box);
		assert(isClose(graph.tensors[$-1].value.valueIterator.front, labelData[i][0]));
	}

	// debug
	// {
	// 	import std.stdio;
	// 	foreach (ref tensor; graph.tensors)
	// 		writeln(tensor.value.valueIterator, " ", typeof(tensor).stringof);

	// 	foreach (i; 0 .. inputData.length)
	// 	{
	// 		graph.forward (inputData[i].box);
	// 		writeln(graph.tensors[$-1].value.valueIterator);
	// 	}
	// }

	// graph.forward(inputData[0].box);
	// assert(graph.tensors[$-1].value.valueIterator.front == 6f);

	// // float label = 24f;
	// // graph.testGradient(label.box);
	// assert(graph.tensors[0].value.valueIterator == [4f, 6f]);

	// // label = -24f;
	// // graph.testGradient(label.box);
	// // assert(graph.tensors[0].value.valueIterator == [-4f, 6f]);
}
