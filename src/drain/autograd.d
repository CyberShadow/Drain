module drain.autograd;

@nogc:

import std.algorithm.mutation : swap;
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

// debug = verbose;
debug (verbose) import std.stdio;

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

	/// Tensors should have a unique name.
	/// Because tensors are identified by their type,
	/// the name is an additional disambiguation
	/// mechanism when the type otherwise coincides.
	&& __traits(hasMember, Tensor, q{name})

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

				debug (verbose) writefln("AdaGrad: Adjusting value at %s from %s to %s (for %s)",
					i,
					tensor.value[i],
					tensor.value[i] + diff * -learningRate.value,
					Tensor.stringof
				);
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

				debug (verbose) writefln("ADAM: Adjusting value at %s from %s to %s (for %s)",
					i,
					tensor.value[i],
						tensor.value[i] + diff * -learningRate.value,
					Tensor.stringof
				);
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

	debug (verbose)
	{
		pragma(msg, "Graph tensors:");
		static foreach (Tensor; Tensors)
			pragma(msg, Tensor.name);
	}

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
struct Value(Box, bool _isInput, bool _isTrainable, string _name)
if (isBox!Box)
{
	alias Parents = AliasSeq!(); /// No parents.
	enum name = _name;

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
alias Constant      (Box, string name = "constant") = Value!(Box, false, false, name);
auto constant      (R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return Constant      !(ElementType!R)(); } /// ditto

/// A trainable non-input value.
alias Variable      (Box, string name = "variable") = Value!(Box, false, true , name);
auto variable      (R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return Variable      !(ElementType!R)(); } /// ditto

/// A non-trainable input value.
alias Input         (Box, string name = "input") = Value!(Box, true , false, name);
auto input         (R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return Input         !(ElementType!R)(); } /// ditto

/// A trainable input value.
alias TrainableInput(Box, string name = "trainableInput") = Value!(Box, true , true , name);
auto trainableInput(R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return TrainableInput!(ElementType!R)(); } /// ditto


// ----------------------------------------------------------------------------


/// Adds values in a box along an axis.
struct Add(Parent, size_t[] axes)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	alias T = typeof(Parent.value).T;
	enum name = Parent.name ~ ".add";

	DenseBox!(T, Parent.value.shape.dropAxes(axes)) value;
	static if (isTrainable!Parent)
		typeof(value) gradient;

	void forward(ref Parents parents)
	{
		foreach (ref v; value.valueIterator)
			v = 0;
		foreach (i; parents[0].value.indexIterator)
			value[i.dropAxes!axes] += parents[0].value[i];
		debug (verbose)
		{
			(ref Parents parents){
				import std.algorithm;
				foreach (j; value.indexIterator)
					writefln("%s: %(%s + %) = %s",
						j.indices, parents[0].value.indexIterator.filter!(i => i.dropAxes!axes == j).map!(i => parents[0].value[i]),
						value[j],
					);
			}(parents);
		}
	}

	static if (isTrainable!Parent)
	void backward(ref Parents parents)
	{
		foreach (i; parents[0].gradient.indexIterator)
		{
			auto j = i.dropAxes!axes;
			parents[0].gradient[i] += this.gradient[j];
		}
		foreach (ref g; this.gradient.valueIterator)
			g = 0;
	}

	static assert(isTensor!(typeof(this)));
}

Add!(Parent, axes) add(size_t[] axes = [0], Parent)(Parent parent)
if (isTensor!Parent)
{
	return Add!(Parent, axes)();
} /// ditto

Add!(Parent, [axis]) add(size_t axis, Parent)(Parent parent)
if (isTensor!Parent)
{
	return Add!(Parent, [axis])();
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
	enum name = Parent.name ~ ".multiply";

	DenseBox!(T, Parent.value.shape.dropAxis(axis)) value;
	static if (isTrainable!Parent)
		typeof(value) gradient;

	void forward(ref Parents parents)
	{
		foreach (ref v; value.valueIterator)
			v = 1;
		foreach (i; parents[0].value.indexIterator)
			value[i.dropAxis!axis] *= parents[0].value[i];
		debug (verbose)
		{
			(ref Parents parents){
				import std.stdio, std.algorithm;
				foreach (j; value.indexIterator)
					writefln("%s: %(%s * %) = %s",
						j.indices, parents[0].value.indexIterator.filter!(i => i.dropAxis!axis == j).map!(i => parents[0].value[i]),
						value[j],
					);
			}(parents);
		}
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

	private alias tensorName(Tensor) = Tensor.name;
	enum name = "[" ~ [staticMap!(tensorName, Parents)].join(", ") ~ "].concatenate";

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


/// Adds dimensions to the front of `Parent` with the given shape.
struct Repeat(Parent, Shape shape, size_t where)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	enum name = Parent.name ~ ".repeat";

	typeof(shapes.repeat!(shape, where)(Parent.value)) value;

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

Repeat!(Parent, shape, where) repeat(Shape shape, size_t where = 0, Parent)(Parent parent)
if (isTensor!Parent)
{
	return Repeat!(Parent, shape, where)();
} /// ditto

Repeat!(Parent, Shape([n]), where) repeat(size_t n, size_t where = 0, Parent)(Parent parent)
if (isTensor!Parent)
{
	return Repeat!(Parent, Shape([n]), where)();
} /// ditto


// ----------------------------------------------------------------------------


struct SwapAxes(Parent, size_t axis1, size_t axis2)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	enum name = Parent.name ~ ".swapAxes";

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
				parents[0].gradient[i] += this.gradient[i.swapAxes!(axis1, axis2)];
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


auto linearDense(Shape outputShape, Parent)(Parent parent)
{
	// Assume first axis is the batch size
	enum batchSizeAxis = 0;
	enum batchSize = Parent.value.shape.dims[batchSizeAxis];
	enum inputShape = Parent.value.shape.dropAxis(batchSizeAxis);

	auto weights = Variable!(DenseBox!(Parent.value.T, Shape(outputShape.dims ~ inputShape.dims)), Parent.name ~ ".dense-weights")();
	auto biases  = Variable!(DenseBox!(Parent.value.T, outputShape                              ), Parent.name ~ ".dense-biases" )();
	return
		concatenate(
			concatenate(
				weights
				// outputShape x inputShape
				.repeat!batchSize // insert batch size axis
				// batchSize x outputShape x inputShape
				.repeat!1 // first multiplicand
				// 1 x batchSize x outputShape x inputShape
				,

				parent
				// batchSize x inputShape
				.repeat!(outputShape, batchSizeAxis + 1)
				// batchSize x outputShape x inputShape
				.repeat!1 // second multiplicand
				// 1 x batchSize x outputShape x inputShape
				,
			)
			// 2 x batchSize x outputShape x inputShape
			.multiply
			// batchSize x outputShape x inputShape
			.add!(iota(1 + outputShape.dims.length, 1 + outputShape.dims.length + inputShape.dims.length).array)
			// batchSize x outputShape
			.repeat!1 // first addend
			// 1 x batchSize x outputShape
			,

			biases
			// outputShape
			.repeat!batchSize // insert batch size axis
			// batchSize x outputShape
			.repeat!1 // last addend
			// 1 x batchSize x outputShape
			,
		)
		// 2 x batchSize x outputShape
		.add
		// batchSize x outputShape
	;
}

auto linearDense(size_t numOutputs, Parent)(Parent parent) { return linearDense!(Shape([numOutputs]))(parent); }


// ----------------------------------------------------------------------------


/// Layer template for a unary function.
template Unary(alias forwardFunc, alias backwardFunc, string _name)
{
	struct Unary(Parent)
	if (isTensor!Parent)
	{
		alias Parents = AliasSeq!Parent;
		alias T = typeof(Parent.value).T;
		enum name = Parent.name ~ "." ~ _name;

		typeof(Parent.value) value;
		static if (isTrainable!Parent)
			typeof(value) gradient;

		void forward(ref Parents parents)
		{
			foreach (i; parents[0].value.indexIterator)
			{
				value[i] = forwardFunc(parents[0].value[i]);
				debug (verbose)
					writefln("%s: %s(%s) = %s", i, _name, parents[0].value[i], value[i]);
			}
		}

		static if (isTrainable!Parent)
		void backward(ref Parents parents)
		{
			foreach (i; gradient.indexIterator)
				parents[0].gradient[i] += backwardFunc(parents[0].value[i], this.value[i], this.gradient[i]);
			foreach (ref g; this.gradient.valueIterator)
				g = 0;
		}

		static assert(isTensor!(typeof(this)));
	}
}

/// ditto
template unary(alias forwardFunc, alias backwardFunc, string name)
{
	alias Inst = Unary!(forwardFunc, backwardFunc, name);

	Inst!(Parent) unary(Parent)(Parent parent)
	if (isTensor!Parent)
	{
		return Inst!(Parent)();
	} /// ditto
}


/// Rectified Linear Unit activation.
T reluForward(T)(T input) { return input > 0 ? input : 0; }
T reluBackward(T)(T input, T output, T gradient) { return input > 0 ? gradient : 0; } /// ditto

alias ReLU = Unary!(reluForward, reluBackward, "relu"); /// ditto
alias relu = unary!(reluForward, reluBackward, "relu"); /// ditto


// note: this is the tanh-based sigmoid function, not the one used by Keras
/// Sigmoid activation.
T sigmoidForward(T)(T input)
{
	enum z = T(0.5);
	return tanh(input * z) * z + z;
}

T sigmoidBackward(T)(T input, T output, T gradient)
{
	return output * (T(1) - output) * gradient;
} /// ditto

alias Sigmoid = Unary!(sigmoidForward, sigmoidBackward, "sigmoid"); /// ditto
alias sigmoid = unary!(sigmoidForward, sigmoidBackward, "sigmoid"); /// ditto


// ----------------------------------------------------------------------------


version (unittest)
private void testProblem(Graph, Input, Output, size_t numObservations)(
	Graph graph,
	ref Input[numObservations] inputs,
	ref Output[numObservations] labels,
)
{
	assert(inputs.length == labels.length);

	foreach (ref tensor; graph.tensors)
		foreach (ref v; tensor.value.valueIterator)
			v = uniform01!float;

	foreach (epoch; 0 .. 1000)
	{
		debug (verbose)
		{
			writefln("\n=== Epoch %d ===", epoch);
			foreach (ref tensor; graph.tensors)
				static if (isOptimizable!(typeof(tensor)))
					writeln(tensor.name, ": ", tensor.value.valueIterator);
		}
		foreach (i; numObservations.iota/*.randomCover*/)
		{
			debug (verbose) { import std.stdio; writefln("--- %s -> %s :", inputs[i], labels[i]); }
			graph.forward (inputs[i].box);
			graph.backward(labels[i].box);
		}
	}

	foreach (i; 0 .. numObservations)
	{
		graph.forward(inputs[i].box);
		auto output = graph.tensors[$-1].value.valueIterator.front;
		auto label = labels[i].box.valueIterator.front;
		debug (verbose) { import std.stdio; writefln("%s -> %s / %s", inputs[i], output, label); }
		assert(round(output) == round(label));
	}
}

/// Simple add+multiply (one layer, one input, one output, one bias, one weight)
unittest
{
	rndGen.seed(0);

	enum numSamples = 16;

	float[1][numSamples] inputs;
	float[1][numSamples] labels;
	foreach (i; 0 .. numSamples)
	{
		enum scale = 1f;
		inputs[i] = [uniform01!float * scale];
		labels[i] = [inputs[i][0] * 3f * scale + 4f * scale];
	}

	auto graph = graph(ADAM!()(),
		inputs[].boxes
		.input
		.linearDense!(Shape())
	);
	testProblem(graph, inputs, labels);
}

/// Ditto (old test)
unittest
{
	rndGen.seed(0);

	float[1][3] inputs = [[1], [2], [3]];
	float[1][3] labels = [[3], [5], [7]];
	auto graph = graph(ADAM!()(),
		inputs[].boxes
		.input
		.linearDense!(Shape())
	);
	testProblem(graph, inputs, labels);
}

/// Simple add+multiply of two inputs (one layer, two inputs, one output, one bias, two weights)
unittest
{
	rndGen.seed(0);

	enum numSamples = 16;

	float[2][numSamples][1] inputs;
	float[1][numSamples][1] labels;
	foreach (i; 0 .. numSamples)
	{
		inputs[0][i] = [uniform01!float, uniform01!float];
		labels[0][i] = [inputs[0][i][0] * 3f + inputs[0][i][1] * 4f + 5f];
	}

	auto graph = graph(ADAM!()(),
		inputs[].boxes
		.input
		.linearDense!1
	);
	testProblem(graph, inputs, labels);
}

/// Sigmoid test (comparison of two numbers)
unittest
{
	rndGen.seed(1);

	enum numSamples = 16;

	float[2][numSamples][1] inputs;
	float[1][numSamples][1] labels;
	foreach (i; 0 .. numSamples)
	{
		inputs[0][i] = [uniform01!float, uniform01!float];
		if ((inputs[0][i][0] < inputs[0][i][1]) != (i % 2))
			swap(inputs[0][i][0], inputs[0][i][1]);
		labels[0][i][0] = i % 2;
	}

	auto graph = graph(ADAM!()(),
		inputs[].boxes
		.input
		.linearDense!1
		.sigmoid
	);
	testProblem(graph, inputs, labels);
}

/// XOR problem
unittest
{
	// We're using few units and non-leaky
	// ReLUs, so the seed is important
	rndGen.seed(0);

	enum numSamples = 4;

	float[2][1][numSamples] inputs;
	float[1][1][numSamples] labels;
	foreach (i; 0 .. numSamples)
	{
		inputs[i][0] = [i % 2, i / 2 % 2];
		labels[i][0] = [inputs[i][0][0] != inputs[i][0][1]];
	}

	auto graph = graph(ADAM!(Constant!(float, 0.01f))(),
		inputs[].boxes
		.input
		.linearDense!4
		.relu
		.linearDense!1
	);
	testProblem(graph, inputs, labels);
}
