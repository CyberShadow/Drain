@nogc:

import std.meta;
import std.range.primitives;

import shapes;

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

		/// Tells children how to distribute the gradient
		/// among its parents.
		&& isBox!(typeof(Tensor.gradientWeights))
		&& Tensor.gradientWeights.shape == Tensor.value.shape
	;
}


/// Type used to represent gradient weights.
alias GradientWeight = uint;

// ----------------------------------------------------------------------------


/// A computation graph, supporting
/// both forward and backpropagation.
struct Graph(Tensors...)
{
	/// All tensors forming this computational graph,
	/// in topological order (inputs first, outputs last).
	/// Each type in `tensors` is unique and statically identifies the tensor.
	Tensors tensors;

	// private enum isInputTensor(Tensor) = is(Tensor == Input!Box, Box);
	// alias InputTensors = Filter!(isInputTensor, Tensors);

	private enum isInputTensor(alias tensor) = __traits(hasMember, typeof(tensor), q{isInput});
	private alias inputTensors = Filter!(isInputTensor, tensors);

	private alias TensorValue(Tensor) = typeof(Tensor.value);

	private alias outputTensors = tensors[$-1 .. $]; // TODO

	/// Calculate output from the given input.
	void forward(staticMap!(TensorValue, typeof(inputTensors)) input)
	{
		static foreach (i; 0 .. inputTensors.length)
			inputTensors[i].value = input[i];

		foreach (i, ref tensor; tensors)
		{
			// TODO multiple parents
			static if (i)
				tensor.forward(tensors[i-1]);
		}
	}

	/// Fit the graph to the given label.
	void backward(staticMap!(TensorValue, typeof(outputTensors)) output)
	{
		// Clear gradients
		foreach_reverse (ti, ref tensor; tensors)
		{
			static if (isTrainable!(typeof(tensor)))
				foreach (ref g; tensor.gradient.valueIterator)
					g = 0;
		}

		static foreach (ti; 0 .. outputTensors.length)
			foreach (i; output[ti].indexIterator)
				outputTensors[ti].gradient[i] = output[ti][i] - outputTensors[ti].value[i];

		foreach_reverse (i, ref tensor; tensors)
		{
			static if (isTrainable!(typeof(tensor)))
			{
				// TODO multiple parents
				static if (i)
					tensor.backward(tensors[i-1]);
				else
					tensor.backward();
			}
		}
	}
}


/// Build a `Graph` starting from the given `outputs`.
/// The computation graph is logically a DAG,
/// with splits (forks) and merges (joins).
/// This is why simple recursive calculation
/// (i.e. where a tensor holds a pointer to
/// a parent or child) is not sufficient.
/// Instead, we order the tensors topologically
/// and invoke them in that order.
auto build(Outputs...)(Outputs outputs)
{
	// TODO actually using topological order
	template ScanTensors(Tensor)
	{
		static assert(isTensor!Tensor);

		static if (Tensor.Parents.length == 0)
			alias ScanTensors = AliasSeq!Tensor;
		else
		static if (Tensor.Parents.length == 1)
			alias ScanTensors = AliasSeq!(ScanTensors!(Tensor.Parents[0]), Tensor);
		else
			static assert(false, "TODO");
	}
	static assert(Outputs.length == 1, "TODO");
	alias Tensors = ScanTensors!(Outputs[0]);
	return Graph!Tensors();
}


// ----------------------------------------------------------------------------


/// Input tensor.
/// Wraps a `Box` into a tensor.
struct Input(Box)
if (isBox!Box)
{
	alias Parents = AliasSeq!(); /// No parents.

	Box value; /// Value is fed in by graph methods.

	/// Never called.
	void forward(ref Parents parents) { assert(false); }

	/// Tells `Graph` to populate `value`.
	enum isInput = true;

	static assert(isTensor!(typeof(this)));
}

auto input(T, Shape shape)()
{
	return Input!(T, shape)();
} /// ditto

auto input(R)(R data)
if (isInputRange!R && isBox!(ElementType!R))
{
	return Input!(ElementType!R)();
} /// ditto


// ----------------------------------------------------------------------------


/// Trainable input tensor.
/// Like `Input`, but accepts and applies gradients.
/// Used for testing backpropagation.
struct TrainableInput(Box)
if (isBox!Box)
{
	alias Parents = AliasSeq!(); /// No parents.

	Box value; /// Value is fed in by graph methods.
	Box gradient; /// Gradient input.
	enum gradientWeights = constant!1.repeat!(Box.shape);

	/// Never called.
	void forward(ref Parents parents) { assert(false); }

	/// Tells `Graph` to populate `value`.
	enum isInput = true;

	void backward(ref Parents parents)
	{
		foreach (i; gradient.indexIterator)
		{
			value[i] += gradient[i];
			gradient[i] = 0;
		}
	}

	static assert(isTensor!(typeof(this)));
	static assert(isTrainable!(typeof(this)));
}

auto trainableInput(T, Shape shape)()
{
	return TrainableInput!(T, shape)();
} /// ditto

auto trainableInput(R)(R data)
if (isInputRange!R && isBox!(ElementType!R))
{
	return TrainableInput!(ElementType!R)();
} /// ditto


// ----------------------------------------------------------------------------


/// Adds values in a box along an axis.
struct Add(Parent, size_t axis)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	alias T = typeof(Parent.value).T;

	DenseBox!(T, Parent.value.shape.dropAxis(axis)) value;
	static if (isTrainable!Parent)
	{
		typeof(value) gradient;
		enum gradientWeights = constant!1.repeat!(typeof(value).shape); // TODO
	}

	void forward(ref Parents parents)
	{
		foreach (ref v; value.valueIterator)
			v = 0;
		foreach (i; parents[0].value.indexIterator)
			value[i.dropAxis!axis] += parents[0].value[i];
	}

	static if (isTrainable!Parent)
	void backward(ref Parents parents)
	{
		DenseBox!(GradientWeight, Parent.value.shape.dropAxis(axis)) weightTotals;
		foreach (i; parents[0].gradientWeights.indexIterator)
			weightTotals[i.dropAxis!axis] += parents[0].gradientWeights[i];
		foreach (i; parents[0].gradient.indexIterator)
			parents[0].gradient[i] += this.gradient[i.dropAxis!axis] * parents[0].gradientWeights[i] / weightTotals[i.dropAxis!axis];
		
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
	auto graph = inputData[].boxes
		.trainableInput
		.add
		.build;

	graph.forward(inputData[0].box);
	assert(graph.tensors[$-1].value.valueIterator.front == 3f);

	float label = 5f;
	graph.backward(label.box);
	assert(graph.tensors[0].value.valueIterator == [2f, 3f]);
}
