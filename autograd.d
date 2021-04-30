@nogc:

import std.math;
import std.meta;
import std.range.primitives;
import std.traits;
debug import std.format;

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

		/// Tells children how to distribute the gradient among its parents.
		/// Should be evaluatable at compile time.
		&& isBox!(typeof(Tensor.gradientWeights))
		&& Tensor.gradientWeights.shape == Tensor.value.shape
	;
}


/// Type used to represent gradient weights.
alias GradientWeight = uint;

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
struct Graph(Outputs...)
{
	alias Tensors = SortTensors!Outputs;

	/// All tensors forming this computational graph,
	/// in topological order (inputs first, outputs last).
	/// Each type in `tensors` is unique and statically identifies the tensor.
	Tensors tensors;

	// private enum isInputTensor(Tensor) = is(Tensor == Input!Box, Box);
	// alias InputTensors = Filter!(isInputTensor, Tensors);

	private enum isInputTensor(alias tensor) = __traits(hasMember, typeof(tensor), q{isInput});
	private alias inputTensors = Filter!(isInputTensor, tensors);

	private alias TensorValue(Tensor) = typeof(Tensor.value);

	enum isTrainable = allSatisfy!(.isTrainable, typeof(outputTensors));

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

	private void initialize()
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
				outputTensors[ti].gradient[i] = output[ti][i] - outputTensors[ti].value[i];

		foreach_reverse (i, ref tensor; tensors)
			static if (.isTrainable!(typeof(tensor)))
				tensor.backward(tensorInstances!(typeof(tensor).Parents));
	}

	/// Backpropagate the given labels, and then do a forward pass.
	/// Assert that the result of the forward pass matches label.
	/// Used to test differentiation.
	static if (this.isTrainable)
	void testGradient(staticMap!(TensorValue, typeof(outputTensors)) output)
	{
		// Clear gradients
		foreach_reverse (ti, ref tensor; tensors)
		{
			static if (.isTrainable!(typeof(tensor)))
				foreach (ref g; tensor.gradient.valueIterator)
					g = 0;
		}

		static foreach (ti; 0 .. outputTensors.length)
			foreach (i; output[ti].indexIterator)
				outputTensors[ti].gradient[i] = output[ti][i] - outputTensors[ti].value[i];

		foreach_reverse (i, ref tensor; tensors)
			static if (.isTrainable!(typeof(tensor)))
				tensor.backward(tensorInstances!(typeof(tensor).Parents));

		foreach (i, ref tensor; tensors)
			tensor.forward(tensorInstances!(typeof(tensor).Parents));

		static foreach (ti; 0 .. outputTensors.length)
			foreach (i; output[ti].indexIterator)
				debug assert(approxEqual(output[ti][i], outputTensors[ti].value[i]),
					format("Wrong output after fitting. Expected: %s, got: %s",
						output[ti][i], outputTensors[ti].value[i],
					),
				);
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
	auto g = Graph!Outputs();
	g.initialize();
	return g;
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

	static if (_isTrainable)
	{
		Box gradient; /// Gradient input.
		enum gradientWeights = constant!1.repeat!(Box.shape);

		void backward(ref Parents parents)
		{
			foreach (i; gradient.indexIterator)
			{
				value[i] += gradient[i];
				gradient[i] = 0;
			}
		}
	}

	static assert(isTensor!(typeof(this)));
	static assert(isTrainable!(typeof(this)) == _isTrainable);
}

alias constant = shapes.constant;

auto constant(R)(R data)
if (isInputRange!R && isBox!(ElementType!R))
{
	return Value!(ElementType!R, false, false)();
} /// ditto

auto variable(R)(R data)
if (isInputRange!R && isBox!(ElementType!R))
{
	return Value!(ElementType!R, false, true)();
} /// ditto

auto input(R)(R data)
if (isInputRange!R && isBox!(ElementType!R))
{
	return Value!(ElementType!R, true, false)();
} /// ditto

auto trainableInput(R)(R data)
if (isInputRange!R && isBox!(ElementType!R))
{
	return Value!(ElementType!R, true, true)();
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
		static immutable gradientWeights = Parent.gradientWeights.fold!axis(sum);
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
		static immutable weightTotals = parents[0].gradientWeights.fold!axis(sum);
		foreach (i; parents[0].gradient.indexIterator)
		{
			auto j = i.dropAxis!axis;
			parents[0].gradient[i] += this.gradient[j] * parents[0].gradientWeights[i] / weightTotals[j];
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
	auto graph = inputData[].boxes
		.trainableInput
		.add
		.build;

	graph.forward(inputData[0].box);
	assert(graph.tensors[$-1].value.valueIterator.front == 3f);

	float label = 5f;
	graph.testGradient(label.box);
	assert(graph.tensors[0].value.valueIterator == [2f, 3f]);
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
	{
		typeof(value) gradient;
		static immutable gradientWeights = Parent.gradientWeights.fold!axis(sum);
	}

	void forward(ref Parents parents)
	{
		foreach (ref v; value.valueIterator)
			v = 1;
		foreach (i; parents[0].value.indexIterator)
			value[i.dropAxis!axis] *= parents[0].value[i];
	}

	static if (isTrainable!Parent)
	void backward(ref Parents parents)
	{
		static immutable weightTotals = Parent.gradientWeights.fold!axis(sum);
		foreach (i; parents[0].gradient.indexIterator)
		{
			auto j = i.dropAxis!axis;
			auto x = parents[0].value[i];
			auto logx = log(abs(x));
			auto y = this.value[j];
			auto logy = log(abs(y));
			auto yg = this.gradient[j];
			auto y2 = y + yg;
			auto logy2 = log(abs(y2));
			auto logyg = logy2 - logy;
			auto logx2 = logx + (logyg * parents[0].gradientWeights[i] / weightTotals[i.dropAxis!axis]);
			auto x2 = exp(logx2) * sgn(x);
			if (sgn(y) != sgn(y2))
				if (i.indices[axis] == 0)
					x2 = -x2;
			auto xg = x2 - x;

			parents[0].gradient[i] += xg;
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
	auto graph = inputData[].boxes
		.trainableInput
		.multiply
		.build;

	graph.forward(inputData[0].box);
	assert(graph.tensors[$-1].value.valueIterator.front == 6f);

	float label = 24f;
	graph.testGradient(label.box);
	assert(graph.tensors[0].value.valueIterator == [4f, 6f]);

	label = -24f;
	graph.testGradient(label.box);
	assert(graph.tensors[0].value.valueIterator == [-4f, 6f]);
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
	{
		typeof(value) gradient;
		static immutable gradientWeights = (){
			DenseBox!(GradientWeight, value.shape) result;
			size_t offset; // along `axis`
			foreach (Parent; Parents)
			{
				static if (isTrainable!Parent)
					foreach (i; Parent.gradientWeights.indexIterator)
					{
						auto j = Index!outputShape(i.indices);
						j[axis] += offset;
						result[j] = Parent.gradientWeights[i];
					}
				offset += Parent.value.shape.dims[axis];
			}
			return result;
		}();
	}

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
					parent.gradient[i] = gradient[j];
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
	auto graph = concatenate!0
		(
			input1[].boxes.input,
			input2[].boxes.input,
		)
		.build;

	graph.forward(input1[0].box, input2[0].box);
	assert(graph.tensors[$-1].value.valueIterator == [1, 2, 3]);
}

unittest
{
	float[2][1] input1 = [[1f, 2f]];
	float[1][1] input2 = [[3f]];
	auto graph = concatenate!0
		(
			input1[].boxes.trainableInput,
			input2[].boxes.trainableInput,
		)
		.add!0
		.build;

	graph.forward(input1[0].box, input2[0].box);
	assert(graph.tensors[$-1].value.valueIterator == [6]);

	float label = 9f;
	graph.testGradient(label.box);
	assert(graph.tensors[0].value.valueIterator == [2f, 3f]);
	assert(graph.tensors[1].value.valueIterator == [4f]);
}
