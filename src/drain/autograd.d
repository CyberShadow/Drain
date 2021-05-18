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

import drain.box.shapes;
import shapes = drain.box.shapes;

import drain.util;

// Introduce overloads
private import drain.box.shapes : constant, Constant, repeat, Repeat, swapAxes, SwapAxes, sliceOne;
private import std.math : exp;

// debug = drain_verbose;
debug (drain_verbose) import std.stdio;

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

	/// The type of the tensor's value
	/// (and also the return type of `getValue`).
	/// Necessary because the `getValue` implementation
	/// is templated on the full graph.
	&& __traits(hasMember, Tensor, q{Value})

	/// Calculates this tensor's value, if necessary,
	/// making `getValue` calls available.
	/// May be empty (if this tensor's value can be
	/// calculated in O(1) per index).
	&& __traits(hasMember, Tensor, q{forward})

	/// Returns a box which holds this tensor's value.
	/// Should be O(1), and reading from the returned value should be O(1).
	&& __traits(hasMember, Tensor, q{getValue})
;


/// Whether a tensor supports backpropagation.
template isTrainable(Tensor)
if (isTensor!Tensor)
{
	enum isTrainable = true

		/// Returns a writable box which holds this tensor's gradient.
		/// Should be O(1), and reading/writing to the returned value should be O(1).
		&& __traits(hasMember, Tensor, q{getGradient})

		/// Function which applies and further propagates the gradient.
		/// The gradient should be reset (to zeroes) after the call, if applicable.
		/// May be empty (if this tensor's gradient can be
		/// propagated in O(1) per index).
		&& __traits(hasMember, Tensor, q{backward})
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

		// Optimizable tensors should be stateful.
		&& isBox!(typeof(Tensor.value))
		&& isBox!(typeof(Tensor.gradient))
	;
}


// ----------------------------------------------------------------------------


struct NullOptimizer(LearningRate)
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

NullOptimizer!(LearningRate) nullOptimizer(LearningRate = typeof(constant!0.1f()))(LearningRate learningRate = LearningRate())
{
	return NullOptimizer!(LearningRate)(learningRate);
}


struct AdaGrad(LearningRate, Eps)
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

				debug (drain_verbose) writefln("AdaGrad: Adjusting value at %s from %s to %s (for %s)",
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
AdaGrad!(LearningRate, Eps) adaGrad(
	// https://issues.dlang.org/show_bug.cgi?id=21917
	LearningRate = typeof(constant!0.1f()),
	Eps          = typeof(constant!1e-8f()),
)(
	LearningRate learningRate = LearningRate(),
	Eps          eps          = Eps(),
)
{
	return AdaGrad!(LearningRate)(learningRate);
}


struct Adam(LearningRate, Beta1, Beta2, Eps)
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

				debug (drain_verbose) writefln("ADAM: Adjusting value at %s from %s to %s (for %s)",
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
Adam!(LearningRate, Beta1, Beta2, Eps) adam(
	// https://issues.dlang.org/show_bug.cgi?id=21917
	LearningRate = typeof(constant!0.1f()),
	Beta1        = typeof(constant!0.9f()),
	Beta2        = typeof(constant!0.999f()),
	Eps          = typeof(constant!1e-8f()),
)(
	LearningRate learningRate = LearningRate(),
	Beta1        beta1        = Beta1       (),
	Beta2        beta2        = Beta2       (),
	Eps          eps          = Eps         (),
)
{
	return Adam!(LearningRate, Beta1, Beta2, Eps)(learningRate, beta1, beta2, eps);
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

	debug (drain_verbose)
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

	private alias TensorValue(Tensor) = Tensor.Value;

	private enum isTrainable = allSatisfy!(.isTrainable, typeof(outputTensors));

	/// Find a tensor by type.
	alias tensorInstance(SoughtTensor) = tensors[staticIndexOf!(SoughtTensor, Tensors)];
	template tensorInstances(SoughtTensors...)
	{
		static if (SoughtTensors.length == 0)
			alias tensorInstances = AliasSeq!();
		else
			alias tensorInstances = AliasSeq!(
				tensorInstance!(SoughtTensors[0]),
				tensorInstances!(SoughtTensors[1 .. $])
			);
	} /// ditto

	/// Reference to output tensors.
	alias outputTensors = tensorInstances!Outputs;

	private template hasName(string name) { enum hasName(alias tensor) = tensor.name == name; }
	alias tensorsByName(string name) = Filter!(hasName!name, tensors); /// Find a tensor by name.
	template tensorByName(string name)
	{
		alias result = tensorsByName!name;
		static assert(result.length > 0, "No tensor with this name");
		static assert(result.length < 2, "Multiple tensors with this name");
		alias tensorByName = result[0];
	} /// ditto

	private alias OptimizerInstance = typeof(Optimizer.init.initialize(tensors));
	/// The optimizer instantiated over this graph.
	OptimizerInstance optimizer;

	this(ref Optimizer optimizer)
	{
		// Clear the initial gradients (as they are probably NaN by default).
		// After this one-time initialization, they should be cleared
		// by the tensors' individual `backward` methods.
		foreach_reverse (ti, ref tensor; tensors)
		{
			static if (__traits(hasMember, typeof(tensor), q{gradient}))
				foreach (ref g; tensor.gradient.valueIterator)
					g = 0;
			static if (.isTrainable!(typeof(tensor)))
				tensor.backward(this); // Flush downwards
		}

		this.optimizer = optimizer.initialize(tensors);
	}

	/// Calculate output from the given input.
	void forward(
		staticMap!(TensorValue, typeof(inputTensors)) inputs,
		ref staticMap!(TensorValue, typeof(outputTensors)) outputs,
	)
	{
		static foreach (i; 0 .. inputTensors.length)
			inputTensors[i].getValue(this) = inputs[i];

		foreach (i, ref tensor; tensors)
			tensor.forward(this);

		static foreach (i; 0 .. outputTensors.length)
			outputs[i] = outputTensors[i].getValue(this);
	}

	/// ditto
	static if (outputTensors.length == 1)
	Outputs[0].Value forward(staticMap!(TensorValue, typeof(inputTensors)) inputs)
	{
		typeof(return) output;
		forward(inputs, output);
		return output;
	}

	/// Fit the graph to the given labels.
	static if (this.isTrainable)
	void backward(staticMap!(TensorValue, typeof(outputTensors)) output)
	{
		static foreach (ti; 0 .. outputTensors.length)
			foreach (i; output[ti].indexIterator)
				outputTensors[ti].getGradient(this)[i] = outputTensors[ti].getValue(this)[i] - output[ti][i];

		foreach_reverse (i, ref tensor; tensors)
		{
			static if (.isOptimizable!(typeof(tensor)))
				optimizer.run(tensor);
			static if (.isTrainable!(typeof(tensor)))
				tensor.backward(this);
		}
	}

	/// Run backpropagation only (no optimization) for the given tensors only.
	static if (this.isTrainable)
	void backwardTensor(Tensors...)()
	{
		foreach_reverse (ref tensor; tensorInstances!Tensors)
		{
			static assert (.isTrainable!(typeof(tensor)));
			tensor.backward(this);
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

	alias Value = Box;
	Box value; /// Value is fed in by graph methods.
	ref Box getValue(Graph)(ref Graph graph) { return value; } /// ditto

	/// No-op.
	void forward(Graph)(ref Graph graph) {}

	/// Tells `Graph` whether populate `value`.
	enum isInput = _isInput;

	/// Tells optimizers whether to adjust `value`.
	enum isOptimizable = _isTrainable;

	static if (_isTrainable)
	{
		Box gradient; /// Gradient input.
		ref Box getGradient(Graph)(ref Graph graph) { return gradient; } /// ditto

		void backward(Graph)(ref Graph graph)
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
auto constant      (string name = "constant"      , R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return Constant      !(ElementType!R, name)(); } /// ditto

/// A trainable non-input value.
alias Variable      (Box, string name = "variable") = Value!(Box, false, true , name);
auto variable      (string name = "variable"      , R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return Variable      !(ElementType!R, name)(); } /// ditto

/// A non-trainable input value.
alias Input         (Box, string name = "input") = Value!(Box, true , false, name);
auto input         (string name = "input"         , R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return Input         !(ElementType!R, name)(); } /// ditto

/// A trainable input value.
alias TrainableInput(Box, string name = "trainableInput") = Value!(Box, true , true , name);
auto trainableInput(string name = "trainableInput", R)(R data) if (isInputRange!R && isBox!(ElementType!R)) { return TrainableInput!(ElementType!R, name)(); } /// ditto


// ----------------------------------------------------------------------------


/// Adds values in a box along an axis.
struct Add(Parent, AxisIndex[] axes)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	alias T = Parent.Value.T;
	enum name = Parent.name ~ ".add";

	alias Value = DenseBox!(T, Parent.Value.shape.dropAxes(axes));
	private Value value;
	ref Value getValue(Graph)(ref Graph graph) { return value; } /// ditto

	static if (isTrainable!Parent)
	{
		private Value gradient;
		ref typeof(gradient) getGradient(Graph)(ref Graph graph) { return gradient; } /// ditto
	}

	void forward(Graph)(ref Graph graph)
	{
		auto parent = &graph.tensorInstance!Parent;
		foreach (ref v; value.valueIterator)
			v = 0;
		foreach (i; parent.getValue(graph).indexIterator)
			value[i.dropAxes!axes] += parent.getValue(graph)[i];
		debug (drain_verbose)
		{
			(ref Parents parents){
				import std.algorithm;
				foreach (j; value.indexIterator)
					writefln("%s: %(%s + %) = %s",
						j, parent.getValue(graph).indexIterator.filter!(i => i.dropAxes!axes == j).map!(i => parent.getValue(graph)[i]),
						value[j],
					);
			}(parents);
		}
	}

	static if (isTrainable!Parent)
	void backward(Graph)(ref Graph graph)
	{
		auto parent = &graph.tensorInstance!Parent;
		foreach (i; parent.getGradient(graph).indexIterator)
		{
			auto j = i.dropAxes!axes;
			parent.getGradient(graph)[i] += this.gradient[j];
		}
		foreach (ref g; this.gradient.valueIterator)
			g = 0;
	}

	static assert(isTensor!(typeof(this)));
}

Add!(Parent, axes) add(AxisIndex[] axes = [0], Parent)(Parent parent)
if (isTensor!Parent)
{
	return Add!(Parent, axes)();
} /// ditto

Add!(Parent, [axis]) add(AxisIndex axis, Parent)(Parent parent)
if (isTensor!Parent)
{
	return Add!(Parent, [axis])();
} /// ditto

unittest
{
	float[2][1] inputData = [[1f, 2f]];
	auto graph = graph(nullOptimizer(),
		inputData[].boxes
		.trainableInput
		.add
	);

	assert(graph.forward(inputData[0].box).valueIterator.front == 3f);

	// float label = 5f;
	// graph.testGradient(label.box);
	// assert(graph.tensors[0].value.valueIterator == [2f, 3f]);
}


// ----------------------------------------------------------------------------


/// Multiplies values in a box along an axis.
/// Note: if the sign is wrong, the backpropagation step fixes this
/// by inverting the sign of the first multiplicand,
/// so that probably should probably be the weight in a NN layer.
struct Multiply(Parent, AxisIndex axis)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	alias T = Parent.Value.T;
	enum name = Parent.name ~ ".multiply";

	alias Value = DenseBox!(T, Parent.Value.shape.dropAxis(axis));
	private Value value;
	ref Value getValue(Graph)(ref Graph graph) { return value; } /// ditto

	static if (isTrainable!Parent)
	{
		private Value gradient;
		ref typeof(gradient) getGradient(Graph)(ref Graph graph) { return gradient; } /// ditto
	}

	void forward(Graph)(ref Graph graph)
	{
		auto parent = &graph.tensorInstance!Parent;
		foreach (ref v; value.valueIterator)
			v = 1;
		foreach (i; parent.getValue(graph).indexIterator)
			value[i.dropAxis!axis] *= parent.getValue(graph)[i];
		debug (drain_verbose)
		{
			(ref Parents parents){
				import std.stdio, std.algorithm;
				foreach (j; value.indexIterator)
					writefln("%s: %(%s * %) = %s",
						j, parent.getValue(graph).indexIterator.filter!(i => i.dropAxis!axis == j).map!(i => parent.getValue(graph)[i]),
						value[j],
					);
			}(parents);
		}
	}

	static if (isTrainable!Parent)
	void backward(Graph)(ref Graph graph)
	{
		auto parent = &graph.tensorInstance!Parent;
		foreach (i; parent.getGradient(graph).indexIterator)
		{
			auto j = i.dropAxis!axis;

			T otherProduct = 1;
			foreach (row; 0 .. Parent.Value.shape.dims[axis])
				if (row != i[axis])
				{
					auto i2 = i;
					i2[axis] = row;
					otherProduct *= parent.getValue(graph)[i2];
				}

			parent.getGradient(graph)[i] += this.gradient[j] * otherProduct;
		}

		foreach (ref g; this.gradient.valueIterator)
			g = 0;
	}

	static assert(isTensor!(typeof(this)));
}

Multiply!(Parent, axis) multiply(AxisIndex axis = 0, Parent)(Parent parent)
if (isTensor!Parent)
{
	return Multiply!(Parent, axis)();
} /// ditto

unittest
{
	float[2][1] inputData = [[2f, 3f]];
	auto graph = graph(nullOptimizer(),
		inputData[].boxes
		.trainableInput
		.multiply
	);

	assert(graph.forward(inputData[0].box).valueIterator.front == 6f);

	// float label = 24f;
	// graph.testGradient(label.box);
	// assert(graph.tensors[0].value.valueIterator == [4f, 6f]);

	// label = -24f;
	// graph.testGradient(label.box);
	// assert(graph.tensors[0].value.valueIterator == [-4f, 6f]);
}


// ----------------------------------------------------------------------------


struct Concatenate(AxisIndex axis, _Parents...)
if (allSatisfy!(isTensor, _Parents))
{
	alias Parents = _Parents;

	private alias TensorValueType(Tensor) = Tensor.Value.T;
	alias T = CommonType!(staticMap!(TensorValueType, Parents));

	enum name = tensorGroupName!Parents ~ ".concatenate";

	private enum tensorShape(Tensor) = Tensor.Value.shape;
	private enum outputShape = Shape.concatenate(axis, staticMap!(tensorShape, Parents));

	alias Value = DenseBox!(T, outputShape);
	private Value value;
	ref Value getValue(Graph)(ref Graph graph) { return value; } /// ditto

	static if (anySatisfy!(isTrainable, Parents))
	{
		private Value gradient;
		ref typeof(gradient) getGradient(Graph)(ref Graph graph) { return gradient; } /// ditto
	}

	void forward(Graph)(ref Graph graph) @nogc
	{
		size_t offset; // along `axis`
		static foreach (pi; 0 .. Parents.length) // https://issues.dlang.org/show_bug.cgi?id=21927
		{{
			auto parent = &graph.tensorInstance!(Parents[pi]);
			foreach (i; parent.getValue(graph).indexIterator)
			{
				auto j = Index!outputShape(i);
				j[axis] += offset;
				value[j] = parent.getValue(graph)[i];
			}
			enum step = Parents[pi].Value.shape.dims[axis]; // https://issues.dlang.org/show_bug.cgi?id=21871
			offset += step;
		}}
	}

	static if (anySatisfy!(isTrainable, Parents))
	void backward(Graph)(ref Graph graph)
	{
		size_t offset; // along `axis`
		static foreach (pi; 0 .. Parents.length) // https://issues.dlang.org/show_bug.cgi?id=21927
		{{
			auto parent = &graph.tensorInstance!(Parents[pi]);
			foreach (i; parent.getValue(graph).indexIterator)
			{
				auto j = Index!outputShape(i);
				j[axis] += offset;
				static if (isTrainable!(typeof(parent)))
					parent.getGradient(graph)[i] += gradient[j];
				gradient[j] = 0;
			}
			enum step = Parents[pi].Value.shape.dims[axis]; // https://issues.dlang.org/show_bug.cgi?id=21871
			offset += step;
		}}
	}

	static assert(isTensor!(typeof(this)));
}

Concatenate!(axis, Parents) concatenate(AxisIndex axis = 0, Parents...)(Parents parents)
if (allSatisfy!(isTensor, Parents))
{
	return Concatenate!(axis, Parents)();
} /// ditto

unittest
{
	float[2][1] input1 = [[1f, 2f]];
	float[1][1] input2 = [[3f]];
	auto graph = graph(nullOptimizer(),
		concatenate!0
		(
			input1[].boxes.input,
			input2[].boxes.input,
		)
	);

	assert(graph.forward(input1[0].box, input2[0].box).valueIterator == [1, 2, 3]);
}

unittest
{
	float[2][1] input1 = [[1f, 2f]];
	float[1][1] input2 = [[3f]];
	auto graph = graph(nullOptimizer(),
		concatenate!0
		(
			input1[].boxes.trainableInput,
			input2[].boxes.trainableInput,
		)
		.add!0
	);

	assert(graph.forward(input1[0].box, input2[0].box).valueIterator == [6]);

	// float label = 9f;
	// graph.testGradient(label.box);
	// assert(graph.tensors[0].value.valueIterator == [2f, 3f]);
	// assert(graph.tensors[1].value.valueIterator == [4f]);
}


// ----------------------------------------------------------------------------


/// Reduce the dimensionality of a tensor by taking just one row/column from a given axis.
struct SliceOne(AxisIndex axis, size_t index, Parent)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	enum name = Parent.name ~ ".sliceOne";

	private enum outputShape = Parent.Value.shape.dropAxis(axis);

	alias Value = DenseBox!(Parent.Value.T, outputShape);
	private Value value;
	ref Value getValue(Graph)(ref Graph graph) { return value; }

	void forward(Graph)(ref Graph graph)
	{
		auto parent = &graph.tensorInstance!Parent;
		foreach (i; parent.getValue(graph).indexIterator)
			if (i[axis] == index)
				value[i.dropAxis!axis] = parent.getValue(graph)[i];
	}

	static if (isTrainable!Parent)
	{
		private Value gradient;
		ref typeof(gradient) getGradient(Graph)(ref Graph graph) { return gradient; }

		void backward(Graph)(ref Graph graph)
		{
			auto parent = &graph.tensorInstance!Parent;

			foreach (i; parent.getGradient(graph).indexIterator)
				if (i[axis] == index)
				{
					auto j = i.dropAxis!axis;
					parent.getGradient(graph)[i] += this.gradient[j];
					this.gradient[j] = 0;
				}
		}
	}

	static assert(isTensor!(typeof(this)));
	static assert(isTrainable!(typeof(this)) == isTrainable!Parents);
}

SliceOne!(axis, index, Parent) sliceOne(AxisIndex axis, size_t index, Parent)(Parent parent)
if (isTensor!Parent)
{
	return SliceOne!(axis, index, Parent)();
} /// ditto


// ----------------------------------------------------------------------------

private auto maybeByRef(Box)(Box box) if (isBox!Box) { return box; }
private auto maybeByRef(Box)(ref Box box) if (isBox!Box) { return box.byRef; }


/// Increases the dimensionality of `Parent` by inserting axes with the given shape at the given position.
struct Repeat(Parent, Shape shape, AxisIndex where)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	enum name = Parent.name ~ ".repeat";

	alias Value = typeof(shapes.repeat!(shape, where)(Parent.Value.init));
	auto getValue(Graph)(ref Graph graph)
	{
		return graph.tensorInstance!Parent.getValue(graph).maybeByRef.repeat!(shape, where)();
	}

	void forward(Graph)(ref Graph graph) {}

	static if (isTrainable!Parent)
	{
		ref auto getGradient(Graph)(ref Graph graph)
		{
			return graph.tensorInstance!Parent.getGradient(graph).maybeByRef.repeat!(shape, where)();
		}
		void backward(Graph)(ref Graph graph) {}
	}

	static assert(isTensor!(typeof(this)));
	static assert(isTrainable!(typeof(this)) == isTrainable!Parent);
}

Repeat!(Parent, shape, where) repeat(Shape shape, AxisIndex where = 0, Parent)(Parent parent)
if (isTensor!Parent)
{
	return Repeat!(Parent, shape, where)();
} /// ditto

Repeat!(Parent, Shape([n]), where) repeat(size_t n, AxisIndex where = 0, Parent)(Parent parent)
if (isTensor!Parent)
{
	return Repeat!(Parent, Shape([n]), where)();
} /// ditto


// ----------------------------------------------------------------------------


struct SwapAxes(Parent, AxisIndex axis1, AxisIndex axis2)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	enum name = Parent.name ~ ".swapAxes";

	alias Value = shapes.SwapAxes!(Parent.Value, axis1, axis2);
	private Value value;
	ref Value getValue(Graph)(ref Graph graph) { return value; }

	void forward(Graph)(ref Graph graph)
	{
		auto parent = &graph.tensorInstance!Parent;
		value.value = parent.getValue(graph);
	}

	static if (isTrainable!Parent)
	{
		private Value gradient;
		ref typeof(gradient) getGradient(Graph)(ref Graph graph) { return gradient; }

		void backward(Graph)(ref Graph graph)
		{
			auto parent = &graph.tensorInstance!Parent;
			foreach (i; parents[0].gradient.indexIterator)
				parents[0].gradient[i] += this.gradient[i.swapAxes!(axis1, axis2)];
			foreach (ref g; this.gradient.value.valueIterator)
				g = 0;
		}
	}

	static assert(isTensor!(typeof(this)));
	static assert(isTrainable!(typeof(this)) == isTrainable!Parent);
}

SwapAxes!(Parent, axis1, axis2) swapAxes(AxisIndex axis1, AxisIndex axis2, Parent)(Parent parent)
if (isTensor!Parent)
{
	return SwapAxes!(Parent, axis1, axis2)();
} /// ditto


// ----------------------------------------------------------------------------


/// Transform values across axes after `firstAxis` according to a
/// trainable dense linear layer.
auto linearDense(Shape outputShape, AxisIndex firstAxis = 1, Parent)(Parent parent)
{
	// Divide input dimensions according to those which will be
	// preserved (e.g. the batch size) and those that will be
	// transformed.  By default, assume first axis is the batch size.
	enum batchShape = Shape(Parent.Value.shape.dims[0 .. firstAxis]);
	enum inputShape = Shape(Parent.Value.shape.dims[firstAxis .. $]);

	auto weights = Variable!(DenseBox!(Parent.Value.T, Shape(outputShape.dims ~ inputShape.dims)), Parent.name ~ ".dense-weights")();
	auto biases  = Variable!(DenseBox!(Parent.Value.T, outputShape                              ), Parent.name ~ ".dense-biases" )();
	return
		concatenate(
			concatenate(
				parent
				// batchShape x inputShape
				.repeat!(outputShape, batchShape.dims.length)
				// batchShape x outputShape x inputShape
				.repeat!1 // second multiplicand
				// 1 x batchShape x outputShape x inputShape
				,

				weights
				// outputShape x inputShape
				.repeat!batchShape // insert batch size axis
				// batchShape x outputShape x inputShape
				.repeat!1 // first multiplicand
				// 1 x batchShape x outputShape x inputShape
				,
			)
			// 2 x batchShape x outputShape x inputShape
			.multiply
			// batchShape x outputShape x inputShape
			.add!(iota(
					batchShape.dims.length + outputShape.dims.length,
					batchShape.dims.length + outputShape.dims.length + inputShape.dims.length
				).array)
			// batchShape x outputShape
			.repeat!1 // first addend
			// 1 x batchShape x outputShape
			,

			biases
			// outputShape
			.repeat!batchShape // insert batch size axis
			// batchShape x outputShape
			.repeat!1 // last addend
			// 1 x batchShape x outputShape
			,
		)
		// 2 x batchShape x outputShape
		.add
		// batchShape x outputShape
	;
}

auto linearDense(size_t numOutputs, AxisIndex firstAxis = 1, Parent)(Parent parent) { return linearDense!(Shape([numOutputs]), firstAxis)(parent); }


// ----------------------------------------------------------------------------


/// Layer template for a unary function.
template Unary(alias forwardFunc, alias backwardFunc, string _name)
{
	struct Unary(Parent)
	if (isTensor!Parent)
	{
		alias Parents = AliasSeq!Parent;
		enum name = Parent.name ~ "." ~ _name;

		alias Value = Parent.Value;
		private Value value;
		ref Value getValue(Graph)(ref Graph graph) { return value; } /// ditto

		void forward(Graph)(ref Graph graph)
		{
			auto parent = &graph.tensorInstance!Parent;
			foreach (i; parent.getValue(graph).indexIterator)
			{
				value[i] = forwardFunc(parent.getValue(graph)[i]);
				debug (drain_verbose)
					writefln("%s: %s(%s) = %s", i, _name, parents[0].getValue(graph)[i], value[i]);
			}
		}

		static if (isTrainable!Parent)
		{
			private Value gradient;
			ref typeof(gradient) getGradient(Graph)(ref Graph graph) { return gradient; }

			void backward(Graph)(ref Graph graph)
			{
				auto parent = &graph.tensorInstance!Parent;
				foreach (i; gradient.indexIterator)
					parent.getGradient(graph)[i] += backwardFunc(parent.getValue(graph)[i], this.value[i], this.gradient[i]);
				foreach (ref g; this.gradient.valueIterator)
					g = 0;
			}
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


/// Exponentiation function.
T expBackward(T)(T input, T output, T gradient) { return output * gradient; }
alias Exp = Unary!(std.math.exp, expBackward, "exp"); /// ditto
alias exp = unary!(std.math.exp, expBackward, "exp"); /// ditto


/// Reciprocal (1/x).
T reciprocalForward(T)(T input) { return T(1) / input; }
T reciprocalBackward(T)(T input, T output, T gradient) { return gradient * -(T(1) / (input * input)); } /// ditto
alias Reciprocal = Unary!(reciprocalForward, reciprocalBackward, "reciprocal"); /// ditto
alias reciprocal = unary!(reciprocalForward, reciprocalBackward, "reciprocal"); /// ditto


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
		static if (__traits(hasMember, typeof(tensor), q{value}))
			foreach (ref v; tensor.value.valueIterator)
				v = uniform01!float;

	foreach (epoch; 0 .. 1024 * 4 / numObservations)
	{
		debug (drain_verbose)
		{
			writefln("\n=== Epoch %d ===", epoch);
			foreach (ref tensor; graph.tensors)
				static if (isOptimizable!(typeof(tensor)))
					writeln(tensor.name, ": ", tensor.value.valueIterator);
		}
		foreach (i; numObservations.iota/*.randomCover*/)
		{
			debug (drain_verbose) { import std.stdio; writefln("--- %s -> %s :", inputs[i], labels[i]); }
			graph.forward (inputs[i].box);
			graph.backward(labels[i].box);
		}
	}

	foreach (i; 0 .. numObservations)
	{
		auto output = graph.forward(inputs[i].box).valueIterator.front;
		auto label = labels[i].box.valueIterator.front;
		debug (drain_verbose) { import std.stdio; writefln("%s -> %s / %s", inputs[i], output, label); }
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

	auto graph = graph(adam(),
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
	auto graph = graph(adam(),
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

	auto graph = graph(adam(),
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

	auto graph = graph(adam(),
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
	rndGen.seed(1);

	enum numSamples = 4;

	float[2][1][numSamples] inputs;
	float[1][1][numSamples] labels;
	foreach (i; 0 .. numSamples)
	{
		inputs[i][0] = [i % 2, i / 2 % 2];
		labels[i][0] = [inputs[i][0][0] != inputs[i][0][1]];
	}

	auto graph = graph(adam(constant!0.01f()),
		inputs[].boxes
		.input
		.linearDense!4
		.relu
		.linearDense!1
	);
	testProblem(graph, inputs, labels);
}


// ----------------------------------------------------------------------------


/// Layer which calculates the weighted average from the input.
/// Params:
///  aggregationAxis = Indicates the axis index along which the sum will be calculated.
///  roleAxis        = Indicates the axis which distinguishes the value and the weight.
///                    Its length should be 2.
struct WeightedAverage(Parent, AxisIndex aggregationAxis = 1, AxisIndex roleAxis = 2)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	enum name = Parent.name ~ ".weightedAverage";

	private enum Role : size_t
	{
		value = 0,
		weight = 1,
	}
	enum numRoles = /*enumLength!Role*/ 2;

	static assert(aggregationAxis < roleAxis);
	static assert(Parent.Value.shape.dims[roleAxis] == numRoles);
	private enum axes = [aggregationAxis, roleAxis];
	enum Shape shape = Parent.Value.shape.dropAxes(axes);

	alias Value = DenseBox!(Parent.Value.T, shape);
	private Value value;
	ref Value getValue(Graph)(ref Graph graph) { return value; } /// ditto

	void forward(Graph)(ref Graph graph)
	{
		auto parent = &graph.tensorInstance!Parent;
	//	auto values  = parent.getValue(graph).byRef.sliceOne!(roleAxis, Role.value);
		auto weights = parent.getValue(graph).byRef.sliceOne!(roleAxis, Role.weight);

		DenseBox!(Parent.Value.T, shape) weightsSum = weights.fold!aggregationAxis(sum);

		foreach (ref v; value.valueIterator)
			v = 0;
		foreach (i; parent.getValue(graph).indexIterator)
		{
			auto role = i[roleAxis];
			if (role != Role.weight)
				continue;

			auto iValue = i; iValue[roleAxis] = Role.value;

			auto weight = parent.getValue(graph)[i];
			auto value = parent.getValue(graph)[iValue];

			auto j = i.dropAxis!roleAxis;
			auto k = j.dropAxis!aggregationAxis;
			auto weightSum = weightsSum[k] + this.value.T.epsilon;
			this.value[k] += value * (weight / weightSum);
		}
	}

	static if (isTrainable!Parent)
	{
		private Value gradient;
		ref typeof(gradient) getGradient(Graph)(ref Graph graph) { return gradient; }

		void backward(Graph)(ref Graph graph)
		{
			auto parent = &graph.tensorInstance!Parent;
		//	auto values  = parent.getValue(graph).byRef.sliceOne!(roleAxis, Role.value);
			auto weights = parent.getValue(graph).byRef.sliceOne!(roleAxis, Role.weight);

			DenseBox!(Parent.Value.T, shape) weightsSum = weights.fold!aggregationAxis(sum);

			// Calculate `weight * value` (intermediate result)
			DenseBox!(Parent.Value.T, Parent.Value.shape.dropAxis(roleAxis)) weightsValuesProd;
			foreach (i; parent.getValue(graph).indexIterator)
			{
				auto role = i[roleAxis];
				if (role != Role.weight)
					continue;
				auto iWeight = i;
				auto iValue = i; iValue[roleAxis] = Role.value;
				auto j = i.dropAxis!roleAxis;
				weightsValuesProd[j] = parent.getValue(graph)[iValue] * weights[j];
			}
			DenseBox!(Parent.Value.T, shape) weightsValuesProdSum = weightsValuesProd.fold!aggregationAxis(sum);

			foreach (i; parent.getValue(graph).indexIterator)
			{
				auto j = i.dropAxis!roleAxis;
				auto k = j.dropAxis!aggregationAxis;

				auto role = i[roleAxis];
				final switch (role)
				{
					case Role.value:
						parent.getGradient(graph)[i] += gradient[k] * weights[j] / (weightsSum[k] + value.T.epsilon);
						break;

					case Role.weight:
					{
						auto iValue = i; iValue[roleAxis] = Role.value;
						auto value = parent.getValue(graph)[iValue];

						auto g = - (weightsSum[k] - weights[j]) * value;
						g += weightsValuesProdSum[k] - (value * weights[j]);
						g /= weightsSum[k] ^^ 2;
						g = -g;
						g *= gradient[k];
						if (g != g) // nan
							g = 0;

						parent.getGradient(graph)[i] += g;

						break;
					}
				}
			}

			foreach (ref g; this.gradient.valueIterator)
				g = 0;
		}
	}

	static assert(isTensor!(typeof(this)));
	static assert(isTrainable!(typeof(this)) == isTrainable!Parent);
}

WeightedAverage!(Parent, aggregationAxis, roleAxis) weightedAverage(AxisIndex aggregationAxis = 1, AxisIndex roleAxis = 2, Parent)(Parent parent)
{
	return WeightedAverage!(Parent, aggregationAxis, roleAxis)();
}/// ditto


unittest
{
	rndGen.seed(5);

	enum numFeatures  =   3; // Presence (0 or 1), prediction (0 or 1), and confidence [0,1]
	enum numTimesteps = 128; // N of observations, max number of items in the set
	enum numSamples   = 256; // Training data size

	float[numFeatures][numTimesteps][numSamples] inputs;
	float                           [numSamples] labels;
	foreach (i; 0 .. numSamples)
	{
		auto result = uniform!ubyte() % 2;

		// auto numPopulatedTimesteps = uniform(numTimesteps / 2, numTimesteps);
		auto numPopulatedTimesteps = uniform!uint % (numTimesteps / 2) + (numTimesteps / 2);

		foreach (j; 0 .. numTimesteps)
		{
			float presence, prediction, confidence;
			if (j < numPopulatedTimesteps)
			{
				presence = 1;
				confidence = uniform01!float;

				if (uniform01!float < confidence)
					prediction = result; // Truth
				else
					prediction = 1 - result;
			}
			else
			{
				presence = 0;
				prediction = 0;
				confidence = 0;
			}

			inputs[i][j] = [presence, prediction, confidence];
		}
		labels[i] = result;
	}

	auto graph = graph(adam(),
		inputs[].boxes
		.input
		// timesteps x features
		.linearDense!(2, 1)
		// timesteps x (value, weight)
		.sigmoid
		.weightedAverage!(0, 1)
	);
	testProblem(graph, inputs, labels);
}


// ----------------------------------------------------------------------------


/// Layer which calculates the softmax weighted average from the input.
/// Params:
///  aggregationAxis = Indicates the axis index along which the sum will be calculated.
///  roleAxis        = Indicates the axis which distinguishes the value and the weight.
///                    Its length should be 2.
struct SoftmaxWeightedAverage(Parent, AxisIndex aggregationAxis = 1, AxisIndex roleAxis = 2)
if (isTensor!Parent)
{
	alias Parents = AliasSeq!Parent;
	enum name = Parent.name ~ ".softmaxWeightedAverage";

	private enum Role : size_t
	{
		value = 0,
		weight = 1,
	}
	enum numRoles = /*enumLength!Role*/ 2;

	static assert(aggregationAxis < roleAxis);
	static assert(Parent.Value.shape.dims[roleAxis] == numRoles);
	private enum axes = [aggregationAxis, roleAxis];
	enum Shape shape = Parent.Value.shape.dropAxes(axes);

	alias Value = DenseBox!(Parent.Value.T, shape);
	private Value value;
	ref Value getValue(Graph)(ref Graph graph) { return value; } /// ditto

	void forward(Graph)(ref Graph graph)
	{
		auto parent = &graph.tensorInstance!Parent;
		DenseBox!(Parent.Value.T, Parent.Value.shape.dropAxis(roleAxis)) weightsExp;
		foreach (i; parent.getValue(graph).indexIterator)
		{
			auto role = i[roleAxis];
			if (role != Role.weight)
				continue;
			auto j = i.dropAxis!roleAxis;
			weightsExp[j] = exp(parent.getValue(graph)[i]);
		}

		DenseBox!(Parent.Value.T, shape) weightsExpSum = weightsExp.fold!aggregationAxis(sum);

		foreach (ref v; value.valueIterator)
			v = 0;
		foreach (i; parent.getValue(graph).indexIterator)
		{
			auto role = i[roleAxis];
			if (role != Role.weight)
				continue;

			auto iValue = i; iValue[roleAxis] = Role.value;

			auto weight = parent.getValue(graph)[i];
			auto value = parent.getValue(graph)[iValue];

			auto j = i.dropAxis!roleAxis;
			auto weightExp = weightsExp[j];

			auto k = j.dropAxis!aggregationAxis;
			auto weightExpSum = weightsExpSum[k] + this.value.T.epsilon;
			this.value[k] += value * (weightExp / weightExpSum);
		}
	}

	static if (isTrainable!Parent)
	{
		private Value gradient;
		ref typeof(gradient) getGradient(Graph)(ref Graph graph) { return gradient; }

		void backward(Graph)(ref Graph graph)
		{
			auto parent = &graph.tensorInstance!Parent;

			DenseBox!(Parent.Value.T, Parent.Value.shape.dropAxis(roleAxis)) weightsExp;
			foreach (i; parent.getValue(graph).indexIterator)
			{
				auto role = i[roleAxis];
				if (role != Role.weight)
					continue;
				auto j = i.dropAxis!roleAxis;
				weightsExp[j] = exp(parent.getValue(graph)[i]);
			}

			DenseBox!(Parent.Value.T, shape) weightsExpSum = weightsExp.fold!aggregationAxis(sum);

			// Calculate `exp(weight) * value` (intermediate result)
			DenseBox!(Parent.Value.T, Parent.Value.shape.dropAxis(roleAxis)) weightsExpValuesProd;
			foreach (i; parent.getValue(graph).indexIterator)
			{
				auto role = i[roleAxis];
				if (role != Role.weight)
					continue;
				auto iWeight = i;
				auto iValue = i; iValue[roleAxis] = Role.value;
				auto j = i.dropAxis!roleAxis;
				weightsExpValuesProd[j] = parent.getValue(graph)[iValue] * weightsExp[j];
			}

			foreach (i; parent.getValue(graph).indexIterator)
			{
				auto j = i.dropAxis!roleAxis;
				auto k = j.dropAxis!aggregationAxis;

				auto role = i[roleAxis];
				final switch (role)
				{
					case Role.value:
						parent.getGradient(graph)[i] += gradient[k] * weightsExp[j] / (weightsExpSum[k] + value.T.epsilon);
						break;

					case Role.weight:
					{
						auto iValue = i; iValue[roleAxis] = Role.value;
						auto value = parent.getValue(graph)[iValue];

						auto g = - (weightsExpSum[k] - weightsExp[j]) * value;
						g += weightsExpValuesProd[j] - (value * weightsExp[j]);
						g /= weightsExpSum[k] ^^ 2;
						g *= weightsExp[j];
						g = -g;
						g *= gradient[k];
						if (g != g) // nan
							g = 0;

						parent.getGradient(graph)[i] += g;

						break;
					}
				}
			}

			foreach (ref g; this.gradient.valueIterator)
				g = 0;
		}
	}

	static assert(isTensor!(typeof(this)));
	static assert(isTrainable!(typeof(this)) == isTrainable!Parent);
}

SoftmaxWeightedAverage!(Parent, aggregationAxis, roleAxis) softmaxWeightedAverage(AxisIndex aggregationAxis = 1, AxisIndex roleAxis = 2, Parent)(Parent parent)
{
	return SoftmaxWeightedAverage!(Parent, aggregationAxis, roleAxis)();
}/// ditto


// /// Calculate the softmax weighted average from the input.
// /// Params:
// ///  aggregationAxis = Indicates the axis index along which the sum will be calculated.
// ///  roleAxis        = Indicates the axis which distinguishes the value and the weight.
// ///                    Its length should be 2.
// auto softmaxWeightedAverage(AxisIndex aggregationAxis = 1, AxisIndex roleAxis = 2, Parent)(Parent parent)
// {
// 	enum Role : size_t
// 	{
// 		value = 0,
// 		weight = 1,
// 	}

// 	static assert(aggregationAxis < roleAxis);
// 	static assert(Parent.Value.shape.dims[roleAxis] == 2);
// 	enum axes = [aggregationAxis, roleAxis];
// 	enum Shape shape = Parent.Value.shape.dropAxes(axes);

// 	auto values  = parent.sliceOne!(roleAxis, Role.value );
// 	auto weights = parent.sliceOne!(roleAxis, Role.weight);
// 	auto weightsExp = weights.exp;

// 	enum inputSize = Parent.Value.shape.dims[aggregationAxis];

// 	return
// 		concatenate(
// 			values
// 			// batch x aggregation x datum
// 			.repeat!1
// 			// 1 x batch x aggregation x datum
// 			,

// 			weightsExp
// 			// batch x aggregation x datum
// 			.repeat!1
// 			// 1 x batch x aggregation x datum
// 			,

// 			weightsExp
// 			// batch x aggregation x datum
// 			.add!aggregationAxis
// 			// batch x datum
// 			.reciprocal
// 			// batch x datum
// 			.repeat!(inputSize, aggregationAxis)
// 			// batch x aggregation x datum
// 			.repeat!1
// 			// 1 x batch x aggregation x datum
// 			,
// 		)
// 		// 3 x batch x aggregation x datum
// 		.multiply
// 		// batch x aggregation x datum
// 		.add!aggregationAxis
// 		// batch x datum
// 	;
// }

unittest
{
	rndGen.seed(1);

	enum numFeatures   = 3; // Presence (0 or 1), prediction (0 or 1), and confidence [0,1]
	enum numTimesteps = 64; // N of observations, max number of items in the set
	enum numSamples =  512; // Training data size

	float[numFeatures][numTimesteps][numSamples] inputs;
	float                           [numSamples] labels;
	foreach (i; 0 .. numSamples)
	{
		auto result = uniform!ubyte() % 2;

		// auto numPopulatedTimesteps = uniform(numTimesteps / 2, numTimesteps);
		auto numPopulatedTimesteps = uniform!uint % (numTimesteps / 2) + (numTimesteps / 2);

		foreach (j; 0 .. numTimesteps)
		{
			float presence, prediction, confidence;
			if (j < numPopulatedTimesteps)
			{
				presence = 1;
				confidence = uniform01!float;

				if (uniform01!float < confidence)
					prediction = result; // Truth
				else
					prediction = 1 - result;
			}
			else
			{
				presence = 0;
				prediction = 0;
				confidence = 0;
			}

			inputs[i][j] = [presence, prediction, confidence];
		}
		labels[i] = result;
	}

	auto graph = graph(adam(),
		inputs[].boxes
		.input
		// timesteps x features
		.linearDense!(2, 1)
		// timesteps x (value, weight)
		.softmaxWeightedAverage!(0, 1)
		.sigmoid
	);
	testProblem(graph, inputs, labels);
}
