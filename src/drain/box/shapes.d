module drain.box.shapes;

import std.algorithm.comparison;
import std.algorithm.iteration;
import std.algorithm.searching : startsWith;
import std.exception : assumeUnique;
import std.range;

// ----------------------------------------------------------------------------
// Shapes
// ----------------------------------------------------------------------------

/// Used to indicate the index of some axis (dimension) within a shape/index.
alias AxisIndex = size_t;

/// Represents the shape of some hyperrectangle
/// (n-dimensional rectangle).
struct Shape
{
	/// Size in each dimension.
	/// Order is outer-to-inner.
	immutable size_t[] dims = [];

	/// Total number of elements
	/// (product of all dimensions).
	@property size_t count() const
	{
		size_t result = 1;
		foreach (dim; dims)
			result *= dim;
		return result;
	}

	/// Return a `Shape` with the given `axis` removed.
	Shape dropAxis(AxisIndex axis)
	{
		return Shape(dims[0 .. axis] ~ dims[axis + 1 .. $]);
	}

	/// Return a `Shape` with the given `axes` removed.
	/// `axes` should be in ascending order.
	Shape dropAxes(scope AxisIndex[] axes)
	{
		if (axes.length == 0)
			return this;
		else
		{
			if (axes.length > 1)
				assert(axes[$-2] < axes[$-1]);
			return dropAxis(axes[$-1]).dropAxes(axes[0 .. $-1]);
		}
	}

	/// Return a `Shape` with the given shape's axes added at the
	/// given position.
	Shape addAxes(Shape shape, AxisIndex where)
	{
		return Shape(dims[0 .. where] ~ shape.dims ~ dims[where .. $]);
	}

	/// Return a `Shape` with the given axes swapped.
	Shape swapAxes(AxisIndex axis1, AxisIndex axis2)
	{
		import std.algorithm.mutation : swap;
		auto dims = this.dims.dup;
		swap(dims[axis1], dims[axis2]);
		return Shape(dims.assumeUnique);
	}

	/// Returns the shape resulting from the concatenation of the
	/// given shapes along the given axis.
	static Shape concatenate(AxisIndex axis, Shape[] shapes...)
	{
		assert(shapes.length > 0, "No shapes to concatenate");
		foreach (shape; shapes)
			assert(shape.dims.length == shapes[0].dims.length, "Dimensionality does not match for concatenate");
		assert(axis < shapes[0].dims.length, "Concatenation axis is out of bounds");
		size_t[] result;
		result.length = shapes[0].dims.length;
		foreach (AxisIndex i; 0 .. result.length)
			if (i == axis)
				result[i] = shapes.map!(shape => shape.dims[axis]).reduce!"a+b";
			else
			{
				foreach (shape; shapes)
					assert(shape.dims[i] == shapes[0].dims[i], "Non-concatenation axis mismatch");
				result[i] = shapes[0].dims[i];
			}
		return Shape(result.assumeUnique);
	}
}

// Concatenate two static arrays.
private T[n+m] sconcat(T, size_t n, size_t m)(auto ref const T[n] a, auto ref const T[m] b)
{
	T[n+m] result = void; // https://issues.dlang.org/show_bug.cgi?id=21876
	result[0 .. n] = a;
	result[n .. $] = b;
	return result;
}

/// Represents an index (coordinate) within
/// a `Shape`-defined hyperrectangle.
struct Index(Shape _shape)
{
	/// Parent `Shape`.
	enum shape = _shape;

	/// The indices along each dimension.
	/// As with `Shape.dims`, order is outer-to-inner.
	size_t[shape.dims.length] indices;
	alias indices this;

	/// Concatenate two indices to create an index within the shape
	/// that is the Cartesian product of this and `otherIndex`'s shape.
	auto opBinary(string op : "~", Shape otherShape)(Index!otherShape otherIndex)
	{
		enum Shape newShape = Shape(shape.dims ~ otherShape.dims);
		return Index!newShape(sconcat(indices, otherIndex.indices));
	}

	/// Return an `Index` with the given `axis` removed.
	Index!(shape.dropAxis(axis)) dropAxis(AxisIndex axis)() const
	{
		return Index!(shape.dropAxis(axis))(sconcat(indices[0 .. axis], indices[axis + 1 .. $]));
	}

	/// Return an `Index` with the given `axes` removed.
	/// `axes` should be in ascending order.
	Index!(shape.dropAxes(axes)) dropAxes(AxisIndex[] axes)()
	{
		static if (axes.length == 0)
			return this;
		else
		{
			static if (axes.length > 1)
				static assert(axes[$-2] < axes[$-1]);
			return dropAxis!(axes[$-1]).dropAxes!(axes[0 .. $-1]);
		}
	}

	/// Return an `Index` with the given index's axes added at the
	/// given position.
	Index!(shape.addAxes(indexShape, where)) addAxes(AxisIndex where, Shape indexShape)(Index!indexShape otherIndex)
	{
		return Index!(shape.addAxes(indexShape, where))(
			sconcat(
				sconcat(
					this.indices[0 .. where],
					otherIndex.indices,
				),
				this.indices[where .. $],
			)
		);
	}
}

unittest
{
	auto i = Index!(Shape([5, 6, 7]))([1, 2, 3]);
	assert(i.dropAxis!1 == [1, 3]);
}

/// The nullary index.
enum nullIndex = Index!(Shape.init).init;


/// Iterates over all valid indices in `shape`.
struct ShapeIterator(Shape _shape)
{
	enum shape = _shape; /// Parent shape.
	Index!shape front; /// Range primitives.

	bool empty = shape.count == 0; /// ditto

	void popFront() nothrow @nogc
	{
		static foreach_reverse (dimIndex; 0 .. shape.dims.length)
		{{
			enum dimLength = shape.dims[dimIndex];
			if (++front[dimIndex] == dimLength)
				front[dimIndex] = 0;
			else
				return;
		}}
		empty = true;
	} /// ditto
}

unittest
{
	enum shape = Shape([2]);
	assert(ShapeIterator!shape().equal([
		Index!shape([0]),
		Index!shape([1]),
	]));
}

unittest
{
	enum shape = Shape([]);
	assert(ShapeIterator!shape().equal([
		Index!shape([]),
	]));
}

unittest
{
	enum shape = Shape([0]);
	assert(ShapeIterator!shape().equal((Index!shape[]).init));
}


// ----------------------------------------------------------------------------
// Static array shape properties
// ----------------------------------------------------------------------------

/// Resolve to a static array of `T` with shape `Shape`.
template StaticArray(T, Shape shape)
{
	///
	static if (shape.dims.length == 0)
		alias StaticArray = T;
	else
		alias StaticArray = StaticArray!(T, Shape(shape.dims[1 .. $]))[shape.dims[0]];
}

/// Infer `Shape` from static array `T`.
template shapeOfStaticArray(T)
{
	///
	static if (is(T == E[n], E, size_t n))
		enum Shape shapeOfStaticArray = Shape(n ~ shapeOfStaticArray!E.dims);
	else
		enum Shape shapeOfStaticArray = Shape.init;
}

static assert(shapeOfStaticArray!int == Shape.init);
static assert(shapeOfStaticArray!(int[2]) == Shape([2]));
static assert(shapeOfStaticArray!(int[2][3]) == Shape([3, 2]));

/// Infer base type from static array `T`, so that
/// `T == StaticArray!(ElementTypeOfStaticArray!T, shapeOfStaticArray!T)`.
template ElementTypeOfStaticArray(T)
{
	///
	static if (is(T == E[n], E, size_t n))
		alias ElementTypeOfStaticArray = ElementTypeOfStaticArray!E;
	else
		alias ElementTypeOfStaticArray = T;
}

static assert(is(ElementTypeOfStaticArray!int == int));
static assert(is(ElementTypeOfStaticArray!(int[2]) == int));
static assert(is(ElementTypeOfStaticArray!(int[2][3]) == int));


// ----------------------------------------------------------------------------
// Box types
// ----------------------------------------------------------------------------

/// A box is logically an n-dimensional array. Also called a hyperrectangle:
/// https://en.wikipedia.org/wiki/Hyperrectangle
/// Boxes may actually be dense, sparse,
/// a function of some other box, or fully procedural.
enum isBox(Box) = true

	/// Box types should declare their shape.
	&& is(typeof(Box.shape) : Shape)

	/// Box types should declare their underlying type
	/// (after stripping away all dimensions in `shape`).
	&& is(Box.T)

	/// The value iterator returns an iterator over all set / unique values.
	&& is(typeof(Box.init.valueIterator.front) : Box.T)

	/// The index iterator returns an iterator over all valid indices in `Box`.
	&& typeof(Box.init.indexIterator.front).shape == Box.shape
;

/// ditto
enum isBoxOf(Box, T) = true
	&& isBox!Box
	&& is(typeof((*cast(Box*)null).valueIterator.front) == T)
;

// ----------------------------------------------------------------------------

/// A box implementation that owns an n-dimensional static array.
/// The static array is represented as nested `DenseBox` instances,
/// all the way to the bottom.
struct DenseBox(_T, Shape _shape)
{
	enum shape = _shape; /// Shape of this box from here.
	alias T = _T; /// Base type (after stripping all dimensions in `shape`).

	/// Wrapped value.
	// StaticArray!(Variable!T, shape) array;
	static if (shape.dims.length == 0)
		T value;
	else
		DenseBox!(T, Shape(shape.dims[1 .. $]))[shape.dims[0]] value;

	/// Return an object that can be iterated over to walk over all elements.
	@property ref T[shape.count()] valueIterator() inout
	{
		return *cast(T[shape.count()]*)&value;
	}

	/// Return an object that can be iterated over to get `opIndex` arguments for all elements.
	auto indexIterator() const
	{
		return ShapeIterator!shape();
	}

	/// Reference a particular element by its `Index`.
	ref inout(T) opIndex(Index!shape index) inout return
	{
		static if (shape.dims.length == 0)
			return value;
		else
			return value[index[0]][Index!(Shape(shape.dims[1 .. $]))(index[1 .. $])];
	}

	// Work-around for https://issues.dlang.org/show_bug.cgi?id=21878
	void opIndexAssign(T newValue, Index!shape index)
	{
		static if (shape.dims.length == 0)
			value = newValue;
		else
			value[index[0]][Index!(Shape(shape.dims[1 .. $]))(index[1 .. $])] = newValue;
	}
}
static assert(isBoxOf!(DenseBox!(float, Shape([1])), float));

/// Convert a static array to a `DenseBox`.
ref DenseBox!(ElementTypeOfStaticArray!T, shapeOfStaticArray!T) box(T)(ref return T array)
{
	static assert(isBox!(typeof(return)));
	return *cast(typeof(return)*)&array;
}

/// Convert a range of static arrays to `DenseBox`es.
auto boxes(R)(R range)
if (isInputRange!R)
{
	alias E = ElementType!R;
	alias T = ElementTypeOfStaticArray!E;
	enum shape = shapeOfStaticArray!E;
	alias Box = DenseBox!(T, shape);
	static if (is(R == E[]))
		return cast(Box[])range;
	else
		return range.map!((ref r) => cast(Box)r);
}

unittest
{
	float[2][2] a = [[1,2],[3,4]];
	assert(a.box.valueIterator == [1,2,3,4]);

	float b = 5;
	assert(b.box.valueIterator == [5]);
}

unittest
{
	import std.algorithm.comparison : equal;

	float[2][3] a;
	assert(a.box.indexIterator.equal([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]]));

	float b = 5;
	assert(b.box.indexIterator.equal([[]]));
}

unittest
{
	import std.algorithm.comparison : equal;
	import std.algorithm.iteration : map;

	float[2][] a = [[1,2],[3,4]];
	assert(a.boxes.map!(box => box.valueIterator).equal([[1,2],[3,4]]));

}

// ----------------------------------------------------------------------------

/// Reduce the dimensionality of `box` by folding all elements along
/// `axes` using the supplied binary predicate, therefore removing them.
DenseBox!(Box.T, Box.shape.dropAxes(axes)) fold(AxisIndex[] axes, Box, Pred)(const auto ref Box box, Pred pred)
if (isBox!Box)
{
	DenseBox!(Box.T, Box.shape.dropAxes(axes)) result;
	foreach (i; box.indexIterator)
	{
		bool first = true;
		static foreach (axis; axes)
			if (i[axis] != 0)
				first = false;
		if (first)
			result[i.dropAxes!axes] = box[i];
		else
			result[i.dropAxes!axes] = pred(result[i.dropAxes!axes], box[i]);
	}
	return result;
}

DenseBox!(Box.T, Box.shape.dropAxis(axis)) fold(AxisIndex axis, Box, Pred)(const auto ref Box box, Pred pred)
if (isBox!Box)
{
	return fold!([axis])(box, pred);
} /// ditto

/// Construct a predicate from a binary function.
// TODO move this somewhere proper.
auto binary(alias fun)()
{
	struct Pred
	{
		auto opCall(A, B)(auto ref A a, auto ref B b)
		{
			import std.functional : binaryFun;
			return binaryFun!fun(a, b);
		}
	}
	return Pred.init;
}

alias sum     = binary!"a+b"; /// Sum predicate.
alias product = binary!"a*b"; /// Product predicate.

unittest
{
	float[2][2] a = [[1,2],[3,4]];
	assert(a.box.fold!0(sum).valueIterator == [4,6]);
}

// ----------------------------------------------------------------------------

/// Wrap a pointer to `Box` into a box.
struct ByRef(Box)
if (isBox!Box)
{
	Box* value;
	alias T = Box.T;
	enum Shape shape = Box.shape;

	ref inout(T) opIndex(Index!shape index) inout return
	{
		return (*value)[index];
	}

	auto ref valueIterator() return inout
	{
		return value.valueIterator;
	}

	auto indexIterator() const { return value.indexIterator; }

	static assert(isBox!(typeof(this)));
}

auto byRef(Box)(ref Box box)
{
	return ByRef!Box(&box);
} /// ditto


// ----------------------------------------------------------------------------

/// Reduce the dimensionality of `box` by taking just one row/column from a given axis.
/// `AxisValue` is a nullary `size_t` box holding the axis value to slice.
struct SliceOne(Box, AxisIndex axis, AxisValue)
if (isBox!Box && isBoxOf!(AxisValue, size_t))
{
	Box value;
	alias T = Box.T;
	AxisValue axisValue;
	enum Shape shape = Box.shape.dropAxis(axis);
	private enum Shape slicedShape = Shape([Box.shape.dims[axis]]);

	ref inout(T) opIndex(Index!shape index) inout return
	{
		return value[Index!(Box.shape)(index.addAxes!axis(Index!slicedShape([axisValue[nullIndex]])))];
	}

	auto ref valueIterator() return
	{
		return indexIterator.map!(i => this[i]);
	}

	auto ref valueIterator() return const
	{
		return indexIterator.map!(i => this[i]);
	}

	auto indexIterator() const @nogc
	{
		struct Iterator
		{
			AxisValue axisValue;
			private ShapeIterator!(Box.shape) nextIndex;

			@property Index!shape front()
			{
				return nextIndex.front.dropAxis!axis();
			}

			@property bool empty() { return nextIndex.empty; }

			void popFront()
			{
				nextIndex.popFront();
				advance();
			}

			private void advance()
			{
				while (!nextIndex.empty && nextIndex.front[axis] != axisValue[nullIndex])
					nextIndex.popFront();
			}
		}
		auto i = Iterator(axisValue);
		i.advance();
		return i;
	}
}

auto sliceOne(AxisIndex axis, size_t axisValue, Box)(Box box)
{
	return SliceOne!(Box, axis, typeof(constant!axisValue()))(box);
} /// ditto

auto sliceOne(AxisIndex axis, Box)(Box box, size_t axisValue)
{
	auto v = variable(axisValue);
	return SliceOne!(Box, axis, typeof(v))(box, v);
} /// ditto

unittest
{
	float[2][2] a = [[1,2],[3,4]];
	assert(a.box      .sliceOne!(1, 1).valueIterator.equal([2,4]));
	assert(a.box      .sliceOne! 1 (1).valueIterator.equal([2,4]));
	assert(a.box.byRef.sliceOne!(1, 1).valueIterator.equal([2,4]));
	assert(a.box.byRef.sliceOne! 1 (1).valueIterator.equal([2,4]));
}

// ----------------------------------------------------------------------------

/// Nullary box wrapping a compile-time value.
struct Constant(_T, _T _value)
{
	alias T = _T;
	enum value = _value;
	enum Shape shape = Shape.init;
	T opIndex(Index!shape index) const { return value; }
	auto valueIterator() inout { return value.only; }
	auto indexIterator() const { return ShapeIterator!shape(); }
}
static assert(isBoxOf!(Constant!(int, 1), int));

Constant!(typeof(args[0]), args[0]) constant(args...)() if (args.length == 1) { return typeof(return)(); } /// ditto

/// Nullary box wrapping a run-time value.
struct Variable(_T)
{
	alias T = _T;
	T value;
	enum Shape shape = Shape.init;
	T opIndex(Index!shape index) const { return value; }
	auto valueIterator() inout return { return (&value)[0..1]; }
	auto indexIterator() const { return ShapeIterator!shape(); }
}
static assert(isBoxOf!(Variable!int, int));

Variable!T variable(T)(T value) { return Variable!T(value); } /// ditto

// ----------------------------------------------------------------------------

/// Increase the dimensionality of `box` by inserting dimensions at the given axis position.
struct Repeat(Box, Shape addedShape, AxisIndex where = 0)
if (isBox!Box)
{
	alias T = Box.T;
	enum shape = Box.shape.addAxes(addedShape, where);
	Box value;

	auto ref valueIterator() inout
	{
		return value.valueIterator;
	}

	auto indexIterator() const @nogc
	{
		struct Iterator
		{
			private ShapeIterator!(addedShape) thisIndex;
			private ShapeIterator!(Box.shape) nextIndex;

			@property Index!shape front()
			{
				return nextIndex.front.addAxes!where(thisIndex.front);
			}

			@property bool empty() { return thisIndex.empty || nextIndex.empty; }

			void popFront()
			{
				assert(!empty);
				nextIndex.popFront();
				if (nextIndex.empty)
				{
					nextIndex = typeof(nextIndex).init;
					thisIndex.popFront();
				}
			}
		}
		return Iterator.init;
	}

	/// Reference a particular element by its `Index`.
	ref auto opIndex(Index!shape index) inout
	{
		Index!(Box.shape) nextIndex = index.dropAxes!(iota(where, where + addedShape.dims.length).array);
		return value[nextIndex];
	}
}
Repeat!(Box, shape     , where) repeat(Shape shape, AxisIndex where = 0, Box)(auto ref Box box) if (isBox!Box) { return Repeat!(Box, shape     , where)(box); } /// ditto
Repeat!(Box, Shape([n]), where) repeat(size_t n   , AxisIndex where = 0, Box)(auto ref Box box) if (isBox!Box) { return Repeat!(Box, Shape([n]), where)(box); } /// ditto

unittest
{
	auto b = constant!1.repeat!2;
	assert(b[Index!(b.shape)([0])] == 1);
	assert(b[Index!(b.shape)([1])] == 1);
}

unittest
{
	auto b = constant!1.repeat!(Shape([2, 2]));
	assert(b[Index!(b.shape)([0, 0])] == 1);
	assert(b[Index!(b.shape)([1, 1])] == 1);
}

// ----------------------------------------------------------------------------

Index!(shape.swapAxes(axis1, axis2)) swapAxes(AxisIndex axis1, AxisIndex axis2, Shape shape)(Index!shape index)
{
	import std.algorithm.mutation : swap;
	size_t[shape.dims.length] indices = index.indices;
	swap(indices[axis1], indices[axis2]);
	return Index!(shape.swapAxes(axis1, axis2))(indices);
}

struct SwapAxes(Box, AxisIndex axis1, AxisIndex axis2)
if (isBox!Box)
{
	alias T = Box.T;
	enum shape = Box.shape.swapAxes(axis1, axis2);
	Box value;

	auto ref valueIterator() inout
	{
		return value.valueIterator;
	}

	auto indexIterator() const @nogc
	{
		import std.algorithm.iteration : map, joiner;
		return value.indexIterator.map!(swapAxes!(axis1, axis2, Box.shape));
	}

	ref auto opIndex(Index!shape index) inout
	{
		return value[index.swapAxes!(axis1, axis2)];
	}
}
SwapAxes!(Box, axis1, axis2) swapAxes(AxisIndex axis1, AxisIndex axis2, Box)(auto ref Box box) if (isBox!Box) { return SwapAxes!(Box, axis1, axis2)(box); } /// ditto

unittest
{
	int[2][2] a = [[1, 2], [3, 4]];
	auto b = a.box;
	assert(b[Index!(b.shape)([0, 1])] == 2);
	auto s = b.swapAxes!(0, 1);
	assert(s[Index!(s.shape)([0, 1])] == 3);
}
