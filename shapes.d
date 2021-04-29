import std.algorithm.searching : startsWith;
import std.range;

// ----------------------------------------------------------------------------
// Shapes
// ----------------------------------------------------------------------------

/// Represents the shape of some hyperrectangle
/// (n-dimensional rectangle).
struct Shape
{
	/// Size in each dimension.
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
	Shape dropAxis(size_t axis)
	{
		return Shape(dims[0 .. axis] ~ dims[axis + 1 .. $]);
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
	enum shape = _shape; /// Parent `Shape`.
	size_t[shape.dims.length] indices; /// The indices along each dimension.
	alias indices this;

	/// Concatenate two indices to create an index within the shape
	/// that is the Cartesian product of this and `otherIndex`'s shape.
	auto opBinary(string op : "~", Shape otherShape)(Index!otherShape otherIndex)
	{
		enum Shape newShape = Shape(shape.dims ~ otherShape.dims);
		return Index!newShape(sconcat(indices, otherIndex.indices));
	}

	/// Return an `Index` with the given `axis` removed.
	Index!(shape.dropAxis(axis)) dropAxis(size_t axis)() const
	{
		return Index!(shape.dropAxis(axis))(sconcat(indices[0 .. axis], indices[axis + 1 .. $]));
	}
}

unittest
{
	auto i = Index!(Shape([5, 6, 7]))([1, 2, 3]);
	assert(i.dropAxis!1.indices == [1, 3]);
}

/// Iterates over all valid indices in `shape`.
struct ShapeIterator(Shape _shape)
{
	enum shape = _shape; /// Parent shape.
	Index!shape front; /// Range primitives.
	bool empty; /// ditto

	void popFront() nothrow @nogc
	{
		static foreach_reverse (dimIndex; 0 .. shape.dims.length)
		{{
			enum dimLength = shape.dims[dimIndex];
			if (++front.indices[dimIndex] == dimLength)
				front.indices[dimIndex] = 0;
			else
				return;
		}}
		empty = true;
	} /// ditto
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
	ref auto opIndex(Index!shape index) inout
	{
		static if (shape.dims.length == 0)
			return value;
		else
			return value[index.indices[0]][Index!(Shape(shape.dims[1 .. $]))(index.indices[1 .. $])];
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
/// `axis` using the supplied binary predicate, therefore removing it.
DenseBox!(Box.T, Box.shape.dropAxis(axis)) fold(size_t axis, Box, Pred)(const auto ref Box box, Pred pred)
{
	DenseBox!(Box.T, Box.shape.dropAxis(axis)) result;
	foreach (i; box.indexIterator)
	{
		if (i.indices[axis] == 0)
			result[i.dropAxis!axis] = box[i];
		else
			result[i.dropAxis!axis] = pred(result[i.dropAxis!axis], box[i]);
	}
	return result;
}

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

/// Nullary box wrapping a compile-time value.
struct Constant(_T, _T value)
{
	alias T = _T;
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

/// Adds a dimension to the front of `Box` with length `n`.
struct Repeat(Box, size_t n)
{
	alias T = Box.T;
	enum shape = Shape(n ~ Box.shape.dims);
	Box value;

	auto ref valueIterator() inout
	{
		return value.valueIterator;
	}

	auto indexIterator() const @nogc
	{
		version (none) // Not @nogc
		{
			import std.algorithm.iteration : map, joiner;
			return n.iota.map!(i => value.indexIterator.map!(vi => Index!shape(sconcat([i].staticArray, vi.indices)))).joiner;
		}
		else
		{
			struct Iterator
			{
				private size_t i; // this dimension
				private ShapeIterator!(Box.shape) nextIndex;
				bool empty;

				@property Index!shape front()
				{
					return Index!shape(sconcat([i].staticArray, nextIndex.front.indices));
				}

				void popFront()
				{
					assert(!empty);
					nextIndex.popFront();
					if (nextIndex.empty)
					{
						nextIndex = typeof(nextIndex).init;
						i++;
						empty = i == n;
					}
				}
			}
			return Iterator.init;
		}
	}

	/// Reference a particular element by its `Index`.
	ref auto opIndex(Index!shape index) inout
	{
		return value[Index!(Shape(shape.dims[1 .. $]))(index.indices[1 .. $])];
	}
}
Repeat!(Box, n) repeat(size_t n, Box)(auto ref Box box) if (isBox!Box) { return Repeat!(Box, n)(box); } /// ditto

auto repeat(Shape shape, Box)(auto ref Box box)
if (isBox!Box)
{
	static if (shape.dims.length == 0)
		return box;
	else
		return repeat!(shape.dims[0])(repeat!(Shape(shape.dims[1 .. $]))(box));
} /// ditto

unittest
{
	auto b = constant!1.repeat!(Shape([2, 2]));
	assert(b[Index!(b.shape)([0, 0])] == 1);
	assert(b[Index!(b.shape)([1, 1])] == 1);
}
