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
	immutable size_t[] dims;

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
private T[n+m] sconcat(T, size_t n, size_t m)(ref const T[n] a, ref const T[m] b)
{
	T[n+m] result;
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

// /// Static array UFCS primitives.
// ref ElementTypeOfStaticArray!T[shapeOfStaticArray!T.count()] valueIterator(T)(ref return T a)
// {
// 	return *cast(typeof(return)*)&a;
// }

// unittest
// {
// 	float[2][2] a = [[1,2],[3,4]];
// 	assert(a.valueIterator == [1,2,3,4]);

// 	float b = 5;
// 	assert(b.valueIterator == [5]);
// }

// /// ditto
// ShapeIterator!(shapeOfStaticArray!T) indexIterator(T)(ref T a)
// {
// 	return typeof(return)();
// }

// unittest
// {
// 	import std.algorithm.comparison : equal;

// 	float[2][3] a;
// 	assert(a.indexIterator.equal([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]]));

// 	float b = 5;
// 	assert(b.indexIterator.equal([[]]));
// }



// ----------------------------------------------------------------------------
// Box types
// ----------------------------------------------------------------------------

/// A box is logically an n-dimensional array. Also called a hyperrectangle:
/// https://en.wikipedia.org/wiki/Hyperrectangle
/// Boxes may actually be dense, sparse,
/// a function of some other box, or fully procedural.
enum isBox(Box) = true
	&& is(typeof(Box.shape) : Shape)
	&& __traits(hasMember, Box, q{T})
	&& __traits(hasMember, Box, q{valueIterator})
	&& __traits(hasMember, Box, q{indexIterator})
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

	/// Reference a particular element or sub-matrix by its `Index`.
	ref auto opIndex(Shape indexShape)(Index!indexShape index) inout
	if (shape.dims.startsWith(indexShape.dims))
	{
		static if (indexShape.dims.length == 0)
			return value;
		else
			return value[index.indices[0]][Index!(Shape(indexShape.dims[1 .. $]))(index.indices[1 .. $])];
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
