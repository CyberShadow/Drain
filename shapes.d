/// Represents the shape of some hyperrectangle
/// (n-dimensional rectangle).
struct Shape
{
	/// Size in each dimension.
	size_t[] dims;

	/// Total number of elements
	/// (product of all dimensions).
	@property size_t count()
	{
		size_t result = 1;
		foreach (dim; dims)
			result *= dim;
		return result;
	}
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
	auto opBinary(string op : "~", Shape otherShape)(Index!otherShape otherIndex) @nogc
	{
		enum Shape newShape = Shape(shape.dims ~ otherShape.dims);
		size_t[newShape.dims.length] newIndices;
		newIndices[0 .. indices.length] = indices;
		newIndices[indices.length .. $] = otherIndex.indices;
		return Index!newShape(newIndices);
	}
}

/// Iterates over all valid indices in `shape`.
struct ShapeIterator(Shape shape)
{
	Index!shape front; /// Range primitives.
	bool empty; /// ditto

	void popFront()
	{
		foreach_reverse (dimIndex; 0 .. shape.dims.length)
		{
			if (++front.indices[dimIndex] == shape.dims[dimIndex])
				front.indices[dimIndex] = 0;
			else
				return;
		}
		empty = true;
	} /// ditto
}

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
		enum Shape shapeOfStaticArray = Shape([n] ~ shapeOfStaticArray!E.dims);
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
