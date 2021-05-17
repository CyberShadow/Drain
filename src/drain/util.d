module drain.util;

import std.meta;
import drain.autograd : isTensor;

// This is here only to avoid the @nogc restriction in drain.autograd.

package template tensorGroupName(Tensors...)
if (allSatisfy!(isTensor, Tensors))
{
	enum tensorGroupName = {
		assert(__ctfe);
		if (!Tensors.length)
			return "[]";
		string[Tensors.length] names;
		foreach (ti, Tensor; Tensors)
			names[ti] = Tensor.name;
		size_t prefixLength = 0;
	prefix:
		foreach (i; 0 .. names[0].length + 1)
		{
			if (i < names[0].length)
				foreach (name; names)
					if (i == name.length || name[i] != names[0][i])
						break prefix;
			if (i == names[0].length || names[0][i] == '.')
				prefixLength = i;
		}
		string result = names[0][0 .. prefixLength] ~ "[";
		foreach (i, name; names)
			result ~= (i ? ", " : "") ~ name[prefixLength .. $];
		result ~= "]";
		return result;
	}();
}
