import std.algorithm.iteration;
import std.array;
import std.math;
import std.stdio;

void main()
{
	auto inputs = [1f, 2f, 3f];
	auto labels = inputs.map!(i => i * 2 + 1).array;

	enum scale = 10f;
	inputs[] *= scale;
	labels[] *= scale;

	enum GradientAlgorithm
	{
		byTheBook,
		solver,
	}
	enum gradientAlgorithm = GradientAlgorithm.byTheBook;

	enum Optimizer
	{
		none,
		adaGrad,
		adam,
	}
	enum optimizer = Optimizer.adam;

	auto weight = 0.5f;
	auto bias   = 0.0f;

	enum eta = 0.1f;

	foreach (epoch; 0 .. 1000)
	{
		writefln("\n=== Epoch %d ===", epoch);
		writefln("weight = %s", weight);
		writefln("bias = %s", bias);

		foreach (i; 0 .. inputs.length)
		{
			auto input = inputs[i];
			auto label = labels[i];
			writefln("--- %s -> %s :", input, label);

			auto prod   = input * weight;
			writefln("%s * %s = %s", input, weight, prod);
			auto output = prod + bias;
			writefln("%s + %s = %s", prod, bias, output);

			float weightGrad, biasGrad;
			final switch (gradientAlgorithm)
			{
				case GradientAlgorithm.byTheBook:
				{
					auto error = (label - output) ^^ 2 / 2;
					writefln("error = %s", error);

					auto d_error_output = output - label;
					writefln("d_error_output = %s", d_error_output);

					auto d_output_prod = 1f;
					auto d_prod_weight = input;

					auto d_error_weight = d_error_output * d_output_prod * d_prod_weight;
					writefln("d_error_weight = %s", d_error_weight);

					weightGrad = d_error_weight;
					biasGrad   = d_error_output;
					break;
				}

				case GradientAlgorithm.solver:
				{
					auto error = label - output;
					writefln("error = %s", error);

					auto error_addend = error / 2;
					biasGrad   = error_addend;
					writefln("biasGrad = %s", biasGrad);
					weightGrad = error_addend / input;
					writefln("weightGrad = %s", weightGrad);

					auto output2 = input * (weight + weightGrad) + (bias + biasGrad);
					writefln("test output = %s", output2);
					assert(isClose(output2, label));
					break;
				}
			}
			// Simple

			final switch (optimizer)
			{
				case Optimizer.none:
				{
					weight -= eta * weightGrad;
					bias   -= eta * biasGrad;
					break;
				}

				case Optimizer.adaGrad:
				{
					void adaGrad(string name)(ref float value, float gradient)
					{
						static float m = 0;
						enum float eps = 1e-8;

						auto g = gradient;
						auto mn = m + g * g;
						auto diff = g / sqrt(mn + eps);
						m = mn;
						transform(diff, value, -eta, 1.0);

					}

					adaGrad!"weight"(weight, weightGrad);
					adaGrad!"bias"  (bias  , biasGrad);
					break;
				}

				case Optimizer.adam:
				{
					void adam(string name)(ref float value, float gradient)
					{
						static float m1 = 0;
						static float m2 = 0;

						enum float beta1 = 0.9;
						enum float beta2 = 0.999;
						enum float eps = 1e-8;

						auto g = gradient;
						auto nextM1 = (1.0 - beta1) * (g     - m1) + m1;
						auto nextM2 = (1.0 - beta2) * (g * g - m2) + m2;
						auto diff = nextM1 / sqrt(nextM2 + eps);
						m1 = nextM1;
						m2 = nextM2;
						transform(diff, value, -eta, 1.0);

					}

					adam!"weight"(weight, weightGrad);
					adam!"bias"  (bias  , biasGrad);
					break;
				}
			}
		}
	}
}

// Equivalent to: dst = (dst * beta) + (src * alpha)
void transform(T)(T src, ref T dst, T alpha = 1, T beta = 0)
{
    if (beta == 0)
	{
        dst = alpha * src;
        return;
    }
    if (beta != 1)
        dst = beta * dst;
    dst += alpha * src;
}
