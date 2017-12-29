#include <cstdio>
#include <cstdint>
#include <type_traits>
#include <cfloat>
#include <cmath>
#include <limits>
#include <vector>
#include <future>

// Intrinsics
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>


constexpr float min(float left, float right) noexcept
{
	return (left < right) ? left : right;
}

constexpr float max(float left, float right) noexcept
{
	return (left > right) ? left : right;
}


union Float_t
{
	float f;
	uint32_t i;

	struct
	{
		uint32_t mantissa : 23;
		uint32_t exponent : 8;
		uint32_t sign : 1;
	} parts;

	constexpr Float_t() : i(0) {}
	inline Float_t(float n) : f{n} {}
	inline Float_t(uint32_t n) : i{n} {}

	inline bool is_negative() const { return (i >> 31) != 0; }
	inline int32_t mantissa() const { return i & ((1 << 23) - 1); }
	inline int32_t exponent() const { return (i >> 23) & 0xFF; }

	inline Float_t make_positive() const { return {i & ~(1 << 31)}; }
	inline Float_t make_negative() const { return {i | (1 << 31)}; }
};


union alignas(16) Vector_t
{
	float f[4];
	uint32_t i[4];
	__m128 v;

	inline Vector_t(__m128 n) : v{n} {}
	constexpr Vector_t(float a, float b, float c, float d) : f{a, b, c, d} {}
	constexpr Vector_t(uint32_t a, uint32_t b, uint32_t c, uint32_t d) : i{a, b, c, d} {}

	inline operator const float*() const { return f; }
	inline operator __m128() const { return v; }
	inline operator __m128i() const { return _mm_castps_si128(v); }
	inline operator __m128d() const { return _mm_castps_pd(v); }
};

constexpr float float_pi = 3.1415927410125732421875f;
constexpr float float_half_pi = 1.57079637050628662109375f;
constexpr float float_inv_pi = 0.3183098733425140380859375f;
constexpr float float_two_pi = 6.283185482025146484375f;
constexpr float float_inv_two_pi = 0.15915493667125701904296875f;

const Vector_t vector_one = { 1.0f, 1.0f, 1.0f, 1.0f };
const Vector_t vector_negative_zero = { 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
const Vector_t vector_pi = { float_pi, float_pi, float_pi, float_pi };
const Vector_t vector_half_pi = { float_half_pi, float_half_pi, float_half_pi, float_half_pi };
const Vector_t vector_inv_pi = { float_inv_pi, float_inv_pi, float_inv_pi, float_inv_pi };
const Vector_t vector_two_pi = { float_two_pi, float_two_pi, float_two_pi, float_two_pi };
const Vector_t vector_inv_two_pi = { float_inv_two_pi, float_inv_two_pi, float_inv_two_pi, float_inv_two_pi };


using ScalarFunction = float (float);
using VectorFunction = __m128 (__m128);
using FloatRange = std::pair<Float_t, Float_t>;


float absolute_error(float actual, float approx)
{
	return std::abs(actual - approx);
}

__m128 absolute_error(__m128 actual, __m128 approx)
{
	__m128 value = _mm_sub_ps(actual, approx);
	__m128 result = _mm_sub_ps(_mm_setzero_ps(), value);
	return _mm_max_ps(result, value);
}

float relative_error(float actual, float approx)
{
	if (actual == 0.0f)
	{
		return approx;
	}
	return 1.0f - (approx / actual);
}

__m128 relative_error(__m128 actual, __m128 approx)
{
	__m128 zeroes = _mm_setzero_ps();
	__m128 ones = vector_one.v;
	__m128 mask = _mm_cmpeq_ps(zeroes, actual);
	__m128 other = _mm_and_ps(mask, approx);
	__m128 result = _mm_div_ps(approx, actual);
	result = _mm_andnot_ps(mask, result);
	result = _mm_sub_ps(ones, result);
	result = _mm_andnot_ps(mask, result);
	return _mm_or_ps(other, result);
}


struct Result
{
	float start;
	float end;
	float start_actual;
	float end_actual;
	float max_abs_error;
	float max_rel_error;
};

constexpr uint32_t bucket_count = 4096;

std::vector<Result> evaluate(ScalarFunction actual_function, ScalarFunction approx_function, Float_t start, Float_t end)
{
	const uint32_t count = end.i - start.i;

	std::vector<Result> result;
	result.reserve((count / bucket_count) + 1);

	Result bucket{start.f, end.f, 0.0f, 0.0f};
	uint32_t i = 0;

	for (Float_t t = start; t.f != end.f; t.i += 1)
	{
		if (i == bucket_count)
		{
			bucket.end = t.f;
			result.push_back(bucket);
			bucket = {t.f, end.f, 0.0f, 0.0f};
			i = 0;
		}

		float actual = actual_function(t.f);
		float approx = approx_function(t.f);
		float abs_error = absolute_error(actual, approx);
		float rel_error = relative_error(actual, approx);

		bucket.max_abs_error = max(abs_error, bucket.max_abs_error);
		bucket.max_rel_error = max(rel_error, bucket.max_rel_error);

		++i;
	}

	if (i != 0)
	{
		result.push_back(bucket);
	}

	return result;
}

std::vector<Result> evaluate(ScalarFunction actual_function, VectorFunction approx_function, Float_t start, Float_t end)
{
	const uint32_t count = end.i - start.i;

	std::vector<Result> result;
	result.reserve((count / bucket_count) + 1);

	Result bucket{start.f, end.f, 0.0f, 0.0f};
	uint32_t i = 0;

	Float_t t = start;
	while (t.f != end.f)
	{
		if (i == bucket_count)
		{
			bucket.end = t.f;
			result.push_back(bucket);
			bucket = {t.f, end.f, 0.0f, 0.0f};
			i = 0;
		}

		Vector_t t_v{t.i, t.i + 1, t.i + 2, t.i + 3};
		t.i += 4;

		Vector_t actual{
			actual_function(t_v.f[0]),
			actual_function(t_v.f[1]),
			actual_function(t_v.f[2]),
			actual_function(t_v.f[3])
		};

		Vector_t approx = approx_function(t_v.v);

		Vector_t abs_error = absolute_error(actual.v, approx.v);
		//Vector_t rel_error = relative_error(actual.v, approx.v);
		Vector_t rel_error{
			relative_error(actual.f[0], approx.f[0]),
			relative_error(actual.f[1], approx.f[1]),
			relative_error(actual.f[2], approx.f[2]),
			relative_error(actual.f[3], approx.f[3])
		};

		bucket.max_abs_error = max(bucket.max_abs_error, abs_error.f[0]);
		bucket.max_rel_error = max(bucket.max_rel_error, rel_error.f[0]);
		if (t_v.f[1] == end.f) break;
		bucket.max_abs_error = max(bucket.max_abs_error, abs_error.f[1]);
		bucket.max_rel_error = max(bucket.max_rel_error, rel_error.f[1]);
		if (t_v.f[2] == end.f) break;
		bucket.max_abs_error = max(bucket.max_abs_error, abs_error.f[2]);
		bucket.max_rel_error = max(bucket.max_rel_error, rel_error.f[2]);
		if (t_v.f[3] == end.f) break;
		bucket.max_abs_error = max(bucket.max_abs_error, abs_error.f[3]);
		bucket.max_rel_error = max(bucket.max_rel_error, rel_error.f[3]);

		i += 4;
	}

	if (i != 0)
	{
		result.push_back(bucket);
	}

	return result;	
}

std::vector<Result> evaluate(VectorFunction actual_function, VectorFunction approx_function, Float_t start, Float_t end)
{
	const uint32_t count = end.i - start.i;

	std::vector<Result> result;
	result.reserve((count / bucket_count) + 1);

	Result bucket{start.f, end.f, 0.0f, 0.0f};
	uint32_t i = 0;

	Float_t t = start;
	while (t.f != end.f)
	{
		if (i == bucket_count)
		{
			bucket.end = t.f;
			result.push_back(bucket);
			bucket = {t.f, end.f, 0.0f, 0.0f};
			i = 0;
		}

		Vector_t t_v{t.i, t.i + 1, t.i + 2, t.i + 3};
		t.i += 4;

		Vector_t actual = actual_function(t_v.v);
		Vector_t approx = approx_function(t_v.v);
		Vector_t abs_error = absolute_error(actual.v, approx.v);
		Vector_t rel_error = relative_error(actual.v, approx.v);

		bucket.max_abs_error = max(bucket.max_abs_error, abs_error.f[0]);
		bucket.max_rel_error = max(bucket.max_rel_error, rel_error.f[0]);
		if (t_v.f[1] == end.f) break;
		bucket.max_abs_error = max(bucket.max_abs_error, abs_error.f[1]);
		bucket.max_rel_error = max(bucket.max_rel_error, rel_error.f[1]);
		if (t_v.f[2] == end.f) break;
		bucket.max_abs_error = max(bucket.max_abs_error, abs_error.f[2]);
		bucket.max_rel_error = max(bucket.max_rel_error, rel_error.f[2]);
		if (t_v.f[3] == end.f) break;
		bucket.max_abs_error = max(bucket.max_abs_error, abs_error.f[3]);
		bucket.max_rel_error = max(bucket.max_rel_error, rel_error.f[3]);

		i += 4;
	}

	if (i != 0)
	{
		result.push_back(bucket);
	}

	return result;
}



float std_sin(float f);
__m128 directx_sin(__m128 v);


int main(int argc, char* argv[])
{
	std::vector<std::future<std::vector<Result>>> futures;

	Float_t start = 0.0f;
	while (start.f < float_pi)
	{
		Float_t end = start.i + (1 << 23);
		if (end.f > float_pi)
		{
			end.f = float_pi;
		}

		futures.push_back(std::async(std::launch::async, [start, end](){ return evaluate(std_sin, directx_sin, start, end); } ));

		start = end;
	}

	FILE* output = fopen("sin.txt", "w");
	for (auto& future : futures)
	{
		auto results = future.get();
		for (const auto& result : results)
		{
			fprintf(output, "%1.8e\t%1.8e\t%1.8e\t%1.8e\t\t%1.8e\t%1.8e\n", result.start, result.end, result.max_abs_error, result.max_rel_error, std_sin(result.start), std_sin(result.end));
		}
	}
	fclose(output);

	return 0;
}


float std_sin(float f)
{
	return std::sinf(f);
}


constexpr Vector_t directx_sine_constants_0 = { -0.16666667f, +0.0083333310f, -0.00019840874f, +2.7525562e-06f };
constexpr Vector_t directx_sine_constants_1 = { -2.3889859e-08f, 0.0f, 0.0f, 0.0f };

__m128 directx_sin(__m128 v)
{
	// Assume all input is in range -pi to pi
	// Convert to [-pi, +pi]
	__m128 t = _mm_mul_ps(v, vector_inv_two_pi);
	t = _mm_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	t = _mm_mul_ps(t, vector_two_pi);
	__m128 x = _mm_sub_ps(v, t);

	// convert to [-pi/2, pi/2]
	__m128 sign = _mm_and_ps(x, vector_negative_zero);
	__m128 c = _mm_or_ps(vector_pi, sign); // +pi or -pi
	__m128 absx = _mm_andnot_ps(sign, x); // absolute v
	__m128 rflx = _mm_sub_ps(c, x);
	__m128 comp = _mm_cmple_ps(absx, vector_half_pi);
	__m128 t0 = _mm_and_ps(comp, x);
	__m128 t1 = _mm_andnot_ps(comp, rflx);
	x = _mm_or_ps(t0, t1);

	__m128 x2 = _mm_mul_ps(x, x);

	const __m128 sine_constants_1 = directx_sine_constants_1;
	__m128 constants = _mm_permute_ps(sine_constants_1, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 result = _mm_mul_ps(constants, x2);

	const __m128 sine_constants_0 = directx_sine_constants_0;

	constants = _mm_permute_ps(sine_constants_0, _MM_SHUFFLE(3, 3, 3, 3));
	result = _mm_add_ps(result, constants);
	result = _mm_mul_ps(result, x2);

	constants = _mm_permute_ps(sine_constants_0, _MM_SHUFFLE(2, 2, 2, 2));
	result = _mm_add_ps(result, constants);
	result = _mm_mul_ps(result, x2);

	constants = _mm_permute_ps(sine_constants_0, _MM_SHUFFLE(1, 1, 1, 1));
	result = _mm_add_ps(result, constants);
	result = _mm_mul_ps(result, x2);

	constants = _mm_permute_ps(sine_constants_0, _MM_SHUFFLE(0, 0, 0, 0));
	result = _mm_add_ps(result, constants);
	result = _mm_mul_ps(result, x2);

	result = _mm_add_ps(result, vector_one);
	result = _mm_mul_ps(result, x);

	return result;
}
