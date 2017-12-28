#include <stdio.h>

#include <cstdint>
#include <type_traits>

#include <cfloat>
#include <cmath>

// Undefine things defined in math.h via cmath
#undef DOMAIN
#undef SING
#undef OVERFLOW
#undef UNDERFLOW
#undef TLOSS
#undef PLOSS

#include <limits>



// SSE2 intrinsics
#include <xmmintrin.h>
#include <emmintrin.h>
// SSE3 intrinsics
#include <pmmintrin.h>
// SSE4 intrinsics
#include <smmintrin.h>
// AVX intrinsics
#include <immintrin.h>


constexpr float float_pi = 3.1415927410125732421875f;
constexpr float float_half_pi =  1.57079637050628662109375f;


struct alignas(16) Vector_F32
{
	union
	{
		float f[4];
		__m128 v;
	};

	inline operator __m128() const { return v; }
	inline operator const float*() const { return f; }
	inline operator __m128i() const { return _mm_castps_si128(v); }
	inline operator __m128d() const { return _mm_castps_pd(v); }
};

struct alignas(16) Vector_U32
{
	union
	{
		uint32_t i[4];
		__m128 v;
	};

	inline operator __m128() const { return v; }
	inline operator __m128i() const { return _mm_castps_si128(v); }
	inline operator __m128d() const { return _mm_castps_pd(v); }
};

const Vector_F32 vector_one = { 1.0f, 1.0f, 1.0f, 1.0f };
const Vector_U32 vector_negative_zero = { 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
const Vector_F32 vector_pi = { float_pi, float_pi, float_pi, float_pi };
const Vector_F32 vector_half_pi = { float_half_pi, float_half_pi, float_half_pi, float_half_pi };
const Vector_F32 vector_sine_constants_0 = { -0.16666667f, +0.0083333310f, -0.00019840874f, +2.7525562e-06f };
const Vector_F32 vector_sine_constants_1 = { -2.3889859e-08f, 0.0f, 0.0f, 0.0f };
const Vector_F32 vector_cosine_constants_0 = { -0.5f, +0.041666638f, -0.0013888378f, +2.4760495e-05f };
const Vector_F32 vector_cosine_constants_1 = { -2.6051615e-07f, 0.0f, 0.0f, 0.0f };


float directx_sin(float f)
{
	__m128 x = _mm_set_ps1(f);

	// Assume all input is in range -pi to pi
	// Convert to [-pi, +pi]
	// __m128 t = _mm_mul_ps(v, vector_inv_two_pi);
	// t = _mm_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	// t = _mm_mul_ps(t, vector_two_pi);
	// __m128 x = _mm_sub_ps(v, t);

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

	__m128 constants = _mm_permute_ps(vector_sine_constants_1, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 result = _mm_mul_ps(constants, x2);

	const __m128 sine_constants = vector_sine_constants_0;

	constants = _mm_permute_ps(sine_constants, _MM_SHUFFLE(3, 3, 3, 3));
	result = _mm_add_ps(result, constants);
	result = _mm_mul_ps(result, x2);

	constants = _mm_permute_ps(sine_constants, _MM_SHUFFLE(2, 2, 2, 2));
	result = _mm_add_ps(result, constants);
	result = _mm_mul_ps(result, x2);

	constants = _mm_permute_ps(sine_constants, _MM_SHUFFLE(1, 1, 1, 1));
	result = _mm_add_ps(result, constants);
	result = _mm_mul_ps(result, x2);

	constants = _mm_permute_ps(sine_constants, _MM_SHUFFLE(0, 0, 0, 0));
	result = _mm_add_ps(result, constants);
	result = _mm_mul_ps(result, x2);

	result = _mm_add_ps(result, vector_one);
	result = _mm_mul_ps(result, x);

	return _mm_cvtss_f32(result);
}


float std_sin(float f)
{
	return std::sinf(f);
}


float calc_relative_error(float result, float actual)
{
	if (actual == 0.0f)
	{
		return result;
	}
	return 1.0f - (result / actual);
}

float calc_absolute_error(float result, float actual)
{
	return std::abs(actual - result);
}

union Float
{
	float f;
	uint32_t i;
};

using math_function = float (float);

void evaluate_over_pi_range(math_function actual_function, math_function approx_function)
{
	Float t;
	for (t.f = -float_pi; t.f != -0.0f; t.i -= 1)
	{
		float actual = actual_function(t.f);
		float approx = approx_function(t.f);
		float error = calc_absolute_error(actual, approx);
		float rel_error = calc_relative_error(actual, approx);

		printf("%.9g, %.9g, %.9g, %.9g, %.9g\n", t.f, actual, approx, error, rel_error );
	}
	for (t.f = 0.0f; t.f != float_pi; t.i += 1)
	{
		float actual = actual_function(t.f);
		float approx = approx_function(t.f);
		float error = calc_absolute_error(actual, approx);
		float rel_error = calc_relative_error(actual, approx);

		printf("%.9g, %.9g, %.9g, %.9g, %.9g\n", t.f, actual, approx, error, rel_error );
	}
}

int main(int argc, char* argv[])
{
	printf("t, sin, dx_sin, error, rel error\n");

	evaluate_over_pi_range(std_sin, directx_sin);

	return 0;
}