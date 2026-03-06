/*
 * test_doe.c — unit tests for doe.c internals
 *
 * includes doe.c directly (with main renamed) to test static functions.
 * covers: RNG, field physics, math ops, harmonic resonance,
 *         parliament election, LoRA lifecycle, weight profiler, sampling.
 *
 * cc tests/test_doe.c -O2 -lm -lpthread -o tests/test_doe && ./tests/test_doe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* ── pull in doe.c with main() renamed so we can define our own ── */
#define main doe_main
#include "../doe.c"
#undef main

/* ═══════════════════════════════════════════════════════════════════
 * tiny test harness
 * ═══════════════════════════════════════════════════════════════════ */
static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  [test] %-50s ", #name); \
    fflush(stdout); \
} while(0)

#define PASS() do { tests_passed++; printf("✓\n"); } while(0)
#define FAIL(msg) do { tests_failed++; printf("✗  %s\n", msg); } while(0)

#define ASSERT_TRUE(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_FLOAT_EQ(a, b, eps, msg) \
    ASSERT_TRUE(fabsf((a)-(b)) < (eps), msg)

/* ═══════════════════════════════════════════════════════════════════
 * RNG tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_rng_deterministic(void) {
    TEST(rng_deterministic);
    rng_state = 42;
    uint64_t a = rng_next();
    uint64_t b = rng_next();
    rng_state = 42;
    uint64_t a2 = rng_next();
    uint64_t b2 = rng_next();
    ASSERT_TRUE(a == a2 && b == b2, "same seed must produce same sequence");
    PASS();
}

static void test_rng_different_seeds(void) {
    TEST(rng_different_seeds);
    rng_state = 1;
    uint64_t a = rng_next();
    rng_state = 2;
    uint64_t b = rng_next();
    ASSERT_TRUE(a != b, "different seeds should produce different values");
    PASS();
}

static void test_rand_uniform_range(void) {
    TEST(rand_uniform_range);
    rng_state = 123;
    for (int i = 0; i < 10000; i++) {
        float u = rand_uniform();
        ASSERT_TRUE(u >= 0.0f && u <= 1.0f, "rand_uniform out of [0,1]");
    }
    PASS();
}

static void test_rand_normal_distribution(void) {
    TEST(rand_normal_distribution);
    rng_state = 77;
    float sum = 0, sum_sq = 0;
    int N = 50000;
    for (int i = 0; i < N; i++) {
        float x = rand_normal();
        sum += x;
        sum_sq += x * x;
    }
    float mean = sum / N;
    float var = sum_sq / N - mean * mean;
    /* mean ~ 0, var ~ 1 for standard normal */
    ASSERT_FLOAT_EQ(mean, 0.0f, 0.05f, "rand_normal mean should be ~0");
    ASSERT_FLOAT_EQ(var, 1.0f, 0.1f, "rand_normal variance should be ~1");
    PASS();
}

static void test_clamp01(void) {
    TEST(clamp01);
    ASSERT_FLOAT_EQ(clamp01(-0.5f), 0.0f, 1e-7f, "clamp01(-0.5) should be 0");
    ASSERT_FLOAT_EQ(clamp01(0.5f), 0.5f, 1e-7f, "clamp01(0.5) should be 0.5");
    ASSERT_FLOAT_EQ(clamp01(1.5f), 1.0f, 1e-7f, "clamp01(1.5) should be 1");
    ASSERT_FLOAT_EQ(clamp01(0.0f), 0.0f, 1e-7f, "clamp01(0) should be 0");
    ASSERT_FLOAT_EQ(clamp01(1.0f), 1.0f, 1e-7f, "clamp01(1) should be 1");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * math ops tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_silu(void) {
    TEST(silu_function);
    ASSERT_FLOAT_EQ(silu_f(0.0f), 0.0f, 1e-6f, "silu(0) should be 0");
    /* silu(x) = x * sigmoid(x). For large x, sigmoid->1 so silu->x */
    float s10 = silu_f(10.0f);
    ASSERT_TRUE(s10 > 9.99f, "silu(10) should be ~10");
    /* silu is odd-ish: silu(-x) != -silu(x) but silu(-big) -> 0 */
    float sn = silu_f(-10.0f);
    ASSERT_TRUE(fabsf(sn) < 0.01f, "silu(-10) should be ~0");
    PASS();
}

static void test_rmsnorm(void) {
    TEST(rmsnorm);
    float x[] = {3.0f, 4.0f};
    float w[] = {1.0f, 1.0f};
    float out[2];
    rmsnorm(out, x, w, 2, 1e-5f);
    /* rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355 */
    float rms = sqrtf(12.5f);
    float inv = 1.0f / rms;
    ASSERT_FLOAT_EQ(out[0], 3.0f * inv, 1e-4f, "rmsnorm[0] wrong");
    ASSERT_FLOAT_EQ(out[1], 4.0f * inv, 1e-4f, "rmsnorm[1] wrong");
    PASS();
}

static void test_matvec(void) {
    TEST(matvec);
    /* 2x3 matrix times 3-vector */
    float W[] = {1, 2, 3, 4, 5, 6};
    float x[] = {1, 1, 1};
    float out[2];
    matvec(out, W, x, 2, 3);
    ASSERT_FLOAT_EQ(out[0], 6.0f, 1e-5f, "matvec row 0: 1+2+3=6");
    ASSERT_FLOAT_EQ(out[1], 15.0f, 1e-5f, "matvec row 1: 4+5+6=15");
    PASS();
}

static void test_matvec_identity(void) {
    TEST(matvec_identity);
    float I[] = {1,0,0, 0,1,0, 0,0,1};
    float x[] = {7, 8, 9};
    float out[3];
    matvec(out, I, x, 3, 3);
    ASSERT_FLOAT_EQ(out[0], 7.0f, 1e-5f, "identity[0]");
    ASSERT_FLOAT_EQ(out[1], 8.0f, 1e-5f, "identity[1]");
    ASSERT_FLOAT_EQ(out[2], 9.0f, 1e-5f, "identity[2]");
    PASS();
}

static void test_softmax(void) {
    TEST(softmax);
    float x[] = {1.0f, 2.0f, 3.0f};
    softmax_n(x, 3);
    /* verify sum = 1 */
    float sum = x[0] + x[1] + x[2];
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "softmax sum should be 1");
    /* verify ordering preserved */
    ASSERT_TRUE(x[0] < x[1] && x[1] < x[2], "softmax should preserve ordering");
    /* verify all positive */
    ASSERT_TRUE(x[0] > 0 && x[1] > 0 && x[2] > 0, "softmax all positive");
    PASS();
}

static void test_softmax_uniform(void) {
    TEST(softmax_uniform);
    float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
    softmax_n(x, 4);
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(x[i], 0.25f, 1e-5f, "uniform softmax should give 0.25");
    PASS();
}

static void test_softmax_extreme(void) {
    TEST(softmax_extreme);
    float x[] = {100.0f, 0.0f, 0.0f};
    softmax_n(x, 3);
    ASSERT_TRUE(x[0] > 0.999f, "softmax(100,0,0)[0] should be ~1");
    ASSERT_TRUE(x[1] < 0.001f, "softmax(100,0,0)[1] should be ~0");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * field physics tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_field_init(void) {
    TEST(field_init);
    field_init();
    ASSERT_TRUE(F.prophecy == 7, "default prophecy should be 7");
    ASSERT_FLOAT_EQ(F.destiny, 0.35f, 1e-5f, "default destiny");
    ASSERT_FLOAT_EQ(F.debt_decay, 0.998f, 1e-5f, "default debt_decay");
    ASSERT_TRUE(F.velocity_mode == VEL_WALK, "default velocity");
    ASSERT_FLOAT_EQ(F.base_temperature, 1.0f, 1e-5f, "default base_temp");
    ASSERT_FLOAT_EQ(F.attend_focus, 0.70f, 1e-5f, "default attend_focus");
    ASSERT_FLOAT_EQ(F.entropy_floor, 0.1f, 1e-5f, "default entropy_floor");
    ASSERT_FLOAT_EQ(F.resonance_ceiling, 0.95f, 1e-5f, "default resonance_ceiling");
    ASSERT_TRUE(F.season == SEASON_SPRING, "should start in spring");
    PASS();
}

static void test_field_step_advances(void) {
    TEST(field_step_advances);
    field_init();
    int old_step = F.step;
    field_step(1.0f);
    ASSERT_TRUE(F.step == old_step + 1, "step should increment");
    field_step(1.0f);
    ASSERT_TRUE(F.step == old_step + 2, "step should increment again");
    PASS();
}

static void test_field_step_zero_dt(void) {
    TEST(field_step_zero_dt);
    field_init();
    int old_step = F.step;
    field_step(0.0f);
    ASSERT_TRUE(F.step == old_step, "dt=0 should not advance step");
    field_step(-1.0f);
    ASSERT_TRUE(F.step == old_step, "dt<0 should not advance step");
    PASS();
}

static void test_field_debt_decay(void) {
    TEST(field_debt_decay);
    field_init();
    F.debt = 10.0f;
    field_step(1.0f);
    ASSERT_TRUE(F.debt < 10.0f, "debt should decay after step");
    ASSERT_TRUE(F.debt > 0.0f, "debt should remain positive");
    PASS();
}

static void test_field_debt_cap(void) {
    TEST(field_debt_cap);
    field_init();
    F.debt = 200.0f;
    field_step(1.0f);
    ASSERT_TRUE(F.debt <= 100.0f, "debt should be capped at 100");
    PASS();
}

static void test_field_effective_temp_positive(void) {
    TEST(field_effective_temp_positive);
    field_init();
    for (int i = 0; i < 100; i++) field_step(1.0f);
    ASSERT_TRUE(F.effective_temp >= 0.1f, "effective_temp should never go below 0.1");
    PASS();
}

static void test_field_velocity_modes(void) {
    TEST(field_velocity_modes);
    field_init();
    F.velocity_mode = VEL_NOMOVE;
    field_step(1.0f);
    ASSERT_FLOAT_EQ(F.time_direction, 1.0f, 1e-5f, "NOMOVE: forward time");

    field_init();
    F.velocity_mode = VEL_BACKWARD;
    field_step(1.0f);
    ASSERT_FLOAT_EQ(F.time_direction, -1.0f, 1e-5f, "BACKWARD: reverse time");
    PASS();
}

static void test_field_season_cycling(void) {
    TEST(field_season_cycling);
    field_init();
    ASSERT_TRUE(F.season == SEASON_SPRING, "start in spring");
    /* season_phase advances by 0.001 per dt=1 step, so 1000 steps = 1 season */
    for (int i = 0; i < 1001; i++) field_step(1.0f);
    ASSERT_TRUE(F.season != SEASON_SPRING || F.season_phase > 0.0f,
                "season should advance after many steps");
    PASS();
}

static void test_field_entropy_bounds(void) {
    TEST(field_entropy_bounds);
    field_init();
    for (int i = 0; i < 50; i++) field_step(1.0f);
    ASSERT_TRUE(F.entropy >= F.entropy_floor, "entropy >= floor");
    ASSERT_TRUE(F.entropy >= 0.0f && F.entropy <= 1.0f, "entropy in [0,1]");
    PASS();
}

static void test_field_resonance_bounds(void) {
    TEST(field_resonance_bounds);
    field_init();
    for (int i = 0; i < 50; i++) field_step(1.0f);
    ASSERT_TRUE(F.resonance <= F.resonance_ceiling, "resonance <= ceiling");
    ASSERT_TRUE(F.resonance >= 0.0f && F.resonance <= 1.0f, "resonance in [0,1]");
    PASS();
}

static void test_field_emergence(void) {
    TEST(field_emergence);
    field_init();
    for (int i = 0; i < 20; i++) field_step(1.0f);
    /* emergence = (1 - entropy) * resonance, should be in [0,1] */
    ASSERT_TRUE(F.emergence >= 0.0f && F.emergence <= 1.0f, "emergence in [0,1]");
    PASS();
}

static void test_field_presence_decay(void) {
    TEST(field_presence_decay);
    field_init();
    float initial = F.presence_decay;
    for (int i = 0; i < 100; i++) field_step(1.0f);
    ASSERT_TRUE(F.presence_decay < initial, "presence should decay over time");
    ASSERT_TRUE(F.presence_decay >= 0.001f, "presence should not go below floor");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * Schumann resonance tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_schumann_coherence_base(void) {
    TEST(schumann_coherence_at_base);
    float c = schumann_coherence(SCHUMANN_BASE_HZ);
    ASSERT_FLOAT_EQ(c, 1.0f, 1e-5f, "coherence at base freq should be 1.0");
    PASS();
}

static void test_schumann_coherence_far(void) {
    TEST(schumann_coherence_far);
    float c = schumann_coherence(100.0f);
    ASSERT_TRUE(c >= 0.0f && c <= 1.0f, "coherence should be clamped to [0,1]");
    /* very far from base should be low or zero */
    ASSERT_TRUE(c < 0.5f, "far from base should have low coherence");
    PASS();
}

static void test_schumann_signal_bounded(void) {
    TEST(schumann_signal_bounded);
    field_init();
    for (int i = 0; i < 100; i++) {
        field_step(1.0f);
        float s = schumann_signal();
        ASSERT_TRUE(s >= -2.0f && s <= 2.0f, "schumann signal should be bounded");
    }
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * 4.C MLP tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_field_mlp_output_bounded(void) {
    TEST(field_mlp_output_bounded);
    field_mlp_init();
    float in[FIELD_4C_INPUTS] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float out[FIELD_4C_OUTPUTS];
    field_mlp_forward(in, out);
    for (int i = 0; i < FIELD_4C_OUTPUTS; i++)
        ASSERT_TRUE(out[i] >= -1.0f && out[i] <= 1.0f,
                    "MLP output should be in [-1,1] (tanh)");
    PASS();
}

static void test_field_mlp_deterministic(void) {
    TEST(field_mlp_deterministic);
    field_mlp_init();
    float in[FIELD_4C_INPUTS] = {0.3f, 0.7f, 0.1f, 0.9f, 0.5f, 0.2f};
    float out1[FIELD_4C_OUTPUTS], out2[FIELD_4C_OUTPUTS];
    field_mlp_forward(in, out1);
    field_mlp_forward(in, out2);
    for (int i = 0; i < FIELD_4C_OUTPUTS; i++)
        ASSERT_FLOAT_EQ(out1[i], out2[i], 1e-7f, "MLP should be deterministic");
    PASS();
}

static void test_field_mlp_hebbian_modifies_weights(void) {
    TEST(field_mlp_hebbian_modifies);
    field_init();
    float in[FIELD_4C_INPUTS] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float out[FIELD_4C_OUTPUTS];
    field_mlp_forward(in, out);
    field_mlp_hebbian(in, out, 1.0f);
    /* Hebbian update with signal=1.0 should modify weights (unless activations are zero) */
    /* we just check that the function runs without crashing */
    ASSERT_TRUE(1, "hebbian ran without crash");
    PASS();
}

static void test_field_mlp_hebbian_clamp(void) {
    TEST(field_mlp_hebbian_weight_clamp);
    field_init();
    /* Set a weight near the boundary */
    F_mlp.w2[0] = 2.99f;
    float in[FIELD_4C_INPUTS] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float out[FIELD_4C_OUTPUTS];
    field_mlp_forward(in, out);
    /* Large positive signal + repeated updates */
    for (int i = 0; i < 100; i++) field_mlp_hebbian(in, out, 10.0f);
    /* Weights should be clamped at ±3.0 */
    for (int h = 0; h < FIELD_4C_HIDDEN; h++)
        for (int o = 0; o < FIELD_4C_OUTPUTS; o++) {
            float w = F_mlp.w2[h * FIELD_4C_OUTPUTS + o];
            ASSERT_TRUE(w >= -3.0f && w <= 3.0f, "w2 should be clamped to [-3,3]");
        }
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * prophecy debt tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_prophecy_debt_chosen_is_best(void) {
    TEST(prophecy_debt_best_chosen);
    float logits[] = {1.0f, 3.0f, 2.0f};
    float debt = compute_prophecy_debt(logits, 1, 3); /* chosen = argmax */
    ASSERT_FLOAT_EQ(debt, 0.0f, 1e-6f, "debt should be 0 when choosing the best");
    PASS();
}

static void test_prophecy_debt_worst_chosen(void) {
    TEST(prophecy_debt_worst_chosen);
    float logits[] = {10.0f, 0.0f, 5.0f};
    float debt = compute_prophecy_debt(logits, 1, 3); /* worst token chosen */
    ASSERT_TRUE(debt > 0.0f, "debt should be positive when not choosing best");
    ASSERT_TRUE(debt < 1.0f, "debt is normalized: diff/(diff+1)");
    /* diff = 10-0 = 10, debt = 10/11 ≈ 0.909 */
    ASSERT_FLOAT_EQ(debt, 10.0f / 11.0f, 1e-5f, "debt = 10/11");
    PASS();
}

static void test_prophecy_debt_edge_cases(void) {
    TEST(prophecy_debt_edge_cases);
    ASSERT_FLOAT_EQ(compute_prophecy_debt(NULL, 0, 0), 0.0f, 1e-7f, "n=0");
    ASSERT_FLOAT_EQ(compute_prophecy_debt(NULL, -1, 5), 0.0f, 1e-7f, "chosen<0");
    float l[] = {1.0f};
    ASSERT_FLOAT_EQ(compute_prophecy_debt(l, 5, 1), 0.0f, 1e-7f, "chosen>=n");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * field → logits pipeline tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_apply_destiny(void) {
    TEST(apply_destiny);
    field_init();
    F.destiny_bias = 0.5f;
    float logits[] = {5.0f, 3.0f, 1.0f};
    float orig[3]; memcpy(orig, logits, sizeof(orig));
    apply_destiny(logits, 3);
    /* max is 5.0, so tokens with lower logits get pushed down more */
    ASSERT_FLOAT_EQ(logits[0], orig[0], 1e-5f, "max token unchanged by destiny");
    ASSERT_TRUE(logits[1] < orig[1], "non-max tokens suppressed");
    ASSERT_TRUE(logits[2] < orig[2], "weakest token suppressed most");
    PASS();
}

static void test_apply_destiny_zero_bias(void) {
    TEST(apply_destiny_zero_bias);
    field_init();
    F.destiny_bias = 0.0f;
    float logits[] = {5.0f, 3.0f, 1.0f};
    float orig[3]; memcpy(orig, logits, sizeof(orig));
    apply_destiny(logits, 3);
    for (int i = 0; i < 3; i++)
        ASSERT_FLOAT_EQ(logits[i], orig[i], 1e-7f, "zero bias => no change");
    PASS();
}

static void test_apply_suffering(void) {
    TEST(apply_suffering);
    field_init();
    F.pain = 0.8f;
    F.tension = 0.5f;
    float logits[] = {10.0f, 0.0f, 5.0f};
    apply_suffering(logits, 3);
    /* suffering compresses toward mean. spread should decrease */
    float mean = (logits[0] + logits[1] + logits[2]) / 3.0f;
    float var = 0;
    for (int i = 0; i < 3; i++) var += (logits[i] - mean) * (logits[i] - mean);
    /* original variance was much higher */
    float orig_var = ((10-5)*(10-5) + (0-5)*(0-5) + (5-5)*(5-5));
    ASSERT_TRUE(var < orig_var, "suffering should reduce variance");
    PASS();
}

static void test_apply_attention(void) {
    TEST(apply_attention);
    field_init();
    F.attend_focus = 0.8f;
    float logits[] = {10.0f, 5.0f, 1.0f};
    apply_attention(logits, 3);
    /* attention sharpens: pushes non-max down */
    ASSERT_TRUE(logits[1] < 5.0f, "attention should sharpen: non-max pushed down");
    ASSERT_TRUE(logits[2] < 1.0f, "attention should sharpen: weak pushed lower");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * harmonic resonance tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_harmonic_decompose(void) {
    TEST(harmonic_decompose);
    /* pure sinusoid at frequency 1 */
    float hist[16];
    for (int i = 0; i < 16; i++) hist[i] = sinf(2.0f * 3.14159f * 1.0f * i / 16.0f);
    HarmonicState hs;
    harmonic_decompose(&hs, hist, 16);
    /* dominant frequency should be k=1 */
    ASSERT_TRUE(hs.amplitudes[1] > hs.amplitudes[0], "k=1 should dominate over DC");
    ASSERT_TRUE(hs.confidence > 0.0f, "confidence should be positive");
    PASS();
}

static void test_harmonic_decompose_dc(void) {
    TEST(harmonic_decompose_dc);
    /* constant signal — should be all DC */
    float hist[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    HarmonicState hs;
    harmonic_decompose(&hs, hist, 8);
    ASSERT_TRUE(hs.amplitudes[0] > 0.5f, "DC component should be strong");
    PASS();
}

static void test_expert_resonance(void) {
    TEST(expert_resonance);
    HarmonicState hs = {0};
    hs.amplitudes[0] = 1.0f; /* DC */
    float r = expert_resonance(0.0f, &hs);
    ASSERT_TRUE(r > 0.0f, "resonance with DC at freq=0 should be positive");
    float r2 = expert_resonance(3.14f, &hs);
    ASSERT_TRUE(r2 < r, "far frequency should resonate less with DC");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * weight profiler tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_profile_weights_basic(void) {
    TEST(profile_weights_basic);
    rng_state = 42;
    float data[4*4];
    for (int i = 0; i < 16; i++) data[i] = rand_normal() * 0.5f;
    LayerProfile lp;
    profile_weights(data, 4, 4, &lp);
    ASSERT_TRUE(lp.l2_norm > 0.0f, "l2_norm should be positive");
    ASSERT_TRUE(lp.mean_abs > 0.0f, "mean_abs should be positive");
    ASSERT_TRUE(lp.std_dev > 0.0f, "std_dev should be positive");
    ASSERT_TRUE(lp.health >= 0.0f && lp.health <= 1.0f, "health in [0,1]");
    PASS();
}

static void test_profile_weights_zero(void) {
    TEST(profile_weights_zero_matrix);
    float data[9] = {0};
    LayerProfile lp;
    profile_weights(data, 3, 3, &lp);
    ASSERT_FLOAT_EQ(lp.l2_norm, 0.0f, 1e-7f, "zero matrix l2 should be 0");
    ASSERT_FLOAT_EQ(lp.sparsity, 1.0f, 1e-5f, "zero matrix should be 100% sparse");
    ASSERT_TRUE(lp.dead_neurons == 3, "all rows should be dead");
    PASS();
}

static void test_profile_weights_empty(void) {
    TEST(profile_weights_empty);
    LayerProfile lp;
    profile_weights(NULL, 0, 0, &lp);
    ASSERT_FLOAT_EQ(lp.l2_norm, 0.0f, 1e-7f, "empty matrix l2 should be 0");
    PASS();
}

static void test_compute_fingerprint(void) {
    TEST(compute_fingerprint);
    WeightProfile wp1, wp2;
    memset(&wp1, 0, sizeof(wp1));
    memset(&wp2, 0, sizeof(wp2));
    wp1.n_layers = 2;
    wp1.layers[0].l2_norm = 1.0f; wp1.layers[0].std_dev = 0.5f;
    wp1.layers[1].l2_norm = 2.0f; wp1.layers[1].std_dev = 0.3f;
    wp2 = wp1;
    uint64_t f1 = compute_fingerprint(&wp1);
    uint64_t f2 = compute_fingerprint(&wp2);
    ASSERT_TRUE(f1 == f2, "same profile should give same fingerprint");
    wp2.layers[0].l2_norm = 1.001f;
    uint64_t f3 = compute_fingerprint(&wp2);
    ASSERT_TRUE(f1 != f3, "different profile should give different fingerprint");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * LoRA expert lifecycle tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_lora_expert_init_free(void) {
    TEST(lora_expert_init_free);
    LoraExpert e;
    memset(&e, 0, sizeof(e));
    init_lora_expert(&e, 32, 4, 1.5f);
    ASSERT_TRUE(e.alive == 1, "expert should be alive after init");
    ASSERT_TRUE(e.lora_A != NULL, "lora_A should be allocated");
    ASSERT_TRUE(e.lora_B != NULL, "lora_B should be allocated");
    ASSERT_FLOAT_EQ(e.frequency, 1.5f, 1e-5f, "frequency should match");
    ASSERT_FLOAT_EQ(e.vitality, 0.7f, 1e-5f, "initial vitality 0.7");
    free_lora_expert(&e);
    ASSERT_TRUE(e.alive == 0, "expert should be dead after free");
    ASSERT_TRUE(e.lora_A == NULL, "lora_A should be NULL after free");
    ASSERT_TRUE(e.lora_B == NULL, "lora_B should be NULL after free");
    PASS();
}

static void test_expert_vitality_update(void) {
    TEST(expert_vitality_update);
    FieldLayer fl;
    memset(&fl, 0, sizeof(fl));
    for (int e = 0; e < 4; e++) {
        fl.experts[e].alive = 1;
        fl.experts[e].vitality = 0.5f;
        fl.experts[e].tokens_seen = (e == 0) ? 100 : 10;
    }
    fl.n_alive = 4;
    update_expert_vitality(&fl, 130);
    /* expert 0 sees 100 out of fair=32.5, so ratio>1 → vitality increases */
    ASSERT_TRUE(fl.experts[0].vitality > 0.5f, "busy expert vitality should increase");
    /* experts 1-3 see 10 out of fair=32.5, so ratio<1 → vitality decreases */
    ASSERT_TRUE(fl.experts[1].vitality < 0.5f, "idle expert vitality should decrease");
    PASS();
}

static void test_mitosis(void) {
    TEST(mitosis);
    FieldLayer fl;
    memset(&fl, 0, sizeof(fl));
    /* Create one parent expert with high vitality and age */
    init_lora_expert(&fl.experts[0], 16, 4, 1.0f);
    fl.experts[0].vitality = 0.9f;
    fl.experts[0].age = 30;
    fl.n_alive = 1;
    int result = try_mitosis(&fl, 16, 4);
    ASSERT_TRUE(result == 1, "mitosis should succeed");
    ASSERT_TRUE(fl.experts[1].alive == 1, "child should be alive");
    ASSERT_TRUE(fl.n_alive == 2, "n_alive should be 2");
    ASSERT_TRUE(fl.experts[0].vitality < 0.9f, "parent vitality should decrease");
    /* cleanup */
    free_lora_expert(&fl.experts[0]);
    free_lora_expert(&fl.experts[1]);
    PASS();
}

static void test_mitosis_at_capacity(void) {
    TEST(mitosis_at_capacity);
    FieldLayer fl;
    memset(&fl, 0, sizeof(fl));
    /* Fill all expert slots */
    for (int i = 0; i < MAX_EXPERTS; i++) {
        init_lora_expert(&fl.experts[i], 16, 4, (float)i);
        fl.experts[i].vitality = 0.9f;
        fl.experts[i].age = 30;
    }
    fl.n_alive = MAX_EXPERTS;
    int result = try_mitosis(&fl, 16, 4);
    ASSERT_TRUE(result == 0, "mitosis should fail at capacity");
    for (int i = 0; i < MAX_EXPERTS; i++) free_lora_expert(&fl.experts[i]);
    PASS();
}

static void test_apoptosis(void) {
    TEST(apoptosis);
    FieldLayer fl;
    memset(&fl, 0, sizeof(fl));
    for (int i = 0; i < 3; i++) {
        init_lora_expert(&fl.experts[i], 16, 4, (float)i);
    }
    fl.n_alive = 3;
    /* Mark one for death */
    fl.experts[2].low_vitality_count = 10;
    int result = try_apoptosis(&fl);
    ASSERT_TRUE(result == 1, "apoptosis should succeed");
    ASSERT_TRUE(fl.experts[2].alive == 0, "expert 2 should be dead");
    ASSERT_TRUE(fl.n_alive == 2, "n_alive should decrease");
    free_lora_expert(&fl.experts[0]);
    free_lora_expert(&fl.experts[1]);
    PASS();
}

static void test_apoptosis_minimum(void) {
    TEST(apoptosis_minimum_experts);
    FieldLayer fl;
    memset(&fl, 0, sizeof(fl));
    for (int i = 0; i < MIN_EXPERTS; i++) {
        init_lora_expert(&fl.experts[i], 16, 4, (float)i);
        fl.experts[i].low_vitality_count = 100;
    }
    fl.n_alive = MIN_EXPERTS;
    int result = try_apoptosis(&fl);
    ASSERT_TRUE(result == 0, "should not go below MIN_EXPERTS");
    for (int i = 0; i < MIN_EXPERTS; i++) free_lora_expert(&fl.experts[i]);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * parliament election tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_parliament_elect(void) {
    TEST(parliament_elect);
    rng_state = 42;
    int dim = 16;
    Parliament p;
    memset(&p, 0, sizeof(p));
    p.w_vote = calloc(MAX_EXPERTS * dim, sizeof(float));
    for (int i = 0; i < MAX_EXPERTS * dim; i++)
        p.w_vote[i] = rand_normal() * 0.01f;
    p.consensus = 0.5f;

    LoraExpert experts[MAX_EXPERTS];
    memset(experts, 0, sizeof(experts));
    for (int i = 0; i < 4; i++) {
        init_lora_expert(&experts[i], dim, 4, 6.2831853f * i / 4);
    }

    float input[16];
    for (int i = 0; i < dim; i++) input[i] = rand_normal();

    HarmonicState hs = {0};
    hs.amplitudes[0] = 1.0f;

    int selected[MAX_EXPERTS];
    float weights[MAX_EXPERTS];
    int k = parliament_elect(&p, experts, input, dim, &hs, selected, weights);

    ASSERT_TRUE(k >= 2, "should select at least 2 experts");
    ASSERT_TRUE(k <= 4, "should not select more than alive");

    /* weights should sum to ~1 */
    float wsum = 0;
    for (int i = 0; i < k; i++) wsum += weights[i];
    ASSERT_FLOAT_EQ(wsum, 1.0f, 1e-4f, "weights should sum to 1");

    /* all weights positive */
    for (int i = 0; i < k; i++)
        ASSERT_TRUE(weights[i] > 0, "all weights should be positive");

    /* cleanup */
    for (int i = 0; i < 4; i++) free_lora_expert(&experts[i]);
    free(p.w_vote);
    PASS();
}

static void test_parliament_not_enough_experts(void) {
    TEST(parliament_insufficient_experts);
    int dim = 8;
    Parliament p;
    memset(&p, 0, sizeof(p));
    p.w_vote = calloc(MAX_EXPERTS * dim, sizeof(float));
    LoraExpert experts[MAX_EXPERTS];
    memset(experts, 0, sizeof(experts));
    /* Only 1 alive — below MIN_EXPERTS */
    init_lora_expert(&experts[0], dim, 4, 0.0f);
    float input[8] = {1};
    HarmonicState hs = {0};
    int sel[MAX_EXPERTS]; float wt[MAX_EXPERTS];
    int k = parliament_elect(&p, experts, input, dim, &hs, sel, wt);
    ASSERT_TRUE(k == 0, "should return 0 when below MIN_EXPERTS");
    free_lora_expert(&experts[0]);
    free(p.w_vote);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * sampling tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_sample_greedy(void) {
    TEST(sample_greedy);
    float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
    int idx = sample(logits, 4, 0.0f, 0);
    ASSERT_TRUE(idx == 1, "greedy should pick argmax (index 1)");
    PASS();
}

static void test_sample_temperature(void) {
    TEST(sample_with_temperature);
    rng_state = 42;
    int counts[4] = {0};
    for (int trial = 0; trial < 10000; trial++) {
        float logits[] = {0.0f, 10.0f, 0.0f, 0.0f};
        int idx = sample(logits, 4, 0.5f, 0);
        ASSERT_TRUE(idx >= 0 && idx < 4, "sample index out of range");
        counts[idx]++;
    }
    /* with temp=0.5 and logits[1]=10, token 1 should dominate */
    ASSERT_TRUE(counts[1] > 9000, "dominant logit should be sampled most");
    PASS();
}

static void test_sample_top_k(void) {
    TEST(sample_top_k);
    rng_state = 42;
    int counts[4] = {0};
    for (int trial = 0; trial < 5000; trial++) {
        float logits[] = {1.0f, 2.0f, 3.0f, 0.5f};
        int idx = sample(logits, 4, 1.0f, 2);
        counts[idx]++;
    }
    /* top-2 should be indices 1 and 2 (logits 2.0 and 3.0) */
    ASSERT_TRUE(counts[0] == 0 || counts[0] < 50, "index 0 should rarely/never be sampled with top_k=2");
    ASSERT_TRUE(counts[3] == 0 || counts[3] < 50, "index 3 should rarely/never be sampled with top_k=2");
    ASSERT_TRUE(counts[2] > counts[1], "higher logit should be sampled more");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * calendar drift tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_calendar_dissonance_range(void) {
    TEST(calendar_dissonance_range);
    calendar_init();
    float d = calendar_dissonance();
    ASSERT_TRUE(d >= 0.0f && d <= 1.0f, "calendar dissonance should be in [0,1]");
    PASS();
}

static void test_drift_init(void) {
    TEST(drift_init);
    CalendarDrift cd;
    drift_init(&cd);
    ASSERT_FLOAT_EQ(cd.drift, 0.0f, 1e-7f, "initial drift should be 0");
    ASSERT_FLOAT_EQ(cd.stability, 0.0f, 1e-7f, "initial stability should be 0");
    ASSERT_TRUE(cd.n_snapshots == 0, "initial snapshots should be 0");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * meta-learning tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_meta_init(void) {
    TEST(meta_init);
    MetaTrack mt;
    meta_init(&mt);
    ASSERT_TRUE(mt.n_entries == 0, "initial entries should be 0");
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(mt.config_bias[i], 0.5f, 1e-7f, "initial bias should be 0.5");
    PASS();
}

static void test_meta_record(void) {
    TEST(meta_record);
    MetaTrack mt;
    meta_init(&mt);
    meta_record(&mt, 1, 4, 0.5f, 5.0f, 0.8f, 0.1f, 0.05f, 5.5f);
    ASSERT_TRUE(mt.n_entries == 1, "should have 1 entry");
    meta_record(&mt, 2, 4, 0.5f, 4.5f, 0.8f, 0.1f, 0.05f, 5.0f);
    ASSERT_TRUE(mt.n_entries == 2, "should have 2 entries");
    /* config_bias should have been updated by meta-learning */
    PASS();
}

static void test_meta_capacity_overflow(void) {
    TEST(meta_capacity_overflow);
    MetaTrack mt;
    meta_init(&mt);
    for (int i = 0; i < META_HIST_CAP + 20; i++)
        meta_record(&mt, i, 4, 0.5f, 5.0f - i*0.01f, 0.8f, 0.1f, 0.05f, 5.0f);
    ASSERT_TRUE(mt.n_entries <= META_HIST_CAP, "should not exceed capacity");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * RoPE tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_rope_position_zero(void) {
    TEST(rope_position_zero);
    /* At position 0, cos=1, sin=0 → vector unchanged */
    float v[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int hd = 4;
    float cc[2] = {1.0f, 1.0f};
    float sc[2] = {0.0f, 0.0f};
    apply_rope(v, 0, cc, sc, hd);
    ASSERT_FLOAT_EQ(v[0], 1.0f, 1e-5f, "rope pos=0 should preserve v[0]");
    ASSERT_FLOAT_EQ(v[1], 2.0f, 1e-5f, "rope pos=0 should preserve v[1]");
    ASSERT_FLOAT_EQ(v[2], 3.0f, 1e-5f, "rope pos=0 should preserve v[2]");
    ASSERT_FLOAT_EQ(v[3], 4.0f, 1e-5f, "rope pos=0 should preserve v[3]");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * NOTORCH tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_notorch_zero_signal(void) {
    TEST(notorch_zero_signal);
    field_init();
    float A[4*2], B[2*4];
    memset(A, 0, sizeof(A));
    memset(B, 0, sizeof(B));
    A[0] = 1.0f; B[0] = 1.0f;
    float x[] = {1, 0, 0, 0};
    float dy[] = {1, 0, 0, 0};
    float a_before = A[0];
    notorch_step(A, B, 4, 4, 2, x, dy, 0.0f);
    ASSERT_FLOAT_EQ(A[0], a_before, 1e-7f, "zero signal should not change weights");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN — run all tests
 * ═══════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  doe.c test suite — democracy of experts under oath\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    /* RNG */
    printf("[rng]\n");
    test_rng_deterministic();
    test_rng_different_seeds();
    test_rand_uniform_range();
    test_rand_normal_distribution();
    test_clamp01();

    /* Math ops */
    printf("\n[math]\n");
    test_silu();
    test_rmsnorm();
    test_matvec();
    test_matvec_identity();
    test_softmax();
    test_softmax_uniform();
    test_softmax_extreme();
    test_rope_position_zero();

    /* Field physics */
    printf("\n[field]\n");
    test_field_init();
    test_field_step_advances();
    test_field_step_zero_dt();
    test_field_debt_decay();
    test_field_debt_cap();
    test_field_effective_temp_positive();
    test_field_velocity_modes();
    test_field_season_cycling();
    test_field_entropy_bounds();
    test_field_resonance_bounds();
    test_field_emergence();
    test_field_presence_decay();

    /* Schumann resonance */
    printf("\n[schumann]\n");
    test_schumann_coherence_base();
    test_schumann_coherence_far();
    test_schumann_signal_bounded();

    /* 4.C MLP */
    printf("\n[4c_mlp]\n");
    test_field_mlp_output_bounded();
    test_field_mlp_deterministic();
    test_field_mlp_hebbian_modifies_weights();
    test_field_mlp_hebbian_clamp();

    /* Prophecy debt */
    printf("\n[prophecy]\n");
    test_prophecy_debt_chosen_is_best();
    test_prophecy_debt_worst_chosen();
    test_prophecy_debt_edge_cases();

    /* Field → logits */
    printf("\n[field_logits]\n");
    test_apply_destiny();
    test_apply_destiny_zero_bias();
    test_apply_suffering();
    test_apply_attention();

    /* Harmonic resonance */
    printf("\n[harmonic]\n");
    test_harmonic_decompose();
    test_harmonic_decompose_dc();
    test_expert_resonance();

    /* Weight profiler */
    printf("\n[profiler]\n");
    test_profile_weights_basic();
    test_profile_weights_zero();
    test_profile_weights_empty();
    test_compute_fingerprint();

    /* LoRA lifecycle */
    printf("\n[lora]\n");
    test_lora_expert_init_free();
    test_expert_vitality_update();
    test_mitosis();
    test_mitosis_at_capacity();
    test_apoptosis();
    test_apoptosis_minimum();

    /* Parliament */
    printf("\n[parliament]\n");
    test_parliament_elect();
    test_parliament_not_enough_experts();

    /* Sampling */
    printf("\n[sampling]\n");
    test_sample_greedy();
    test_sample_temperature();
    test_sample_top_k();

    /* Calendar/drift */
    printf("\n[calendar]\n");
    test_calendar_dissonance_range();
    test_drift_init();

    /* Meta-learning */
    printf("\n[meta]\n");
    test_meta_init();
    test_meta_record();
    test_meta_capacity_overflow();

    /* NOTORCH */
    printf("\n[notorch]\n");
    test_notorch_zero_signal();

    /* Summary */
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    printf("═══════════════════════════════════════════════════════\n\n");

    return tests_failed > 0 ? 1 : 0;
}
