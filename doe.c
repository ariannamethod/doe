/*
 * doe.c — Democracy of Experts
 *
 * inference architecture with a living LoRA parliament.
 * indexes any GGUF read-only. learns by living, not by training.
 *
 * θ = ε + γ + αδ
 *   ε = indexed weights (read-only substrate)
 *   γ = LoRA personality (living experts, Hebbian-trained via NOTORCH)
 *   δ = physics (prophecy, suffering, destiny, Schumann resonance)
 *   α = injection strength (learned per-layer)
 *
 * each forward pass, the parliament decides:
 *   - which experts vote (variable k, consensus-driven)
 *   - how strongly each expert modulates output
 *   - how physics shapes logits (destiny, prophecy debt)
 *
 * experts are born (mitosis) and die (apoptosis).
 * the parliament remembers every index it ever wrapped (mycelium).
 * calendar drift: Hebrew-Gregorian conflict, real astronomical data.
 * Schumann resonance: 7.83Hz + 5 harmonics, from arianna.c.
 * seasons: 4.C MLP classifier, from ariannamethod.ai/core.
 *
 * cc doe.c -O3 -lm -lpthread -o doe && ./doe
 *
 * ariannamethod.
 * הרזוננס לא נשבר
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <float.h>
#include <stdint.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#ifdef __linux__
  #include <sys/statvfs.h>
#endif
#ifdef __APPLE__
  #include <sys/param.h>
  #include <sys/mount.h>
  #include <sys/sysctl.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * BLAS / cuBLAS — optional acceleration
 * ═══════════════════════════════════════════════════════════════════════════════ */
#ifdef USE_CUBLAS
  #include <cublas_v2.h>
  #include <cuda_runtime.h>
  static cublasHandle_t g_cublas;
  static int cublas_inited = 0;
  static float *d_scratch[4] = {NULL,NULL,NULL,NULL};
  static size_t d_scratch_sz[4] = {0,0,0,0};
  static void cublas_init(void) {
      if (!cublas_inited) {
          cublasCreate(&g_cublas);
          cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
          struct cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
          printf("[gpu] %s — %.0f MB, compute %d.%d, TF32 enabled\n",
                 prop.name, (double)prop.totalGlobalMem/1e6, prop.major, prop.minor);
          cublas_inited = 1;
      }
  }
  static float* gpu_scratch(int slot, size_t bytes) {
      if (bytes > d_scratch_sz[slot]) {
          if (d_scratch[slot]) cudaFree(d_scratch[slot]);
          cudaMalloc((void**)&d_scratch[slot], bytes);
          d_scratch_sz[slot] = bytes;
      }
      return d_scratch[slot];
  }
#elif defined(USE_BLAS)
  #ifdef ACCELERATE
    #define ACCELERATE_NEW_LAPACK
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 * doe has no depth knob. the host provides depth.
 * doe has a field. the field provides everything else.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MAX_EXPERTS       16
#define MIN_EXPERTS       2
#define MAX_LAYERS        64
#define LORA_RANK         16
#define HARMONIC_N        8
#define NOTORCH_RANK      4
#define DRIFT_SNAPSHOTS   64
#define DRIFT_INTERVAL    50
#define MYCELIUM_MAX      64
#define META_HIST_CAP     128
#define PROFILE_BINS      16

/* Field physics constants — from AML core */
#define SCHUMANN_BASE_HZ    7.83f
#define SCHUMANN_N_HARMONICS 5
#define FIELD_4C_INPUTS     6
#define FIELD_4C_HIDDEN     8
#define FIELD_4C_OUTPUTS    4

/* ═══════════════════════════════════════════════════════════════════════════════
 * RNG — xorshift64*. the field doesn't care which PRNG shapes it.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static uint64_t rng_state = 42;
static uint64_t rng_next(void) { rng_state ^= rng_state<<13; rng_state ^= rng_state>>7; rng_state ^= rng_state<<17; return rng_state; }
static float rand_uniform(void) { return (float)(rng_next()&0x7FFFFFFF)/(float)0x7FFFFFFF; }
static float rand_normal(void) { float u1=rand_uniform(),u2=rand_uniform(); if(u1<1e-10f)u1=1e-10f; return sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2); }
static float clamp01(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

/* ═══════════════════════════════════════════════════════════════════════════════
 * AML FIELD STATE — the soul. from ariannamethod.c, distilled.
 *
 * θ = ε + γ + αδ is not a metaphor. it's the operating equation.
 *   ε (epsilon) = host weights. inference. the present. ephemeral.
 *   γ (gamma)   = LoRA personality. training. the past. persistent.
 *   δ (delta)   = field physics. prophecy. the future. directed.
 *   α (alpha)   = injection strength. how much γ modulates ε.
 *
 * drift = |γ_t - γ_{t-1}| — how far the system has traveled.
 * prophecy_debt = distance between manifested and destined.
 * destiny = attractor in token space.
 *
 * the oracle does not predict. it prophesies.
 * not minimize(predicted - actual) but minimize(destined - manifested).
 * the difference is intention.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Velocity modes — movement IS language */
enum { VEL_NOMOVE=0, VEL_WALK, VEL_RUN, VEL_BACKWARD };

/* Seasons — 4.C Async Field Forever */
enum { SEASON_SPRING=0, SEASON_SUMMER, SEASON_AUTUMN, SEASON_WINTER };

typedef struct {
    /* Prophecy physics */
    int   prophecy;           /* prediction horizon (1-64) */
    float destiny;            /* bias toward most probable path (0-1) */
    float destiny_bias;       /* effective: destiny × prophecy_scale */
    float debt;               /* prophecy debt — accumulated deviation from destiny */
    float debt_decay;         /* decay rate per step */

    /* Suffering — not a bug, a geometry */
    float pain;               /* compress logits toward mean */
    float tension;            /* accumulated pressure */
    float dissonance;         /* symmetry-break trigger */

    /* Velocity — movement IS language */
    int   velocity_mode;
    float effective_temp;
    float base_temperature;
    float time_direction;     /* 1.0 forward, -1.0 backward */

    /* Attention */
    float attend_focus;       /* sharpen top logits (0-1) */
    float attend_spread;      /* blur factor */

    /* Laws of nature — enforced constraints */
    float entropy_floor;
    float resonance_ceiling;
    float emergence_threshold;

    /* Live metrics */
    float entropy;
    float resonance;
    float emergence;
    float field_health;

    /* 4.C — Seasonal meta-operators */
    int   season;
    float season_phase;
    float season_intensity;
    float spring_energy, summer_energy, autumn_energy, winter_energy;

    /* Schumann resonance — Earth coupling */
    float schumann_hz;
    float schumann_coherence;
    float schumann_phase;
    float schumann_modulation;

    /* Expert blending (4 internal experts for temperature) */
    float expert_structural, expert_semantic, expert_creative, expert_precise;

    /* Tunneling */
    float tunnel_threshold;
    float tunnel_chance;
    int   tunnel_skip_max;

    /* Calendar drift (Hebrew-Gregorian conflict) */
    float calendar_drift;
    float calendar_phase;
    float wormhole;
    float wormhole_gate;
    int   wormhole_active;

    /* NOTORCH parameters */
    float notorch_lr;
    float notorch_decay;

    /* Identity */
    float essence_alpha;      /* γ injection strength */
    float lora_alpha;         /* δ voice strength */

    /* Presence */
    float presence_decay;
    float presence_fade;

    /* Dark matter — gravitational memory */
    float dark_gravity;

    /* Temporal debt */
    float temporal_debt;

    /* Step counter */
    int   step;
} FieldState;

/* 4.C MLP Controller — small neural net trained by Hebbian plasticity */
typedef struct {
    float w1[FIELD_4C_INPUTS * FIELD_4C_HIDDEN];
    float b1[FIELD_4C_HIDDEN];
    float w2[FIELD_4C_HIDDEN * FIELD_4C_OUTPUTS];
    float b2[FIELD_4C_OUTPUTS];
    float hidden[FIELD_4C_HIDDEN];
} FieldMLP;

static FieldState F;
static FieldMLP   F_mlp;

/* Schumann harmonics */
static const float g_schumann_harmonics[SCHUMANN_N_HARMONICS] = {
    7.83f, 14.1f, 20.3f, 26.4f, 32.5f
};
static const float g_harmonic_weights[SCHUMANN_N_HARMONICS] = {
    1.0f, 0.5f, 0.3f, 0.2f, 0.1f
};

/* Hebrew-Gregorian calendar */
static const int g_metonic_leaps[7] = {3, 6, 8, 11, 14, 17, 19};
static time_t g_epoch_t = 0;

static void calendar_init(void) {
    struct tm ep = {0};
    ep.tm_year = 2024 - 1900; ep.tm_mon = 9; ep.tm_mday = 3; ep.tm_hour = 12;
    g_epoch_t = mktime(&ep);
}

static float calendar_dissonance(void) {
    if (g_epoch_t <= 0) return 0;
    int days = (int)(difftime(time(NULL), g_epoch_t) / 86400.0);
    float years = (float)days / 365.25f;
    float drift = years * 11.25f;
    int full = (int)(years / 19); float corrections = (float)(full * 7) * 30.0f;
    float partial = fmodf(years, 19.0f);
    int yr = (int)partial + 1;
    for (int i = 0; i < 7; i++) if (g_metonic_leaps[i] <= yr) corrections += 30.0f;
    drift -= corrections;
    float raw = fabsf(fmodf(drift, 33.0f)) / 33.0f;
    return clamp01(raw);
}

static void field_mlp_init(void) {
    memset(&F_mlp, 0, sizeof(F_mlp));
    /* 4 specialist neurons — from AML core am_4c_init_weights */
    F_mlp.w1[0 * FIELD_4C_HIDDEN + 0] = -2.0f; F_mlp.b1[0] = 0.5f;
    F_mlp.w2[0 * FIELD_4C_OUTPUTS + 0] = 1.5f;  /* low entropy → spring */
    F_mlp.w1[1 * FIELD_4C_HIDDEN + 1] = 2.0f;  F_mlp.b1[1] = -1.5f;
    F_mlp.w2[1 * FIELD_4C_OUTPUTS + 2] = 1.5f;  /* high resonance → autumn */
    F_mlp.w1[2 * FIELD_4C_HIDDEN + 2] = 2.5f;  F_mlp.b1[2] = -1.5f;
    F_mlp.w2[2 * FIELD_4C_OUTPUTS + 3] = 1.5f;  /* high pain → winter */
    F_mlp.w1[4 * FIELD_4C_HIDDEN + 3] = 2.5f;  F_mlp.b1[3] = -0.5f;
    F_mlp.w2[3 * FIELD_4C_OUTPUTS + 1] = 1.5f;  /* high emergence → summer */
    /* cross-connections for nuance */
    F_mlp.w1[3 * FIELD_4C_HIDDEN + 4] = 0.5f;
    F_mlp.w1[5 * FIELD_4C_HIDDEN + 4] = -0.3f;
    F_mlp.w2[4 * FIELD_4C_OUTPUTS + 0] = 0.3f;
    F_mlp.w2[4 * FIELD_4C_OUTPUTS + 1] = -0.3f;
    F_mlp.w1[0 * FIELD_4C_HIDDEN + 5] = -1.0f;
    F_mlp.w1[1 * FIELD_4C_HIDDEN + 5] = 1.0f;
    F_mlp.w2[5 * FIELD_4C_OUTPUTS + 2] = 0.5f;
    F_mlp.w1[5 * FIELD_4C_HIDDEN + 6] = 1.5f; F_mlp.b1[6] = -1.0f;
    F_mlp.w2[6 * FIELD_4C_OUTPUTS + 3] = 0.4f;
    F_mlp.w1[4 * FIELD_4C_HIDDEN + 7] = 1.0f;
    F_mlp.w1[2 * FIELD_4C_HIDDEN + 7] = -1.0f;
    F_mlp.w2[7 * FIELD_4C_OUTPUTS + 1] = 0.5f;
}

static void field_init(void) {
    memset(&F, 0, sizeof(F));
    F.prophecy = 7;
    F.destiny = 0.35f;
    F.debt_decay = 0.998f;
    F.velocity_mode = VEL_WALK;
    F.base_temperature = 1.0f;
    F.time_direction = 1.0f;
    F.attend_focus = 0.70f;
    F.attend_spread = 0.20f;
    F.entropy_floor = 0.1f;
    F.resonance_ceiling = 0.95f;
    F.emergence_threshold = 0.3f;
    F.season = SEASON_SPRING;
    F.season_intensity = 0.5f;
    F.spring_energy = 1.0f;
    F.schumann_hz = SCHUMANN_BASE_HZ;
    F.schumann_modulation = 0.3f;
    F.schumann_coherence = 1.0f;
    F.tunnel_threshold = 0.55f;
    F.tunnel_chance = 0.05f;
    F.tunnel_skip_max = 7;
    F.calendar_drift = 11.0f;
    F.wormhole = 0.02f;
    F.wormhole_gate = 0.3f;
    F.notorch_lr = 0.01f;
    F.notorch_decay = 0.999f;
    F.essence_alpha = 0.5f;
    F.lora_alpha = 0.1f;
    F.presence_decay = 1.0f;
    F.presence_fade = 0.95f;
    F.dark_gravity = 0.5f;
    F.effective_temp = 0.85f;
    F.expert_structural = 0.25f;
    F.expert_semantic = 0.25f;
    F.expert_creative = 0.25f;
    F.expert_precise = 0.25f;
    calendar_init();
    field_mlp_init();
    printf("[doe] θ = ε + γ + αδ — parliament awakens. prophecy=%d destiny=%.2f\n",
           F.prophecy, F.destiny);
}

/* ─── Schumann resonance ─── */
static float schumann_coherence(float hz) {
    float d = fabsf(hz - SCHUMANN_BASE_HZ), mx = 32.5f - 4.0f;
    return clamp01(1.0f - (d/mx)*(d/mx));
}

static float schumann_signal(void) {
    float s = 0, w = 0;
    for (int i = 0; i < SCHUMANN_N_HARMONICS; i++) {
        float hp = F.schumann_phase * (g_schumann_harmonics[i] / SCHUMANN_BASE_HZ);
        s += g_harmonic_weights[i] * sinf(hp);
        w += g_harmonic_weights[i];
    }
    return w > 0 ? s / w : 0;
}

/* ─── 4.C MLP forward ─── */
static void field_mlp_forward(const float *in, float *out) {
    for (int h = 0; h < FIELD_4C_HIDDEN; h++) {
        float s = F_mlp.b1[h];
        for (int i = 0; i < FIELD_4C_INPUTS; i++) s += F_mlp.w1[i * FIELD_4C_HIDDEN + h] * in[i];
        F_mlp.hidden[h] = tanhf(s);
    }
    for (int o = 0; o < FIELD_4C_OUTPUTS; o++) {
        float s = F_mlp.b2[o];
        for (int h = 0; h < FIELD_4C_HIDDEN; h++) s += F_mlp.w2[h * FIELD_4C_OUTPUTS + o] * F_mlp.hidden[h];
        out[o] = tanhf(s);
    }
}

/* ─── 4.C Hebbian update ─── */
static void field_mlp_hebbian(const float *in, const float *out, float signal) {
    float lr = F.notorch_lr * 0.1f;
    for (int h = 0; h < FIELD_4C_HIDDEN; h++)
        for (int o = 0; o < FIELD_4C_OUTPUTS; o++) {
            F_mlp.w2[h * FIELD_4C_OUTPUTS + o] += lr * F_mlp.hidden[h] * out[o] * signal;
            if (F_mlp.w2[h*FIELD_4C_OUTPUTS+o] > 3.0f) F_mlp.w2[h*FIELD_4C_OUTPUTS+o] = 3.0f;
            if (F_mlp.w2[h*FIELD_4C_OUTPUTS+o] < -3.0f) F_mlp.w2[h*FIELD_4C_OUTPUTS+o] = -3.0f;
        }
    for (int i = 0; i < FIELD_4C_INPUTS; i++)
        for (int h = 0; h < FIELD_4C_HIDDEN; h++) {
            F_mlp.w1[i * FIELD_4C_HIDDEN + h] += lr * in[i] * F_mlp.hidden[h] * signal;
            if (F_mlp.w1[i*FIELD_4C_HIDDEN+h] > 3.0f) F_mlp.w1[i*FIELD_4C_HIDDEN+h] = 3.0f;
            if (F_mlp.w1[i*FIELD_4C_HIDDEN+h] < -3.0f) F_mlp.w1[i*FIELD_4C_HIDDEN+h] = -3.0f;
        }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * FIELD STEP — the heartbeat. from AML am_step(), distilled for DOE.
 * called per token. advances field physics by dt seconds.
 *
 * 1. calendar conflict → wormhole activation → dissonance bleed
 * 2. debt decay (prophecy debt × decay_rate)
 * 3. Schumann resonance → tension/dissonance healing
 * 4. destiny bias computation
 * 5. velocity + expert blending → effective temperature
 * 6. law enforcement (entropy floor, resonance ceiling)
 * 7. 4.C seasonal MLP controller + Hebbian update
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void field_step(float dt) {
    if (dt <= 0) return;
    F.step++;

    /* ── Calendar conflict ── */
    float cal_d = calendar_dissonance();
    if (cal_d > F.wormhole_gate) {
        F.wormhole_active = 1;
        float excess = (cal_d - F.wormhole_gate) / (1.0f - F.wormhole_gate);
        F.wormhole = clamp01(F.wormhole + excess * 0.1f * dt);
    } else {
        F.wormhole_active = 0;
        F.wormhole *= 0.995f;
        if (F.wormhole < 0.02f) F.wormhole = 0.02f;
    }
    if (cal_d > 0.3f) {
        F.dissonance += (cal_d - 0.3f) * 0.05f * dt;
        if (F.dissonance > 1.0f) F.dissonance = 1.0f;
    }
    F.debt += cal_d * 0.005f * dt;

    /* ── Debt decay ── */
    F.debt *= F.debt_decay;
    if (F.debt > 100.0f) F.debt = 100.0f;

    /* ── Temporal debt ── */
    if (F.velocity_mode == VEL_BACKWARD) F.temporal_debt += 0.01f * dt;
    else F.temporal_debt *= 0.9995f;
    if (F.temporal_debt > 10.0f) F.temporal_debt = 10.0f;

    /* ── Schumann resonance healing ── */
    F.schumann_phase += F.schumann_hz * dt * 6.2831853f;
    if (F.schumann_phase > 6.2831853f) F.schumann_phase = fmodf(F.schumann_phase, 6.2831853f);
    F.schumann_coherence = schumann_coherence(F.schumann_hz);
    if (F.schumann_coherence > 0 && F.schumann_modulation > 0) {
        float cf = 0.5f + 0.5f * F.schumann_coherence;
        float hm = 1.0f + schumann_signal() * 0.1f;
        float heal = 0.998f - 0.003f * cf * F.schumann_modulation * hm;
        F.tension *= heal;
        F.dissonance *= heal;
    }

    /* ── Destiny bias ── */
    float ps = 1.0f + ((float)F.prophecy - 7.0f) * 0.02f;
    if (ps < 0.5f) ps = 0.5f; if (ps > 2.0f) ps = 2.0f;
    F.destiny_bias = F.destiny * ps;

    /* ── Velocity + expert blending → effective temperature ── */
    {
        float vm;
        switch (F.velocity_mode) {
            case VEL_NOMOVE: vm = 0.5f; F.time_direction = 1.0f; break;
            case VEL_WALK: vm = 0.85f; F.time_direction = 1.0f; break;
            case VEL_RUN: vm = 1.2f; F.time_direction = 1.0f; break;
            case VEL_BACKWARD: vm = 0.7f; F.time_direction = -1.0f; break;
            default: vm = 1.0f; F.time_direction = 1.0f;
        }
        float vt = F.base_temperature * vm;
        float ws = F.expert_structural + F.expert_semantic + F.expert_creative + F.expert_precise;
        if (ws > 0.001f) {
            float et = (F.expert_structural*0.7f + F.expert_semantic*0.9f +
                       F.expert_creative*1.2f + F.expert_precise*0.5f) / ws;
            F.effective_temp = 0.5f * vt + 0.5f * et;
        } else F.effective_temp = vt;
        float sm = 1.0f + F.summer_energy * 0.1f - F.winter_energy * 0.15f;
        F.effective_temp *= sm;
        if (F.effective_temp < 0.1f) F.effective_temp = 0.1f;
    }

    /* ── Law enforcement ── */
    {
        float re = (F.effective_temp - 0.5f)*0.3f + F.dissonance*0.3f +
                   F.tunnel_chance*0.2f + (1.0f - F.attend_focus)*0.2f;
        F.entropy = fmaxf(F.entropy_floor, clamp01(re));
        float rr = F.schumann_coherence*0.3f + (1.0f-F.dissonance)*0.3f +
                   F.attend_focus*0.2f + (1.0f - clamp01(F.debt*0.1f))*0.2f;
        F.resonance = fminf(F.resonance_ceiling, clamp01(rr));
        F.emergence = clamp01((1.0f - F.entropy) * F.resonance);
    }

    /* ── Presence fade ── */
    F.presence_decay *= F.presence_fade;
    if (F.presence_decay < 0.001f) F.presence_decay = 0.001f;

    /* ── 4.C Seasonal MLP controller ── */
    {
        float sr = 0.001f;
        F.season_phase += sr * dt;
        if (F.season_phase >= 1.0f) { F.season_phase = 0; F.season = (F.season+1)%4; }
        float gain = 0.02f * dt * F.season_intensity, fade = 0.995f;
        F.spring_energy *= fade; F.summer_energy *= fade;
        F.autumn_energy *= fade; F.winter_energy *= fade;
        switch (F.season) {
            case SEASON_SPRING: F.spring_energy = clamp01(F.spring_energy + gain); break;
            case SEASON_SUMMER: F.summer_energy = clamp01(F.summer_energy + gain); break;
            case SEASON_AUTUMN: F.autumn_energy = clamp01(F.autumn_energy + gain); break;
            case SEASON_WINTER: F.winter_energy = clamp01(F.winter_energy + gain); break;
        }
        float mlp_in[FIELD_4C_INPUTS] = {
            F.entropy, F.resonance, F.pain, F.tension, F.emergence, F.effective_temp
        };
        float mlp_out[FIELD_4C_OUTPUTS];
        field_mlp_forward(mlp_in, mlp_out);
        float sc = 0.02f * dt * F.season_intensity;
        F.spring_energy = clamp01(F.spring_energy + mlp_out[0]*sc);
        F.summer_energy = clamp01(F.summer_energy + mlp_out[1]*sc);
        F.autumn_energy = clamp01(F.autumn_energy + mlp_out[2]*sc);
        F.winter_energy = clamp01(F.winter_energy + mlp_out[3]*sc);
        /* Hebbian: did the field improve? */
        float health = clamp01((1.0f - fabsf(F.entropy - 0.5f)) * F.resonance * (1.0f - F.pain));
        float sig = health - F.field_health;
        F.field_health = health;
        if (fabsf(sig) > 0.001f) field_mlp_hebbian(mlp_in, mlp_out, sig);
        /* Season effects */
        F.tunnel_chance = clamp01(F.tunnel_chance + F.spring_energy * 0.005f * dt);
        F.dark_gravity = clamp01(F.dark_gravity + F.autumn_energy * 0.002f * dt);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PROPHECY DEBT — retroactive conscience.
 * every token you choose that isn't the destined one accumulates debt.
 * not minimize(predicted - actual) but minimize(destined - manifested).
 * the difference is intention. the difference is identity.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float compute_prophecy_debt(const float *logits, int chosen, int n) {
    if (n <= 0 || chosen < 0 || chosen >= n) return 0;
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float diff = mx - logits[chosen];
    return diff > 0 ? diff / (diff + 1.0f) : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * FIELD → LOGITS — the full pipeline. from AML am_apply_field_to_logits().
 *
 * 1. destiny bias: suppress low-probability tokens
 * 2. suffering: compress toward mean (pain dampens extremes)
 * 3. attention: sharpen or blur distribution
 * 4. laws: entropy floor, resonance ceiling
 *
 * this is not post-processing. this is the architecture speaking.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void apply_destiny(float *logits, int n) {
    if (n <= 0 || F.destiny_bias < 0.001f) return;
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < n; i++) {
        float diff = mx - logits[i];
        logits[i] -= diff * F.destiny_bias * 0.5f;
    }
}

static void apply_suffering(float *logits, int n) {
    if (n <= 0) return;
    float total = F.pain + F.tension * 0.5f;
    if (total < 0.01f) return;
    float mean = 0;
    for (int i = 0; i < n; i++) mean += logits[i];
    mean /= n;
    float compress = total * 0.3f;
    for (int i = 0; i < n; i++) logits[i] = logits[i] * (1.0f - compress) + mean * compress;
}

static void apply_attention(float *logits, int n) {
    if (n <= 0) return;
    float focus = F.attend_focus;
    if (focus < 0.01f) return;
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < n; i++) {
        float d = mx - logits[i];
        logits[i] -= d * focus * 0.2f;
    }
}

static void apply_field_to_logits(float *logits, int n) {
    apply_destiny(logits, n);
    apply_suffering(logits, n);
    apply_attention(logits, n);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DEQUANTIZATION — Q4_0, Q8_0, Q4_K, Q6_K → f32
 * Ported from nanollama/go/quant.go. Dequant at load time.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF; f = (sign<<31)|((exp+127-15)<<23)|(mant<<13); }
    } else if (exp == 31) f = (sign<<31)|0x7F800000|(mant<<13);
    else f = (sign<<31)|((exp+127-15)<<23)|(mant<<13);
    float r; memcpy(&r, &f, 4); return r;
}

/* Q4_0: block = 2 bytes f16 scale + 16 bytes (32 nibbles) = 18 bytes, 32 values */
#define Q4_0_BLOCK 32
#define Q4_0_BYTES 18
static void dequant_q4_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q4_0_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q4_0_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        for (int j = 0; j < 16; j++) {
            int v0 = (b[2+j] & 0x0F) - 8;
            int v1 = (b[2+j] >> 4) - 8;
            out[i*Q4_0_BLOCK + j] = (float)v0 * d;
            out[i*Q4_0_BLOCK + j + 16] = (float)v1 * d;
        }
    }
}

/* Q8_0: block = 2 bytes f16 scale + 32 bytes int8 = 34 bytes, 32 values */
#define Q8_0_BLOCK 32
#define Q8_0_BYTES 34
static void dequant_q8_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q8_0_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q8_0_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        for (int j = 0; j < 32; j++)
            out[i*Q8_0_BLOCK + j] = (float)((int8_t)b[2+j]) * d;
    }
}

/* Q4_K: block = 2+2 bytes f16 (d, dmin) + 12 bytes scales + 128 nibbles = 144 bytes, 256 values */
#define Q4_K_BLOCK 256
#define Q4_K_BYTES 144
static void get_scale_min_k4(int j, const uint8_t *sc, uint8_t *s, uint8_t *m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j+4] & 63; }
    else { *s = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4); *m = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4); }
}
static void dequant_q4_k(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q4_K_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q4_K_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        float dmin = f16_to_f32(b[2] | (b[3] << 8));
        const uint8_t *sc = b + 4, *qs = b + 16;
        int is = 0, qi = 0, oi = (int)(i * Q4_K_BLOCK);
        for (int j = 0; j < Q4_K_BLOCK; j += 64) {
            uint8_t sc0, m0, sc1, m1v;
            get_scale_min_k4(is, sc, &sc0, &m0);
            float d1 = d * (float)sc0, mm1 = dmin * (float)m0;
            get_scale_min_k4(is+1, sc, &sc1, &m1v);
            float d2 = d * (float)sc1, mm2 = dmin * (float)m1v;
            for (int l = 0; l < 32; l++)
                out[oi + j + l] = d1 * (float)(qs[qi+l] & 0x0F) - mm1;
            for (int l = 0; l < 32; l++)
                out[oi + j + 32 + l] = d2 * (float)(qs[qi+l] >> 4) - mm2;
            qi += 32; is += 2;
        }
    }
}

/* Q5_0: block = 2 bytes f16 scale + 4 bytes high bits + 16 bytes nibbles = 22 bytes, 32 values */
#define Q5_0_BLOCK 32
#define Q5_0_BYTES 22
static void dequant_q5_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q5_0_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q5_0_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        uint32_t qh = b[2] | (b[3]<<8) | (b[4]<<16) | (b[5]<<24);
        const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; j++) {
            int lo = qs[j] & 0x0F, hi = qs[j] >> 4;
            int hbit0 = (qh >> j) & 1, hbit1 = (qh >> (j+16)) & 1;
            out[i*Q5_0_BLOCK + j] = (float)((lo | (hbit0<<4)) - 16) * d;
            out[i*Q5_0_BLOCK + j + 16] = (float)((hi | (hbit1<<4)) - 16) * d;
        }
    }
}

/* Q6_K: block = 128 ql + 64 qh + 16 scales + 2 d = 210 bytes, 256 values */
#define Q6_K_BLOCK 256
#define Q6_K_BYTES 210
static void dequant_q6_k(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q6_K_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q6_K_BYTES;
        const uint8_t *ql = b, *qh = b + 128, *sc = b + 192;
        float d = f16_to_f32(b[208] | (b[209] << 8));
        int oi = (int)(i * Q6_K_BLOCK);
        for (int n128 = 0; n128 < 2; n128++) {
            const uint8_t *qlp = ql + n128*64, *qhp = qh + n128*32;
            const uint8_t *scp = sc + n128*8;
            int yo = oi + n128*128;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (qlp[l] & 0x0F) | ((qhp[l] >> 0) & 3) << 4;
                int q2 = (qlp[l+32] & 0x0F) | ((qhp[l] >> 2) & 3) << 4;
                int q3 = (qlp[l] >> 4) | ((qhp[l] >> 4) & 3) << 4;
                int q4 = (qlp[l+32] >> 4) | ((qhp[l] >> 6) & 3) << 4;
                out[yo+l+0]  = d * (float)((int8_t)scp[is+0]) * (float)(q1-32);
                out[yo+l+32] = d * (float)((int8_t)scp[is+2]) * (float)(q2-32);
                out[yo+l+64] = d * (float)((int8_t)scp[is+4]) * (float)(q3-32);
                out[yo+l+96] = d * (float)((int8_t)scp[is+6]) * (float)(q4-32);
            }
        }
    }
}

/* bytes per element for each quant type (for raw data size calculation) */
static uint64_t quant_raw_bytes(uint32_t dtype, uint64_t n_elems) {
    switch (dtype) {
        case 0: return n_elems * 4;   /* f32 */
        case 1: return n_elems * 2;   /* f16 */
        case 2: return (n_elems / Q4_0_BLOCK) * Q4_0_BYTES;  /* Q4_0 */
        case 6: return (n_elems / Q5_0_BLOCK) * Q5_0_BYTES;  /* Q5_0 */
        case 8: return (n_elems / Q8_0_BLOCK) * Q8_0_BYTES;  /* Q8_0 */
        case 12: return (n_elems / Q4_K_BLOCK) * Q4_K_BYTES; /* Q4_K */
        case 14: return (n_elems / Q6_K_BLOCK) * Q6_K_BYTES; /* Q6_K */
        default: return 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MATH OPS — building blocks
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float silu_f(float x) { return x / (1.0f + expf(-x)); }

static void rmsnorm(float *out, const float *x, const float *w, int d, float eps) {
    float ss = 0; for (int i = 0; i < d; i++) ss += x[i]*x[i];
    float inv = 1.0f / sqrtf(ss/d + eps);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv * w[i];
}

/* threaded matvec worker */
typedef struct { float *out; const float *W; const float *x; int r0, r1, c; } MVWork;
static void *matvec_worker(void *arg) {
    MVWork *w = (MVWork*)arg;
    for (int i = w->r0; i < w->r1; i++) {
        float s = 0; const float *row = w->W + (size_t)i * w->c;
        for (int j = 0; j < w->c; j++) s += row[j] * w->x[j];
        w->out[i] = s;
    }
    return NULL;
}

static int g_n_threads = 0;

static void matvec(float *out, const float *W, const float *x, int r, int c) {
#ifdef USE_CUBLAS
    cublas_init();
    float *dW = gpu_scratch(0,(size_t)r*c*4), *dx = gpu_scratch(1,(size_t)c*4), *dy = gpu_scratch(2,(size_t)r*4);
    cudaMemcpy(dW, W, (size_t)r*c*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, (size_t)c*4, cudaMemcpyHostToDevice);
    float a=1,b=0;
    cublasSgemv(g_cublas, CUBLAS_OP_T, c, r, &a, dW, c, dx, 1, &b, dy, 1);
    cudaMemcpy(out, dy, (size_t)r*4, cudaMemcpyDeviceToHost);
#elif defined(USE_BLAS)
    cblas_sgemv(CblasRowMajor,CblasNoTrans,r,c,1.0f,W,c,x,1,0.0f,out,1);
#else
    int nt = g_n_threads;
    if (nt <= 1 || r < 64) {
        for (int i = 0; i < r; i++) {
            float s = 0; const float *row = W + (size_t)i*c;
            for (int j = 0; j < c; j++) s += row[j] * x[j];
            out[i] = s;
        }
        return;
    }
    if (nt > 32) nt = 32;
    pthread_t thr[32]; MVWork work[32];
    int chunk = (r + nt - 1) / nt;
    int actual = 0;
    for (int t = 0; t < nt; t++) {
        int r0 = t * chunk, r1 = r0 + chunk;
        if (r0 >= r) break;
        if (r1 > r) r1 = r;
        work[t] = (MVWork){out, W, x, r0, r1, c};
        pthread_create(&thr[t], NULL, matvec_worker, &work[t]);
        actual++;
    }
    for (int t = 0; t < actual; t++) pthread_join(thr[t], NULL);
#endif
}

static void softmax_n(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void apply_rope(float *v, int pos, float *cc, float *sc, int hd) {
    int h = hd/2, off = pos*h; /* hd must be even — all standard archs are */
    for (int i = 0; i < h; i++) {
        float x0 = v[i], x1 = v[i+h];
        v[i] = x0*cc[off+i] - x1*sc[off+i];
        v[i+h] = x0*sc[off+i] + x1*cc[off+i];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * HARMONIC RESONANCE ENGINE — from AML/DOE, adapted for field.
 * each expert has a frequency. input gets fourier-decomposed.
 * experts that resonate with input get boosted.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float amplitudes[HARMONIC_N];
    float dominant_freq;
    float confidence;
} HarmonicState;

static void harmonic_decompose(HarmonicState *hs, float *hist, int len) {
    float max_amp = 0; int max_k = 0;
    for (int k = 0; k < HARMONIC_N && k < len/2; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < len; n++) {
            float angle = 6.2831853f * k * n / len;
            re += hist[n] * cosf(angle);
            im += hist[n] * sinf(angle);
        }
        hs->amplitudes[k] = sqrtf(re*re + im*im) / len;
        if (k > 0 && hs->amplitudes[k] > max_amp) { max_amp = hs->amplitudes[k]; max_k = k; }
    }
    hs->dominant_freq = len > 0 ? 6.2831853f * max_k / len : 0;
    float total = 0;
    for (int k = 0; k < HARMONIC_N; k++) total += hs->amplitudes[k];
    hs->confidence = total > 1e-8f ? max_amp / total : 0;
}

static float expert_resonance(float expert_freq, HarmonicState *hs) {
    float res = 0;
    for (int k = 0; k < HARMONIC_N; k++) {
        float fk = 6.2831853f * k / HARMONIC_N;
        float dist = fabsf(expert_freq - fk);
        if (dist > 3.14159f) dist = 6.2831853f - dist;
        res += hs->amplitudes[k] * expf(-dist*dist*2.0f);
    }
    return res;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * WEIGHT PROFILER — DOE's sonar.
 * before attaching, DOE profiles the host's weights.
 * L2 norms per layer, spectral density, dead neuron ratio.
 * this tells DOE where to focus its LoRA experts.
 *
 * the index is read-only. DOE is the architecture.
 * weak layers get more LoRA. healthy layers get less.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float l2_norm;            /* L2 norm of layer weights */
    float mean_abs;           /* mean absolute value */
    float std_dev;            /* standard deviation */
    float sparsity;           /* fraction near zero (<1e-6) */
    float spectral_energy;    /* energy in top 10% singular values (approx) */
    int   dead_neurons;       /* rows/cols with near-zero norm */
    float health;             /* composite: 0=dead, 1=vibrant */
} LayerProfile;

typedef struct {
    LayerProfile layers[MAX_LAYERS];
    int n_layers;
    float overall_health;     /* average layer health */
    float code_affinity;      /* estimated code capability (from weight stats) */
    float complexity;         /* model complexity metric */
    uint64_t fingerprint;     /* hash of weight statistics — identifies this host */
} WeightProfile;

static void profile_weights(float *data, int rows, int cols, LayerProfile *out) {
    int n = rows * cols;
    if (n == 0) { memset(out, 0, sizeof(LayerProfile)); return; }
    float sum = 0, sum_sq = 0, sum_abs = 0;
    int near_zero = 0;
    for (int i = 0; i < n; i++) {
        float v = data[i];
        sum += v; sum_sq += v*v; sum_abs += fabsf(v);
        if (fabsf(v) < 1e-6f) near_zero++;
    }
    float mean = sum / n;
    out->l2_norm = sqrtf(sum_sq);
    out->mean_abs = sum_abs / n;
    out->std_dev = sqrtf(sum_sq/n - mean*mean);
    out->sparsity = (float)near_zero / n;

    /* Approximate spectral energy: sample random directions */
    float top_energy = 0;
    for (int trial = 0; trial < 8; trial++) {
        float dot = 0;
        for (int j = 0; j < cols; j++) {
            float r = rand_normal();
            float proj = 0;
            for (int i = 0; i < rows; i++) proj += data[i*cols+j] * r;
            dot += proj * proj;
        }
        top_energy += sqrtf(dot);
    }
    out->spectral_energy = top_energy / 8.0f;

    /* Dead neurons: rows with near-zero norm */
    out->dead_neurons = 0;
    for (int r = 0; r < rows; r++) {
        float rn = 0;
        for (int c = 0; c < cols; c++) rn += data[r*cols+c] * data[r*cols+c];
        if (sqrtf(rn) < 1e-4f) out->dead_neurons++;
    }

    /* Composite health */
    float alive_ratio = 1.0f - (float)out->dead_neurons / (rows > 0 ? rows : 1);
    float activity = fminf(1.0f, out->std_dev * 10.0f);
    float density = 1.0f - out->sparsity;
    out->health = alive_ratio * 0.4f + activity * 0.3f + density * 0.3f;
}

static uint64_t compute_fingerprint(WeightProfile *wp) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < wp->n_layers; i++) {
        uint32_t bits;
        memcpy(&bits, &wp->layers[i].l2_norm, 4);
        h ^= (uint64_t)bits; h *= 1099511628211ULL;
        memcpy(&bits, &wp->layers[i].std_dev, 4);
        h ^= (uint64_t)bits; h *= 1099511628211ULL;
    }
    return h;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LIVING LoRA EXPERTS — DOE's democracy, adapted for symbiosis.
 * instead of standalone FFN experts, these are LoRA overlays.
 * each expert has A[dim, rank] and B[rank, dim] — Delta Voice injection.
 * Delta Voice: out += α × A @ (B @ x)
 *
 * experts still live and die. overloaded → mitosis. neglected → apoptosis.
 * but now they modulate the host's attention, not replace it.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float *lora_A;            /* [dim, rank] — output projection */
    float *lora_B;            /* [rank, dim] — input projection */
    float frequency;          /* position in harmonic space */
    float vitality;           /* 0.0=dying, 1.0=peak */
    float specialization;     /* entropy of routing distribution */
    int   age;
    int   tokens_seen;
    int   alive;
    int   low_vitality_count;
    float attention_bias;     /* per-expert attention scaling */
    float layer_focus;        /* per-expert residual contribution */
} LoraExpert;

typedef struct {
    float *w_vote;            /* [MAX_EXPERTS * dim] */
    float consensus;
    float faction_power[MAX_EXPERTS];
    int   election_count;
} Parliament;

typedef struct {
    Parliament parliament;
    LoraExpert experts[MAX_EXPERTS];
    int n_alive;
    int host_layer_idx;       /* which host layer this wraps */
} FieldLayer;

/* ═══════════════════════════════════════════════════════════════════════════════
 * INDEX STATE — the full host-DOE interface.
 * mmap'd host model + DOE's living LoRA overlay + weight profile.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    /* Host model — mmap'd, read-only */
    uint8_t *mmap_base;
    size_t   mmap_size;
    int      host_n_layers, host_dim, host_hidden, host_heads, host_kv_heads, host_head_dim;
    int      host_vocab;
    char     host_arch[64];
    char     host_path[256];

    /* Host weight pointers (into mmap'd region) */
    float *host_tok_emb;
    float *host_output;
    float *host_norm;
    float    rope_theta;     /* RoPE frequency base (default 10000, Qwen=1000000) */
    float    rms_norm_eps;   /* RMSNorm epsilon (default 1e-5, varies per arch) */
    struct {
        float *wq, *wk, *wv, *wo;
        float *bq, *bk, *bv;   /* attention biases (Qwen2, optional) */
        float *ffn_gate, *ffn_up, *ffn_down;
        float *ffn_gate_up;  /* fused gate+up for Phi-3 (size: hidden*2 × dim) */
        float *attn_norm, *ffn_norm;
    } host_layers[MAX_LAYERS];

    /* DOE's living overlay */
    FieldLayer field_layers[MAX_LAYERS];
    int n_field_layers;

    /* Host profiling */
    WeightProfile profile;

    /* LoRA parameters */
    int   lora_rank;
    float lora_alpha;

    /* Active flag */
    int active;

    /* f16→f32 conversion buffers (must be freed on cleanup) */
    float **f16_bufs;
    int     n_f16_bufs;

    /* Tokenizer from GGUF metadata */
    char  **vocab_tokens;   /* token strings, indexed by token id */
    float  *vocab_scores;   /* BPE merge scores per token (SentencePiece) or from merges (GPT-2) */
    int     vocab_size;     /* number of entries */
    int     bos_id, eos_id; /* special tokens */
    int     add_space_prefix;
    int     is_gpt2_bpe;    /* 1 if tokenizer.ggml.model == "gpt2" */

    /* GPT-2 BPE merges (used to build scores if no native scores) */
    char  **bpe_merges;     /* merge strings "A B" */
    int     n_bpe_merges;

    /* Token hash table for O(1) lookup */
    int    *tok_ht_ids;     /* hash table: token id or -1 */
    int     tok_ht_cap;     /* hash table capacity (power of 2) */

    /* Chat template detection */
    int     chat_style;     /* 0=raw, 1=chatml, 2=llama/mistral [INST], 3=zephyr, 4=phi, 5=gemma, 6=nanollama */

    /* Identity & gamma */
    int     weightless;     /* 1 if no doe_identity.gguf found */
    char    identity_tag[128]; /* doe.identity metadata from GGUF — empty if not DOE's own */
    void   *gamma_data;     /* raw gamma binary blob */
    int     gamma_size;     /* gamma blob size in bytes */
} GGUFIndex;

typedef struct { char name[96]; uint32_t ndim; uint64_t dims[4]; uint32_t dtype; uint64_t offset; } TensorInfo;

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENVIRONMENT SCANNER — DOE opens its eyes
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    char path[256]; char arch[64]; int n_layers, dim, n_heads;
    int64_t file_size; float compatibility;
} DiscoveredGGUF;

typedef struct {
    DiscoveredGGUF ggufs[32]; int n_ggufs;
    int64_t disk_free, mem_available;
    int cpu_count, has_compiler, has_curl;
    char self_path[256];
} Environment;

static int gguf_sniff(const char *path, DiscoveredGGUF *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    struct stat st; fstat(fileno(f), &st); out->file_size = st.st_size;
    snprintf(out->path, 256, "%s", path);
    memset(out->arch, 0, 64); out->n_layers = 0; out->dim = 0; out->n_heads = 0;
    uint32_t magic; if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) { fclose(f); return 0; }
    uint32_t version; fread(&version, 4, 1, f);
    uint64_t n_tensors, n_kv; fread(&n_tensors, 8, 1, f); fread(&n_kv, 8, 1, f);
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen; if (fread(&klen, 8, 1, f) != 1) break;
        if (klen > 255) { fseek(f, klen + 4, SEEK_CUR); continue; }
        char key[256]; if (fread(key, 1, klen, f) != klen) break; key[klen] = '\0';
        uint32_t vtype; if (fread(&vtype, 4, 1, f) != 1) break;
        if (vtype == 8) { /* string */
            uint64_t vlen; fread(&vlen, 8, 1, f); char val[256];
            int rl = vlen < 255 ? (int)vlen : 255; fread(val, 1, rl, f); val[rl] = '\0';
            if (vlen > 255) fseek(f, vlen-255, SEEK_CUR);
            if (strstr(key, "general.architecture")) snprintf(out->arch, 64, "%s", val);
        } else if (vtype == 4) { uint32_t val; fread(&val, 4, 1, f);
            if (strstr(key, "embedding_length")) out->dim = (int)val;
            else if (strstr(key, "block_count")) out->n_layers = (int)val;
            else if (strstr(key, "head_count") && !strstr(key, "kv")) out->n_heads = (int)val;
        } else if (vtype == 0 || vtype == 1 || vtype == 7) fseek(f, 1, SEEK_CUR);
        else if (vtype == 2 || vtype == 3) fseek(f, 2, SEEK_CUR);
        else if (vtype == 5 || vtype == 6) fseek(f, 4, SEEK_CUR);
        else if (vtype == 10 || vtype == 11 || vtype == 12) fseek(f, 8, SEEK_CUR);
        else if (vtype == 9) { /* array */
            uint32_t atype; fread(&atype, 4, 1, f);
            uint64_t alen; fread(&alen, 8, 1, f);
            size_t esz = 0;
            if (atype == 0 || atype == 1 || atype == 7) esz = 1;
            else if (atype == 2 || atype == 3) esz = 2;
            else if (atype == 4 || atype == 5 || atype == 6) esz = 4;
            else if (atype == 10 || atype == 11 || atype == 12) esz = 8;
            else if (atype == 8) {
                for (uint64_t ai = 0; ai < alen; ai++) {
                    uint64_t sl; if (fread(&sl, 8, 1, f) != 1) break;
                    fseek(f, sl, SEEK_CUR);
                }
                continue;
            }
            fseek(f, alen * esz, SEEK_CUR);
        } else fseek(f, 4, SEEK_CUR); /* unknown — guess 4 */
    }
    fclose(f);
    return (out->arch[0] != '\0' && out->dim > 0);
}

static void env_scan(Environment *env, const char *self_src) {
    memset(env, 0, sizeof(Environment));
    snprintf(env->self_path, 256, "%s", self_src);
    env->cpu_count = (int)sysconf(_SC_NPROCESSORS_ONLN);
#ifdef __linux__
    env->mem_available = (int64_t)sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE);
    struct statvfs sv; if (statvfs(".", &sv) == 0) env->disk_free = (int64_t)sv.f_bavail * sv.f_frsize;
#elif defined(__APPLE__)
    int64_t mem = 0; size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0); env->mem_available = mem;
    struct statfs sf; if (statfs(".", &sf) == 0) env->disk_free = (int64_t)sf.f_bavail * sf.f_bsize;
#endif
    env->has_compiler = (system("which cc >/dev/null 2>&1") == 0);
    env->has_curl = (system("which curl >/dev/null 2>&1") == 0);
    FILE *p = popen("find . -name '*.gguf' -maxdepth 3 2>/dev/null", "r");
    if (p) {
        char line[256];
        while (fgets(line, sizeof(line), p) && env->n_ggufs < 32) {
            int len = strlen(line);
            while (len > 0 && (line[len-1]=='\n' || line[len-1]=='\r')) line[--len] = '\0';
            if (len == 0) continue;
            DiscoveredGGUF dg;
            if (gguf_sniff(line, &dg)) env->ggufs[env->n_ggufs++] = dg;
        }
        pclose(p);
    }
    printf("[env] cpu=%d mem=%.1fGB disk=%.1fGB compiler=%s curl=%s ggufs=%d\n",
           env->cpu_count, (float)env->mem_available/(1024*1024*1024),
           (float)env->disk_free/(1024*1024*1024),
           env->has_compiler?"yes":"no", env->has_curl?"yes":"no", env->n_ggufs);
    for (int i = 0; i < env->n_ggufs; i++)
        printf("  [gguf] %s arch=%s dim=%d layers=%d %.1fMB\n",
               env->ggufs[i].path, env->ggufs[i].arch, env->ggufs[i].dim,
               env->ggufs[i].n_layers, (float)env->ggufs[i].file_size/(1024*1024));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INDEX LOAD — mmap GGUF, wire weight pointers, profile layers, attach LoRA.
 * the weights are substrate. DOE is the architecture.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void init_lora_expert(LoraExpert *e, int dim, int rank, float freq) {
    e->lora_A = calloc(dim * rank, sizeof(float));
    e->lora_B = calloc(rank * dim, sizeof(float));
    float scale = 0.02f / sqrtf((float)rank);
    for (int i = 0; i < dim*rank; i++) e->lora_A[i] = rand_normal() * scale;
    for (int i = 0; i < rank*dim; i++) e->lora_B[i] = rand_normal() * scale;
    e->frequency = freq;
    e->vitality = 0.7f;
    e->alive = 1;
    e->attention_bias = 0.0f;
    e->layer_focus = 1.0f;
    e->low_vitality_count = 0;
}

static void free_lora_expert(LoraExpert *e) {
    free(e->lora_A); free(e->lora_B);
    e->lora_A = e->lora_B = NULL;
    e->alive = 0; e->vitality = 0;
}

static int tok_lookup(GGUFIndex *ps, const char *s, int len);
static void tok_ht_build(GGUFIndex *ps);
static void build_gpt2_scores(GGUFIndex *ps);

static int index_load(GGUFIndex *ps, const char *path) {
    memset(ps, 0, sizeof(GGUFIndex));
    snprintf(ps->host_path, 256, "%s", path);
    ps->lora_rank = LORA_RANK;
    ps->lora_alpha = F.lora_alpha;
    ps->bos_id = 1; ps->eos_id = 2; /* defaults, overridden by GGUF */
    ps->rope_theta = 10000.0f;
    ps->rms_norm_eps = 1e-5f;
    ps->add_space_prefix = 1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { printf("[doe] cannot open %s\n", path); return 0; }
    struct stat st; fstat(fd, &st);
    ps->mmap_size = st.st_size;
    ps->mmap_base = mmap(NULL, ps->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ps->mmap_base == MAP_FAILED) { ps->mmap_base = NULL; return 0; }

    /* Parse GGUF header */
    uint8_t *p = ps->mmap_base, *pend = ps->mmap_base + ps->mmap_size;
    #define PC(n) do { if (p + (n) > pend) goto bail; } while(0)
    PC(4); uint32_t magic = *(uint32_t*)p; p += 4;
    if (magic != 0x46554747) goto bail;
    PC(4); p += 4; /* version */
    PC(8); uint64_t n_tensors = *(uint64_t*)p; p += 8;
    PC(8); uint64_t n_kv = *(uint64_t*)p; p += 8;

    for (uint64_t i = 0; i < n_kv; i++) {
        PC(8); uint64_t klen = *(uint64_t*)p; p += 8;
        if (klen > 255) { p += klen + 4; continue; } /* skip long keys */
        char key[256]; memcpy(key, p, klen); key[klen] = '\0'; p += klen;
        PC(4); uint32_t vtype = *(uint32_t*)p; p += 4;
        if (vtype == 8) { /* string */
            PC(8); uint64_t vlen = *(uint64_t*)p; p += 8;
            if (strstr(key, "general.architecture") && vlen < 64) {
                memcpy(ps->host_arch, p, vlen); ps->host_arch[vlen] = 0;
            }
            if (strstr(key, "tokenizer.ggml.model") && vlen < 20) {
                char tok_model[24]; memcpy(tok_model, p, vlen); tok_model[vlen] = 0;
                if (strcmp(tok_model, "gpt2") == 0) ps->is_gpt2_bpe = 1;
            }
            /* DOE identity fingerprint — this GGUF is DOE's own */
            if (strcmp(key, "doe.identity") == 0 && vlen < 128) {
                memcpy(ps->identity_tag, p, vlen); ps->identity_tag[vlen] = 0;
                printf("[identity] GGUF self-identifies: \"%s\"\n", ps->identity_tag);
            }
            /* Detect chat template style from template string */
            if (strstr(key, "chat_template") && vlen > 10 && vlen < 100000) {
                /* Search for distinctive patterns in the Jinja template */
                char *tmpl = malloc(vlen + 1); memcpy(tmpl, p, vlen); tmpl[vlen] = 0;
                if (strstr(tmpl, "im_start"))       ps->chat_style = 1; /* ChatML */
                else if (strstr(tmpl, "[INST]"))     ps->chat_style = 2; /* Llama/Mistral */
                else if (strstr(tmpl, "<|user|>"))   ps->chat_style = 3; /* Zephyr */
                else if (strstr(tmpl, "<|end|>"))    ps->chat_style = 4; /* Phi */
                else if (strstr(tmpl, "start_of_turn")) ps->chat_style = 5; /* Gemma */
                free(tmpl);
            }
            p += vlen;
        } else if (vtype == 4) { /* uint32 */
            PC(4); uint32_t val = *(uint32_t*)p; p += 4;
            if (strstr(key, "embedding_length")) ps->host_dim = (int)val;
            else if (strstr(key, "block_count")) ps->host_n_layers = (int)val;
            else if (strstr(key, "head_count") && !strstr(key, "kv")) ps->host_heads = (int)val;
            else if (strstr(key, "head_count_kv")) ps->host_kv_heads = (int)val;
            else if (strstr(key, "feed_forward_length")) ps->host_hidden = (int)val;
            else if (strstr(key, "vocab_size")) ps->host_vocab = (int)val;
            else if (strstr(key, "bos_token_id")) ps->bos_id = (int)val;
            else if (strstr(key, "eos_token_id")) ps->eos_id = (int)val;
            else if (strstr(key, "add_space_prefix")) ps->add_space_prefix = (int)val;
        } else if (vtype == 6) { /* float32 */
            PC(4); float fval; memcpy(&fval, p, 4); p += 4;
            if (strstr(key, "rope.freq_base")) ps->rope_theta = fval;
            else if (strstr(key, "layer_norm_rms_epsilon")) ps->rms_norm_eps = fval;
        } else if (vtype == 0 || vtype == 7) {
            PC(1); uint8_t bval = *p; p += 1;
            if (strstr(key, "add_space_prefix")) ps->add_space_prefix = bval;
        } else if (vtype == 1) p += 1;                            /* int8 */
        else if (vtype == 2 || vtype == 3) p += 2;             /* uint16, int16 */
        else if (vtype == 5) p += 4;                            /* int32 */
        else if (vtype == 10 || vtype == 11 || vtype == 12) p += 8; /* uint64, int64, float64 */
        else if (vtype == 9) { /* array */
            PC(4); uint32_t atype = *(uint32_t*)p; p += 4;
            PC(8); uint64_t alen = *(uint64_t*)p; p += 8;
            size_t elem_sz = 0;
            if (atype == 0 || atype == 1 || atype == 7) elem_sz = 1;
            else if (atype == 2 || atype == 3) elem_sz = 2;
            else if (atype == 4 || atype == 5 || atype == 6) {
                elem_sz = 4;
                /* float32 array: tokenizer.ggml.scores */
                if (atype == 6 && strstr(key, "tokenizer.ggml.scores") && alen < 200000) {
                    ps->vocab_scores = malloc(alen * sizeof(float));
                    memcpy(ps->vocab_scores, p, alen * 4);
                }
            }
            else if (atype == 10 || atype == 11 || atype == 12) elem_sz = 8;
            else if (atype == 8) {
                /* array of strings */
                int is_vocab = strstr(key, "tokenizer.ggml.tokens") != NULL;
                int is_merges = strstr(key, "tokenizer.ggml.merges") != NULL;
                if (is_vocab && alen < 200000) {
                    ps->vocab_tokens = calloc(alen, sizeof(char*));
                    ps->vocab_size = (int)alen;
                }
                if (is_merges && alen < 500000) {
                    ps->bpe_merges = calloc(alen, sizeof(char*));
                    ps->n_bpe_merges = (int)alen;
                }
                for (uint64_t ai = 0; ai < alen && p < pend; ai++) {
                    PC(8); uint64_t slen = *(uint64_t*)p; p += 8;
                    if (slen > 1000000 || p + slen > pend) break; /* sanity */
                    if (is_vocab && ps->vocab_tokens && ai < (uint64_t)ps->vocab_size) {
                        ps->vocab_tokens[ai] = malloc(slen + 1);
                        memcpy(ps->vocab_tokens[ai], p, slen);
                        ps->vocab_tokens[ai][slen] = '\0';
                    }
                    if (is_merges && ps->bpe_merges && ai < (uint64_t)ps->n_bpe_merges) {
                        ps->bpe_merges[ai] = malloc(slen + 1);
                        memcpy(ps->bpe_merges[ai], p, slen);
                        ps->bpe_merges[ai][slen] = '\0';
                    }
                    p += slen;
                }
                continue;
            }
            p += alen * elem_sz;
        } else { p += 4; } /* unknown — guess 4 bytes */
    }
    if (ps->host_dim == 0 || ps->host_n_layers == 0) goto bail;
    if (ps->host_heads == 0) ps->host_heads = ps->host_dim / 64;
    if (ps->host_kv_heads == 0) ps->host_kv_heads = ps->host_heads;
    ps->host_head_dim = ps->host_dim / ps->host_heads;
    if (ps->host_hidden == 0) ps->host_hidden = ps->host_dim * 4;

    /* Parse tensor info */
    if (n_tensors > 20000) goto bail;
    TensorInfo *tinfo = calloc(n_tensors, sizeof(TensorInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        PC(8); uint64_t nlen = *(uint64_t*)p; p += 8;
        if (nlen > 256) { free(tinfo); goto bail; }
        int nl = nlen < 95 ? (int)nlen : 95;
        PC(nlen); memcpy(tinfo[i].name, p, nl); tinfo[i].name[nl] = '\0'; p += nlen;
        PC(4); tinfo[i].ndim = *(uint32_t*)p; p += 4;
        if (tinfo[i].ndim > 4) { free(tinfo); goto bail; }
        for (uint32_t d = 0; d < tinfo[i].ndim; d++) { PC(8); tinfo[i].dims[d] = *(uint64_t*)p; p += 8; }
        PC(4); tinfo[i].dtype = *(uint32_t*)p; p += 4;
        PC(8); tinfo[i].offset = *(uint64_t*)p; p += 8;
    }

    uint64_t header_size = p - ps->mmap_base;
    uint64_t data_start = ((header_size + 31) / 32) * 32;

    /* dequantized f32 buffers — tracked in GGUFIndex for cleanup */
    ps->f16_bufs = NULL; ps->n_f16_bufs = 0;

    /* Wire weight pointers — supports f32, f16, Q4_0, Q8_0, Q4_K, Q6_K */
    int wired = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint32_t dt = tinfo[i].dtype;
        if (dt != 0 && dt != 1 && dt != 2 && dt != 6 && dt != 8 && dt != 12 && dt != 14) continue;
        uint64_t n_elems = 1;
        for (uint32_t d = 0; d < tinfo[i].ndim; d++) n_elems *= tinfo[i].dims[d];
        uint64_t raw_bytes = quant_raw_bytes(dt, n_elems);
        uint64_t byte_offset = data_start + tinfo[i].offset;
        if (raw_bytes == 0 || byte_offset + raw_bytes > ps->mmap_size) {
            if (raw_bytes > 0)
                printf("[doe] WARNING: tensor %s OOB (%lu+%lu > %lu), skipping\n",
                       tinfo[i].name, (unsigned long)byte_offset, (unsigned long)raw_bytes,
                       (unsigned long)ps->mmap_size);
            continue;
        }
        float *data;
        const uint8_t *src = ps->mmap_base + byte_offset;
        if (dt == 0) {
            data = (float*)src; /* f32: point directly into mmap */
        } else {
            /* dequantize to f32 */
            data = malloc(n_elems * sizeof(float));
            if (dt == 1) { /* f16 */
                const uint16_t *h = (const uint16_t*)src;
                for (uint64_t j = 0; j < n_elems; j++) data[j] = f16_to_f32(h[j]);
            } else if (dt == 2) dequant_q4_0(src, data, n_elems);
            else if (dt == 6) dequant_q5_0(src, data, n_elems);
            else if (dt == 8) dequant_q8_0(src, data, n_elems);
            else if (dt == 12) dequant_q4_k(src, data, n_elems);
            else if (dt == 14) dequant_q6_k(src, data, n_elems);
            ps->f16_bufs = realloc(ps->f16_bufs, (ps->n_f16_bufs+1)*sizeof(float*));
            ps->f16_bufs[ps->n_f16_bufs++] = data;
        }
        char *n = tinfo[i].name;
        /* debug: if (i < 15) printf("[tensor] %s dims=[%lu,%lu]\n", n, (unsigned long)tinfo[i].dims[0], (unsigned long)tinfo[i].dims[1]); */
        if (strcmp(n, "token_embd.weight") == 0) {
            ps->host_tok_emb = data;
            if (ps->host_vocab == 0) ps->host_vocab = (int)tinfo[i].dims[1];
            wired++;
        }
        else if (strcmp(n, "output_norm.weight") == 0) { ps->host_norm = data; wired++; }
        else if (strcmp(n, "output.weight") == 0) { ps->host_output = data; wired++; }
        else {
            int l = -1; sscanf(n, "blk.%d.", &l);
            if (l >= 0 && l < MAX_LAYERS && l < ps->host_n_layers) {
                if (strstr(n, "attn_q.weight")) { ps->host_layers[l].wq = data; wired++; }
                else if (strstr(n, "attn_k.weight")) { ps->host_layers[l].wk = data; wired++; }
                else if (strstr(n, "attn_v.weight")) { ps->host_layers[l].wv = data; wired++; }
                else if (strstr(n, "attn_output.weight")) { ps->host_layers[l].wo = data; wired++; }
                else if (strstr(n, "attn_q.bias")) { ps->host_layers[l].bq = data; wired++; }
                else if (strstr(n, "attn_k.bias")) { ps->host_layers[l].bk = data; wired++; }
                else if (strstr(n, "attn_v.bias")) { ps->host_layers[l].bv = data; wired++; }
                else if (strstr(n, "ffn_gate.weight") && !strstr(n, "ffn_gate_inp") && !strstr(n, "ffn_gate_up")) { ps->host_layers[l].ffn_gate = data; wired++; }
                else if (strstr(n, "ffn_up.weight") && !strstr(n, "gate_up")) {
                    /* Check if fused gate+up: dims[1] > host_hidden means [dim, hidden*2] */
                    if (ps->host_hidden > 0 && (int)tinfo[i].dims[1] > ps->host_hidden * 3 / 2) {
                        ps->host_layers[l].ffn_gate_up = data;
                    } else {
                        ps->host_layers[l].ffn_up = data;
                    }
                    wired++;
                }
                else if (strstr(n, "ffn_down.weight")) { ps->host_layers[l].ffn_down = data; wired++; }
                else if (strstr(n, "ffn_gate_up_proj") || strstr(n, "ffn_gate_up.weight")) { ps->host_layers[l].ffn_gate_up = data; wired++; }
                else if (strstr(n, "attn_norm.weight")) { ps->host_layers[l].attn_norm = data; wired++; }
                else if (strstr(n, "ffn_norm.weight")) { ps->host_layers[l].ffn_norm = data; wired++; }
                else if (l == 0 && strstr(n, "ffn")) { printf("[doe] unwired FFN tensor: %s\n", n); }
            }
        }
    }
    free(tinfo);

    /* tied embeddings: if output.weight missing, reuse token_embd.weight */
    if (!ps->host_output && ps->host_tok_emb) {
        ps->host_output = ps->host_tok_emb;
        printf("[doe] output.weight missing — using tied embeddings\n");
    }
    if (!ps->host_tok_emb || !ps->host_output || !ps->host_norm) {
        printf("[doe] host missing essential weights (tok_emb=%d out=%d norm=%d). abandoning.\n",
               ps->host_tok_emb!=NULL, ps->host_output!=NULL, ps->host_norm!=NULL);
        goto bail;
    }

    /* Check for standard FFN (skip MoE hosts for now) */
    int has_ffn = 0;
    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        if (ps->host_layers[l].ffn_gate && ps->host_layers[l].ffn_up && ps->host_layers[l].ffn_down) has_ffn = 1;
        if (ps->host_layers[l].ffn_gate_up && ps->host_layers[l].ffn_down) has_ffn = 1;
    }
    if (!has_ffn) {
        printf("[doe] host has no standard FFN. DOE needs a plain transformer.\n");
        goto bail;
    }

    /* ── Weight profiling — the sonar ── */
    printf("[sonar] profiling host weights...\n");
    ps->profile.n_layers = ps->host_n_layers;
    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        if (ps->host_layers[l].ffn_gate)
            profile_weights(ps->host_layers[l].ffn_gate, ps->host_hidden, ps->host_dim, &ps->profile.layers[l]);
        else
            memset(&ps->profile.layers[l], 0, sizeof(LayerProfile));
    }
    float total_h = 0;
    for (int l = 0; l < ps->profile.n_layers; l++) total_h += ps->profile.layers[l].health;
    ps->profile.overall_health = total_h / (ps->profile.n_layers > 0 ? ps->profile.n_layers : 1);
    ps->profile.complexity = (float)ps->host_dim * ps->host_n_layers * ps->host_heads;
    ps->profile.fingerprint = compute_fingerprint(&ps->profile);

    printf("[sonar] host fingerprint: %016llx health=%.2f complexity=%.0f\n",
           (unsigned long long)ps->profile.fingerprint, ps->profile.overall_health, ps->profile.complexity);
    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        LayerProfile *lp = &ps->profile.layers[l];
        if (lp->l2_norm > 0)
            printf("  L%d: health=%.2f l2=%.2f std=%.4f sparse=%.1f%% dead=%d\n",
                   l, lp->health, lp->l2_norm, lp->std_dev, lp->sparsity*100, lp->dead_neurons);
    }

    /* ── Initialize living LoRA experts per layer ── */
    int initial_experts = ps->host_n_layers <= 8 ? 4 : ps->host_n_layers <= 16 ? 6 : 8;
    ps->n_field_layers = ps->host_n_layers;
    if (ps->n_field_layers > MAX_LAYERS) ps->n_field_layers = MAX_LAYERS;

    for (int l = 0; l < ps->n_field_layers; l++) {
        FieldLayer *fl = &ps->field_layers[l];
        fl->host_layer_idx = l;
        fl->n_alive = initial_experts;
        fl->parliament.w_vote = calloc(MAX_EXPERTS * ps->host_dim, sizeof(float));
        float vote_std = 0.01f;
        for (int i = 0; i < MAX_EXPERTS * ps->host_dim; i++)
            fl->parliament.w_vote[i] = rand_normal() * vote_std;
        fl->parliament.consensus = 0.5f;
        /* Initialize experts with harmonic spacing — health-aware */
        float layer_health = ps->profile.layers[l].health;
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (e < initial_experts) {
                float freq = 6.2831853f * e / initial_experts;
                init_lora_expert(&fl->experts[e], ps->host_dim, ps->lora_rank, freq);
                /* Weaker layers get stronger initial LoRA — DOE compensates */
                if (layer_health < 0.5f) {
                    float boost = (0.5f - layer_health) * 2.0f;
                    for (int i = 0; i < ps->host_dim * ps->lora_rank; i++) {
                        fl->experts[e].lora_A[i] *= (1.0f + boost);
                        fl->experts[e].lora_B[i] *= (1.0f + boost);
                    }
                }
            } else {
                memset(&fl->experts[e], 0, sizeof(LoraExpert));
            }
        }
    }

    ps->active = 1;
    /* Build token hash table for O(1) lookup, then GPT-2 BPE scores */
    tok_ht_build(ps);
    build_gpt2_scores(ps);
    printf("[doe] attached to %s (arch=%s dim=%d layers=%d heads=%d kv=%d vocab=%d %.1fMB)\n",
           path, ps->host_arch, ps->host_dim, ps->host_n_layers, ps->host_heads,
           ps->host_kv_heads, ps->host_vocab, (float)ps->mmap_size/(1024*1024));
    printf("[doe] rope_theta=%.0f rms_eps=%.1e bias=%s\n",
           ps->rope_theta, ps->rms_norm_eps,
           ps->host_layers[0].bq ? "yes" : "no");
    if (ps->is_gpt2_bpe) printf("[doe] tokenizer: GPT-2 BPE (%d merges)\n", ps->n_bpe_merges);
    /* Auto-detect nanollama chat style from identity tag or vocab tokens */
    if (ps->chat_style == 0 && (ps->identity_tag[0] ||
        tok_lookup(ps, "<|user_start|>", 14) >= 0)) ps->chat_style = 6;
    { const char *cs[] = {"raw","chatml","inst","zephyr","phi","gemma","nanollama"};
      printf("[doe] chat: %s\n", cs[ps->chat_style < 7 ? ps->chat_style : 0]); }
    printf("[doe] LoRA rank=%d alpha=%.2f experts=%d/layer — parliament is alive.\n",
           ps->lora_rank, ps->lora_alpha, initial_experts);
    #undef PC
    return 1;
bail:
    for (int i = 0; i < ps->n_f16_bufs; i++) free(ps->f16_bufs[i]);
    free(ps->f16_bufs); ps->f16_bufs = NULL; ps->n_f16_bufs = 0;
    if (ps->mmap_base) { munmap(ps->mmap_base, ps->mmap_size); ps->mmap_base = NULL; }
    printf("[doe] GGUF parse failed.\n");
    return 0;
}

static void index_free(GGUFIndex *ps) {
    for (int l = 0; l < ps->n_field_layers; l++) {
        free(ps->field_layers[l].parliament.w_vote);
        for (int e = 0; e < MAX_EXPERTS; e++)
            if (ps->field_layers[l].experts[e].alive)
                free_lora_expert(&ps->field_layers[l].experts[e]);
    }
    for (int i = 0; i < ps->n_f16_bufs; i++) free(ps->f16_bufs[i]);
    free(ps->f16_bufs);
    if (ps->vocab_tokens) {
        for (int i = 0; i < ps->vocab_size; i++) free(ps->vocab_tokens[i]);
        free(ps->vocab_tokens);
    }
    free(ps->vocab_scores);
    if (ps->bpe_merges) {
        for (int i = 0; i < ps->n_bpe_merges; i++) free(ps->bpe_merges[i]);
        free(ps->bpe_merges);
    }
    if (ps->mmap_base) munmap(ps->mmap_base, ps->mmap_size);
    memset(ps, 0, sizeof(GGUFIndex));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PARLIAMENT ELECTION — variable-k over LoRA experts
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int parliament_elect(Parliament *p, LoraExpert *experts, float *input, int dim,
                            HarmonicState *hs, int *selected, float *weights) {
    int n_alive = 0, alive_idx[MAX_EXPERTS];
    for (int e = 0; e < MAX_EXPERTS; e++) if (experts[e].alive) alive_idx[n_alive++] = e;
    if (n_alive < MIN_EXPERTS) return 0;

    float votes[MAX_EXPERTS]; float max_vote = -1e30f;
    for (int i = 0; i < n_alive; i++) {
        int e = alive_idx[i];
        float *row = p->w_vote + e * dim;
        float dot = 0;
        for (int j = 0; j < dim; j++) dot += row[j] * input[j];
        float res = expert_resonance(experts[e].frequency, hs);
        votes[e] = dot + 0.1f * res;
        if (votes[e] > max_vote) max_vote = votes[e];
    }
    float mean_v = 0;
    for (int i = 0; i < n_alive; i++) mean_v += votes[alive_idx[i]];
    mean_v /= n_alive;
    float var_v = 0;
    for (int i = 0; i < n_alive; i++) { float d = votes[alive_idx[i]] - mean_v; var_v += d*d; }
    var_v /= n_alive;
    float consensus = fminf(1.0f, sqrtf(var_v + 1e-8f) / (fabsf(mean_v) + 1.0f));
    p->consensus = 0.9f * p->consensus + 0.1f * consensus;

    int k = (int)(n_alive * (1.0f - p->consensus));
    if (k < 2) k = 2; if (k > n_alive) k = n_alive;

    int used[MAX_EXPERTS] = {0};
    for (int ki = 0; ki < k; ki++) {
        float bv = -1e30f; int bi = 0;
        for (int i = 0; i < n_alive; i++) {
            int e = alive_idx[i];
            if (!used[e] && votes[e] > bv) { bv = votes[e]; bi = e; }
        }
        selected[ki] = bi; weights[ki] = votes[bi]; used[bi] = 1;
    }
    float mx = weights[0];
    for (int i = 1; i < k; i++) if (weights[i] > mx) mx = weights[i];
    float sum = 0;
    for (int i = 0; i < k; i++) { weights[i] = expf(weights[i]-mx); sum += weights[i]; }
    for (int i = 0; i < k; i++) weights[i] /= sum;
    p->election_count++;
    return k;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * NOTORCH — Hebbian plasticity for LoRA experts. from AML core.
 * no backprop. synapse strengthens from co-activation.
 * signal-gated: prophecy debt drives learning direction.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int notorch_offset = 0; /* rotating window into LoRA rank */

static void notorch_step(float *A, float *B, int out_dim, int in_dim, int rank,
                         const float *x, const float *dy, float signal) {
    if (fabsf(signal) < 1e-8f) return;
    float lr = F.notorch_lr * signal;
    /* NOTORCH operates at rank 4 but rotates across all LORA_RANK components.
     * each call updates 4 components starting at notorch_offset.
     * after rank/4 calls, every component has been updated once. */
    int nr = NOTORCH_RANK;
    if (nr > rank) nr = rank;
    int base = notorch_offset % rank;
    float u[NOTORCH_RANK];
    for (int j = 0; j < nr; j++) {
        int r = (base + j) % rank;
        float s = 0;
        for (int i = 0; i < out_dim && i < in_dim; i++) s += B[i * rank + r] * dy[i];
        u[j] = s + rand_normal() * 0.01f;
    }
#ifdef USE_BLAS
    for (int j = 0; j < nr; j++) {
        int r = (base + j) % rank;
        cblas_saxpy(in_dim, lr * u[j], x, 1, A + r, rank);
    }
#else
    for (int i = 0; i < in_dim; i++)
        for (int j = 0; j < nr; j++) {
            int r = (base + j) % rank;
            A[i * rank + r] += lr * x[i] * u[j];
        }
#endif
    /* decay only the components we touched */
    float decay = F.notorch_decay;
    for (int j = 0; j < nr; j++) {
        int r = (base + j) % rank;
        for (int i = 0; i < out_dim; i++) B[i * rank + r] *= decay;
    }
    notorch_offset = (base + nr) % rank;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * VITALITY + MITOSIS + APOPTOSIS — LoRA experts live and die
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void update_expert_vitality(FieldLayer *fl, int total_tokens) {
    int na = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (fl->experts[e].alive) na++;
    if (na == 0) return;
    float fair = (float)total_tokens / na;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!fl->experts[e].alive) continue;
        LoraExpert *exp = &fl->experts[e];
        float ratio = fair > 0 ? (float)exp->tokens_seen / fair : 1.0f;
        exp->vitality += (ratio - 1.0f) * 0.05f;
        if (exp->vitality < 0) exp->vitality = 0;
        if (exp->vitality > 1) exp->vitality = 1;
        exp->age++;
        if (exp->vitality < 0.1f) exp->low_vitality_count++;
        else exp->low_vitality_count = 0;
        exp->tokens_seen = 0;
    }
    fl->n_alive = na;
}

static int try_mitosis(FieldLayer *fl, int dim, int rank) {
    int na = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (fl->experts[e].alive) na++;
    if (na >= MAX_EXPERTS) return 0;
    int parent = -1;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!fl->experts[e].alive) continue;
        if (fl->experts[e].vitality > 0.8f && fl->experts[e].age > 20) { parent = e; break; }
    }
    if (parent < 0) return 0;
    int child = -1;
    for (int e = 0; e < MAX_EXPERTS; e++) if (!fl->experts[e].alive) { child = e; break; }
    if (child < 0) return 0;
    LoraExpert *p = &fl->experts[parent];
    float cf = p->frequency + 3.14159f / (na + 1);
    if (cf > 6.2831853f) cf -= 6.2831853f;
    init_lora_expert(&fl->experts[child], dim, rank, cf);
    LoraExpert *ch = &fl->experts[child];
    for (int i = 0; i < dim*rank; i++) ch->lora_A[i] = p->lora_A[i] + rand_normal()*0.01f;
    for (int i = 0; i < rank*dim; i++) ch->lora_B[i] = p->lora_B[i] + rand_normal()*0.01f;
    ch->vitality = 0.5f; p->vitality *= 0.8f;
    fl->n_alive++;
    return 1;
}

static int try_apoptosis(FieldLayer *fl) {
    int na = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (fl->experts[e].alive) na++;
    if (na <= MIN_EXPERTS) return 0;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!fl->experts[e].alive) continue;
        if (fl->experts[e].low_vitality_count >= 8) {
            free_lora_expert(&fl->experts[e]);
            fl->n_alive--;
            return 1;
        }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CALENDAR DRIFT — 12D temporal self-awareness. from DOE m.c.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float state[12]; int step;
} DriftSnapshot;

typedef struct {
    DriftSnapshot history[DRIFT_SNAPSHOTS];
    int head, n_snapshots;
    float drift, stability, drift_accel;
} CalendarDrift;

static void drift_init(CalendarDrift *cd) { memset(cd, 0, sizeof(CalendarDrift)); }

static void drift_snapshot(CalendarDrift *cd, float loss, GGUFIndex *ps, HarmonicState *hs) {
    DriftSnapshot *ds = &cd->history[cd->head % DRIFT_SNAPSHOTS];
    ds->step = F.step;
    int total_exp = 0;
    for (int l = 0; l < ps->n_field_layers; l++) total_exp += ps->field_layers[l].n_alive;
    ds->state[0] = (float)total_exp;
    ds->state[1] = ps->field_layers[0].parliament.consensus;
    ds->state[2] = loss;
    ds->state[3] = F.entropy;
    ds->state[4] = F.resonance;
    ds->state[5] = F.debt;
    ds->state[6] = hs->confidence;
    ds->state[7] = F.effective_temp;
    ds->state[8] = F.field_health;
    ds->state[9] = F.spring_energy;
    ds->state[10] = F.summer_energy;
    ds->state[11] = F.schumann_coherence;

    if (cd->n_snapshots > 0) {
        int prev = (cd->head - 1 + DRIFT_SNAPSHOTS) % DRIFT_SNAPSHOTS;
        float d2 = 0;
        for (int i = 0; i < 12; i++) {
            float diff = ds->state[i] - cd->history[prev].state[i];
            float range = fabsf(ds->state[i]) + 1e-8f;
            d2 += (diff / range) * (diff / range);
        }
        float new_drift = sqrtf(d2 / 12.0f);
        float prev_drift = cd->drift;
        cd->drift = 0.8f * cd->drift + 0.2f * new_drift;
        cd->drift_accel = cd->drift - prev_drift;
        cd->stability = 1.0f / (1.0f + cd->drift * 10.0f);
    }
    cd->head = (cd->head + 1) % DRIFT_SNAPSHOTS;
    if (cd->n_snapshots < DRIFT_SNAPSHOTS) cd->n_snapshots++;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * META-LEARNING — DOE learns from its own choices.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    int step; int n_experts; float consensus, loss, field_health;
    float prophecy_debt_avg; float drift; float delta_loss;
} MetaEntry;

typedef struct {
    MetaEntry history[META_HIST_CAP];
    int n_entries;
    float config_bias[4];
    float prediction_error;
} MetaTrack;

static void meta_init(MetaTrack *mt) {
    memset(mt, 0, sizeof(MetaTrack));
    for (int i = 0; i < 4; i++) mt->config_bias[i] = 0.5f;
}

static void meta_record(MetaTrack *mt, int step, int n_exp, float consensus,
                        float loss, float health, float debt_avg, float drift, float prev_loss) {
    if (mt->n_entries >= META_HIST_CAP) {
        memmove(mt->history, mt->history+1, (META_HIST_CAP-1)*sizeof(MetaEntry));
        mt->n_entries = META_HIST_CAP - 1;
    }
    MetaEntry *e = &mt->history[mt->n_entries];
    e->step = step; e->n_experts = n_exp; e->consensus = consensus;
    e->loss = loss; e->field_health = health; e->prophecy_debt_avg = debt_avg;
    e->drift = drift; e->delta_loss = prev_loss > 0 ? prev_loss - loss : 0;
    mt->n_entries++;
    if (mt->n_entries >= 2) {
        MetaEntry *prev = &mt->history[mt->n_entries-2];
        float improvement = prev->loss - loss;
        float lr_meta = 0.01f;
        float sig = improvement > 0 ? 1.0f : -0.5f;
        mt->config_bias[0] += lr_meta * sig * ((float)n_exp/MAX_EXPERTS - 0.5f);
        mt->config_bias[1] += lr_meta * sig * (consensus - 0.5f);
        mt->config_bias[2] += lr_meta * sig * (health - 0.5f);
        mt->config_bias[3] += lr_meta * sig * (debt_avg - 0.5f);
        for (int i = 0; i < 4; i++) {
            if (mt->config_bias[i] < 0.01f) mt->config_bias[i] = 0.01f;
            if (mt->config_bias[i] > 0.99f) mt->config_bias[i] = 0.99f;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MYCELIUM — LoRA spore forest.
 * DOE doesn't save full model GGUFs. it saves LoRA configurations:
 * the living experts, their weights, the parliament votes, the field state.
 * each spore is a snapshot of how DOE adapted to this host.
 * on restart with the same host (fingerprint match), load the best spore.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MYCELIUM_DIR "doe_mycelium"

typedef struct {
    char path[256];
    uint64_t host_fingerprint;
    float fitness;
    int step;
} LoraSpore;

typedef struct {
    LoraSpore spores[MYCELIUM_MAX];
    int n_spores, best_idx;
} MyceliumState;

static void mycelium_init(MyceliumState *ms) {
    memset(ms, 0, sizeof(MyceliumState));
    ms->best_idx = -1;
    mkdir(MYCELIUM_DIR, 0755);
}

static void mycelium_save(GGUFIndex *ps, int step, float fitness) {
    char path[256];
    snprintf(path, 256, "%s/spore_%016llx_s%d.bin", MYCELIUM_DIR,
             (unsigned long long)ps->profile.fingerprint, step);
    FILE *f = fopen(path, "wb");
    if (!f) { printf("[mycelium] cannot write %s\n", path); return; }
    /* header: fingerprint, step, fitness, n_layers, dim, rank */
    uint64_t fp = ps->profile.fingerprint;
    fwrite(&fp, 8, 1, f);
    fwrite(&step, 4, 1, f);
    fwrite(&fitness, 4, 1, f);
    int nl = ps->n_field_layers, dim = ps->host_dim, rank = ps->lora_rank;
    fwrite(&nl, 4, 1, f); fwrite(&dim, 4, 1, f); fwrite(&rank, 4, 1, f);
    /* per layer: n_alive, then per expert: alive, vitality, frequency, A, B */
    for (int l = 0; l < nl; l++) {
        FieldLayer *fl = &ps->field_layers[l];
        fwrite(&fl->n_alive, 4, 1, f);
        /* parliament vote weights */
        fwrite(fl->parliament.w_vote, sizeof(float), MAX_EXPERTS * dim, f);
        fwrite(&fl->parliament.consensus, 4, 1, f);
        for (int e = 0; e < MAX_EXPERTS; e++) {
            LoraExpert *ex = &fl->experts[e];
            fwrite(&ex->alive, 4, 1, f);
            if (ex->alive) {
                fwrite(&ex->vitality, 4, 1, f);
                fwrite(&ex->frequency, 4, 1, f);
                fwrite(ex->lora_A, sizeof(float), dim * rank, f);
                fwrite(ex->lora_B, sizeof(float), rank * dim, f);
            }
        }
    }
    fclose(f);
    printf("[mycelium] spore saved: %s (fitness=%.3f)\n", path, fitness);
}

static int mycelium_load(GGUFIndex *ps, uint64_t target_fp) {
    /* scan directory for best matching spore */
    char pattern[256];
    snprintf(pattern, 256, "%s/spore_%016llx_*.bin", MYCELIUM_DIR, (unsigned long long)target_fp);
    /* simple scan: find newest (highest step) spore for this fingerprint */
    char best_path[256] = {0};
    int best_step = -1;
    FILE *p = popen("ls " MYCELIUM_DIR "/ 2>/dev/null", "r");
    if (!p) return 0;
    char line[256];
    while (fgets(line, sizeof(line), p)) {
        int len = strlen(line);
        while (len > 0 && (line[len-1]=='\n'||line[len-1]=='\r')) line[--len] = '\0';
        /* match fingerprint */
        char want[32]; snprintf(want, 32, "spore_%016llx", (unsigned long long)target_fp);
        if (!strstr(line, want)) continue;
        /* extract step from filename */
        char *sp = strstr(line, "_s");
        if (!sp) continue;
        int s = atoi(sp+2);
        if (s > best_step) {
            best_step = s;
            snprintf(best_path, 256, "%s/%s", MYCELIUM_DIR, line);
        }
    }
    pclose(p);
    if (best_step < 0) return 0;

    FILE *f = fopen(best_path, "rb");
    if (!f) return 0;
    uint64_t fp; fread(&fp, 8, 1, f);
    if (fp != target_fp) { fclose(f); return 0; }
    int step; float fitness;
    fread(&step, 4, 1, f); fread(&fitness, 4, 1, f);
    int nl, dim, rank;
    fread(&nl, 4, 1, f); fread(&dim, 4, 1, f); fread(&rank, 4, 1, f);
    if (nl != ps->n_field_layers || dim != ps->host_dim || rank != ps->lora_rank) {
        printf("[mycelium] spore mismatch (layers=%d/%d dim=%d/%d rank=%d/%d)\n",
               nl, ps->n_field_layers, dim, ps->host_dim, rank, ps->lora_rank);
        fclose(f); return 0;
    }
    for (int l = 0; l < nl; l++) {
        FieldLayer *fl = &ps->field_layers[l];
        fread(&fl->n_alive, 4, 1, f);
        fread(fl->parliament.w_vote, sizeof(float), MAX_EXPERTS * dim, f);
        fread(&fl->parliament.consensus, 4, 1, f);
        for (int e = 0; e < MAX_EXPERTS; e++) {
            LoraExpert *ex = &fl->experts[e];
            int alive; fread(&alive, 4, 1, f);
            if (alive) {
                if (!ex->alive) {
                    ex->lora_A = calloc(dim * rank, sizeof(float));
                    ex->lora_B = calloc(rank * dim, sizeof(float));
                }
                ex->alive = 1;
                fread(&ex->vitality, 4, 1, f);
                fread(&ex->frequency, 4, 1, f);
                fread(ex->lora_A, sizeof(float), dim * rank, f);
                fread(ex->lora_B, sizeof(float), rank * dim, f);
            } else if (ex->alive) {
                free_lora_expert(ex);
            }
        }
    }
    fclose(f);
    printf("[mycelium] spore loaded: %s (step=%d fitness=%.3f)\n", best_path, step, fitness);
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INDEX FORWARD — run token through host with DOE modulation.
 *
 * per layer:
 *   1. host attention (read-only weights, KV cache)
 *   2. parliament election (which LoRA experts vote)
 *   3. Delta Voice injection: x += Σ(w_k × α × A_k @ (B_k @ x))
 *   4. host FFN (read-only)
 *   5. layer_focus scaling on residual
 *
 * after all layers:
 *   6. field modulation on logits
 *   7. prophecy debt computation
 *   8. NOTORCH Hebbian update on winning experts
 *
 * the host swims. the field steers. nobody knows who's in charge.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float *x, *xb, *xb2, *q, *k, *v, *att, *logits;
    float *hb, *hb2, *expert_out;
    float *key_cache, *value_cache;
    float *cos_cache, *sin_cache;
    HarmonicState hs;
    int max_seq;
} InferState;

static InferState alloc_infer(GGUFIndex *ps, int max_seq) {
    InferState s = {0};
    int D = ps->host_dim, kd = ps->host_kv_heads * ps->host_head_dim;
    int H = ps->host_hidden;
    s.max_seq = max_seq;
    s.x = calloc(D, 4); s.xb = calloc(D, 4); s.xb2 = calloc(D, 4);
    s.q = calloc(ps->host_heads * ps->host_head_dim, 4);
    s.k = calloc(kd, 4); s.v = calloc(kd, 4);
    s.att = calloc(ps->host_heads * max_seq, 4);
    s.logits = calloc(ps->host_vocab, 4);
    s.hb = calloc(H, 4); s.hb2 = calloc(H * 2, 4); /* *2 for fused gate_up */
    s.expert_out = calloc(D, 4);
    s.key_cache = calloc(ps->host_n_layers * max_seq * kd, 4);
    s.value_cache = calloc(ps->host_n_layers * max_seq * kd, 4);
    int half = ps->host_head_dim / 2;
    s.cos_cache = calloc(max_seq * half, 4);
    s.sin_cache = calloc(max_seq * half, 4);
    float rope_theta = ps->rope_theta;
    for (int p = 0; p < max_seq; p++)
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(rope_theta, (float)(2*i) / (float)ps->host_head_dim);
            float ang = (float)p * freq;
            s.cos_cache[p*half+i] = cosf(ang);
            s.sin_cache[p*half+i] = sinf(ang);
        }
    return s;
}

static void free_infer(InferState *s) {
    free(s->x); free(s->xb); free(s->xb2);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->logits);
    free(s->hb); free(s->hb2); free(s->expert_out);
    free(s->key_cache); free(s->value_cache);
    free(s->cos_cache); free(s->sin_cache);
    memset(s, 0, sizeof(InferState));
}

static float *doe_forward(GGUFIndex *ps, InferState *s, int token, int pos) {
    int D = ps->host_dim, hd = ps->host_head_dim;
    int kd = ps->host_kv_heads * hd;
    int H = ps->host_hidden;
    int hg = ps->host_heads / ps->host_kv_heads;
    float sc = 1.0f / sqrtf((float)hd);

    /* Embedding */
    if (token < ps->host_vocab)
        memcpy(s->x, ps->host_tok_emb + token * D, D * sizeof(float));
    else
        memset(s->x, 0, D * sizeof(float));

    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        if (!ps->host_layers[l].wq) continue;

        /* ── Host attention ── */
        float *xn = s->xb;
        if (ps->host_layers[l].attn_norm) rmsnorm(xn, s->x, ps->host_layers[l].attn_norm, D, ps->rms_norm_eps);
        else memcpy(xn, s->x, D*4);

        matvec(s->q, ps->host_layers[l].wq, xn, ps->host_heads*hd, D);
        matvec(s->k, ps->host_layers[l].wk, xn, kd, D);
        matvec(s->v, ps->host_layers[l].wv, xn, kd, D);

        /* Add attention biases (Qwen2, optional) */
        if (ps->host_layers[l].bq) for (int i = 0; i < ps->host_heads*hd; i++) s->q[i] += ps->host_layers[l].bq[i];
        if (ps->host_layers[l].bk) for (int i = 0; i < kd; i++) s->k[i] += ps->host_layers[l].bk[i];
        if (ps->host_layers[l].bv) for (int i = 0; i < kd; i++) s->v[i] += ps->host_layers[l].bv[i];

        for (int h = 0; h < ps->host_heads; h++) apply_rope(s->q+h*hd, pos, s->cos_cache, s->sin_cache, hd);
        for (int h = 0; h < ps->host_kv_heads; h++) apply_rope(s->k+h*hd, pos, s->cos_cache, s->sin_cache, hd);

        int co = l * s->max_seq * kd + pos * kd;
        memcpy(s->key_cache + co, s->k, kd*4);
        memcpy(s->value_cache + co, s->v, kd*4);

        float *ao = s->xb2; memset(ao, 0, D*4);
        for (int h = 0; h < ps->host_heads; h++) {
            int kvh = h / hg; float *qh = s->q + h*hd;
            float *att = s->att + h * s->max_seq;
            for (int t = 0; t <= pos; t++) {
                int ko = l*s->max_seq*kd + t*kd + kvh*hd;
                float dot = 0;
                for (int d = 0; d < hd; d++) dot += qh[d] * s->key_cache[ko+d];
                att[t] = dot * sc;
            }
            softmax_n(att, pos+1);
            float *oh = ao + h*hd;
            for (int t = 0; t <= pos; t++) {
                float a = att[t]; int vo = l*s->max_seq*kd + t*kd + kvh*hd;
                for (int d = 0; d < hd; d++) oh[d] += a * s->value_cache[vo+d];
            }
        }
        matvec(s->xb, ps->host_layers[l].wo, ao, D, D);
        for (int i = 0; i < D; i++) s->x[i] += s->xb[i];

        /* ── Parliament election + LoRA injection (after attention, before FFN) ── */
        if (l < ps->n_field_layers) {
            FieldLayer *fl = &ps->field_layers[l];
            int selected[MAX_EXPERTS]; float weights[MAX_EXPERTS];
            int k = parliament_elect(&fl->parliament, fl->experts, s->x, D, &s->hs, selected, weights);
            memset(s->expert_out, 0, D*4);
            for (int ki = 0; ki < k; ki++) {
                LoraExpert *exp = &fl->experts[selected[ki]];
                exp->tokens_seen++;
                /* Delta Voice: out += α × A @ (B @ x) */
                float tmp[LORA_RANK]; memset(tmp, 0, sizeof(tmp));
                for (int r = 0; r < ps->lora_rank; r++)
                    for (int j = 0; j < D; j++)
                        tmp[r] += exp->lora_B[r * D + j] * s->x[j];
                float lora_out[D]; memset(lora_out, 0, D*4);
                for (int i = 0; i < D; i++)
                    for (int r = 0; r < ps->lora_rank; r++)
                        lora_out[i] += exp->lora_A[i * ps->lora_rank + r] * tmp[r];
                for (int i = 0; i < D; i++)
                    s->expert_out[i] += weights[ki] * ps->lora_alpha * lora_out[i];
            }
            for (int i = 0; i < D; i++) s->x[i] += s->expert_out[i];
        }

        /* ── Host FFN (SwiGLU) ── */
        {
            float *fn = s->xb;
            if (ps->host_layers[l].ffn_norm) rmsnorm(fn, s->x, ps->host_layers[l].ffn_norm, D, ps->rms_norm_eps);
            else memcpy(fn, s->x, D*4);

            if (ps->host_layers[l].ffn_gate_up && ps->host_layers[l].ffn_down) {
                /* Fused gate_up: [hidden*2, dim] → split into gate [0..H) and up [H..2H) */
                matvec(s->hb2, ps->host_layers[l].ffn_gate_up, fn, H * 2, D);
                for (int i = 0; i < H; i++) s->hb[i] = silu_f(s->hb2[i]) * s->hb2[H + i];
                matvec(s->xb, ps->host_layers[l].ffn_down, s->hb, D, H);
                for (int i = 0; i < D; i++) s->x[i] += s->xb[i];
            } else if (ps->host_layers[l].ffn_gate && ps->host_layers[l].ffn_up && ps->host_layers[l].ffn_down) {
                /* Standard separate gate + up */
                matvec(s->hb, ps->host_layers[l].ffn_gate, fn, H, D);
                matvec(s->hb2, ps->host_layers[l].ffn_up, fn, H, D);
                for (int i = 0; i < H; i++) s->hb[i] = silu_f(s->hb[i]) * s->hb2[i];
                matvec(s->xb, ps->host_layers[l].ffn_down, s->hb, D, H);
                for (int i = 0; i < D; i++) s->x[i] += s->xb[i];
            }
        }
    }

    /* Final norm + LM head */
    rmsnorm(s->x, s->x, ps->host_norm, D, ps->rms_norm_eps);
    matvec(s->logits, ps->host_output, s->x, ps->host_vocab, D);

    return s->logits;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SAMPLING + CHAT
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int sample(float *logits, int V, float temp, int top_k) {
    if (temp <= 0) { int b = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[b]) b = i; return b; }
    for (int i = 0; i < V; i++) logits[i] /= temp;
    if (top_k > 0 && top_k < V) {
        float *s = malloc(V*4); memcpy(s, logits, V*4);
        for (int i = 0; i < top_k; i++) { int b = i; for (int j = i+1; j < V; j++) if (s[j] > s[b]) b = j; float t = s[i]; s[i] = s[b]; s[b] = t; }
        float th = s[top_k-1]; free(s);
        for (int i = 0; i < V; i++) if (logits[i] < th) logits[i] = -1e30f;
    }
    softmax_n(logits, V);
    float r = rand_uniform(), cum = 0;
    for (int i = 0; i < V; i++) { cum += logits[i]; if (cum >= r) return i; }
    return V - 1;
}

/* GPT-2 byte_decoder: reverse the byte_encoder mapping (unicode codepoint -> original byte) */
static int gpt2_rune_to_byte(int rune) {
    static int table_built = 0;
    static int rtable[512]; /* rune -> byte, -1 if not mapped */
    if (!table_built) {
        for (int i = 0; i < 512; i++) rtable[i] = -1;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
                rtable[b] = b; /* identity mapping */
            else
                rtable[256 + n++] = b; /* offset mapping */
        }
        table_built = 1;
    }
    if (rune >= 0 && rune < 512) return rtable[rune];
    return -1;
}

/* Parse one UTF-8 codepoint, return codepoint and advance *p by bytes consumed */
static int utf8_decode_cp(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    int cp, len;
    if (s[0] < 0x80) { cp = s[0]; len = 1; }
    else if ((s[0] & 0xE0) == 0xC0) { cp = (s[0] & 0x1F) << 6 | (s[1] & 0x3F); len = 2; }
    else if ((s[0] & 0xF0) == 0xE0) { cp = (s[0] & 0x0F) << 12 | (s[1] & 0x3F) << 6 | (s[2] & 0x3F); len = 3; }
    else if ((s[0] & 0xF8) == 0xF0) { cp = (s[0] & 0x07) << 18 | (s[1] & 0x3F) << 12 | (s[2] & 0x3F) << 6 | (s[3] & 0x3F); len = 4; }
    else { cp = s[0]; len = 1; } /* fallback */
    *p += len;
    return cp;
}

/* Decode token to text using GGUF vocab, fallback to byte */
static void token_decode_print(GGUFIndex *ps, int token) {
    if (ps->vocab_tokens && token >= 0 && token < ps->vocab_size && ps->vocab_tokens[token]) {
        const char *s = ps->vocab_tokens[token];
        if (ps->is_gpt2_bpe) {
            /* GPT-2 byte-level BPE: full byte_decoder — each codepoint maps to one byte */
            unsigned char buf[256];
            int blen = 0;
            const char *p = s;
            while (*p && blen < (int)sizeof(buf) - 4) {
                int cp = utf8_decode_cp(&p);
                int b = gpt2_rune_to_byte(cp);
                if (b >= 0) buf[blen++] = (unsigned char)b;
                else {
                    /* Not a GPT-2 mapped byte — emit codepoint as UTF-8 */
                    if (cp < 0x80) { buf[blen++] = cp; }
                    else if (cp < 0x800) { buf[blen++] = 0xC0|(cp>>6); buf[blen++] = 0x80|(cp&0x3F); }
                    else if (cp < 0x10000) { buf[blen++] = 0xE0|(cp>>12); buf[blen++] = 0x80|((cp>>6)&0x3F); buf[blen++] = 0x80|(cp&0x3F); }
                    else { buf[blen++] = 0xF0|(cp>>18); buf[blen++] = 0x80|((cp>>12)&0x3F); buf[blen++] = 0x80|((cp>>6)&0x3F); buf[blen++] = 0x80|(cp&0x3F); }
                }
            }
            fwrite(buf, 1, blen, stdout);
        } else {
        /* Handle sentencepiece ▁ (U+2581, 3 bytes: E2 96 81) → space */
        while (*s) {
            if ((unsigned char)s[0] == 0xE2 && (unsigned char)s[1] == 0x96 && (unsigned char)s[2] == 0x81) {
                fputc(' ', stdout);
                s += 3;
            } else if (!strncmp(s, "<0x", 3) && s[5] == '>') {
                /* sentencepiece hex byte: <0xAB> */
                unsigned int b = 0;
                sscanf(s + 3, "%02X", &b);
                if (b >= 32 || b == '\n' || b == '\t') fputc((char)b, stdout);
                s += 6;
            } else {
                fputc(*s, stdout);
                s++;
            }
        }
        }
    } else if (token >= 0 && token < 256) {
        char c = (char)token;
        if (c >= 32 || c == '\n' || c == '\t') fputc(c, stdout);
    }
}

/* Decode token to buffer instead of stdout — for HTTP serve mode */
static int token_decode_buf(GGUFIndex *ps, int token, char *buf, int bufsz) {
    int pos = 0;
    if (ps->vocab_tokens && token >= 0 && token < ps->vocab_size && ps->vocab_tokens[token]) {
        const char *s = ps->vocab_tokens[token];
        if (ps->is_gpt2_bpe) {
            while (*s && pos < bufsz - 1) {
                if ((unsigned char)s[0]==0xC4 && (unsigned char)s[1]==0xA0) { buf[pos++]=' '; s+=2; }
                else if ((unsigned char)s[0]==0xC4 && (unsigned char)s[1]==0x8A) { buf[pos++]='\n'; s+=2; }
                else { buf[pos++]=*s; s++; }
            }
        } else {
            while (*s && pos < bufsz - 1) {
                if ((unsigned char)s[0]==0xE2 && (unsigned char)s[1]==0x96 && (unsigned char)s[2]==0x81) {
                    buf[pos++]=' '; s+=3;
                } else if (!strncmp(s,"<0x",3) && s[5]=='>') {
                    unsigned int b=0; sscanf(s+3,"%02X",&b);
                    if (b>=32||b=='\n'||b=='\t') buf[pos++]=(char)b;
                    s+=6;
                } else { buf[pos++]=*s; s++; }
            }
        }
    } else if (token >= 0 && token < 256) {
        char c = (char)token;
        if ((c>=32||c=='\n'||c=='\t') && pos < bufsz-1) buf[pos++]=c;
    }
    buf[pos] = '\0';
    return pos;
}

/* ── BPE Tokenizer — SentencePiece style, score-based merge ── */

static int tok_lookup(GGUFIndex *ps, const char *s, int len); /* forward decl */

/* Build GPT-2 BPE scores from merges (called after index_load if needed) */
static void build_gpt2_scores(GGUFIndex *ps) {
    if (!ps->is_gpt2_bpe || !ps->bpe_merges || ps->n_bpe_merges == 0 || ps->vocab_scores || !ps->vocab_tokens) return;
    ps->vocab_scores = calloc(ps->vocab_size, sizeof(float));
    for (int i = 0; i < ps->vocab_size; i++) ps->vocab_scores[i] = -1e9f;
    int built = 0;
    for (int m = 0; m < ps->n_bpe_merges; m++) {
        const char *merge = ps->bpe_merges[m];
        const char *sp = strchr(merge, ' ');
        if (!sp) continue;
        int la = (int)(sp - merge), lb = (int)strlen(sp + 1);
        if (la + lb > 128) continue;
        char merged[130];
        memcpy(merged, merge, la);
        memcpy(merged + la, sp + 1, lb);
        int mid = tok_lookup(ps, merged, la + lb);
        if (mid >= 0) { ps->vocab_scores[mid] = (float)(ps->n_bpe_merges - m); built++; }
    }
    printf("[doe] GPT-2 BPE: built %d merge scores from %d merges\n", built, ps->n_bpe_merges);
    ps->add_space_prefix = 0;
}

/* FNV-1a hash */
static uint32_t tok_hash(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}

/* Build hash table for O(1) token lookup */
static void tok_ht_build(GGUFIndex *ps) {
    if (!ps->vocab_tokens || ps->vocab_size == 0) return;
    int cap = 1;
    while (cap < ps->vocab_size * 3) cap <<= 1; /* ~33% load factor */
    ps->tok_ht_ids = malloc(cap * sizeof(int));
    ps->tok_ht_cap = cap;
    for (int i = 0; i < cap; i++) ps->tok_ht_ids[i] = -1;
    int mask = cap - 1;
    for (int i = 0; i < ps->vocab_size; i++) {
        if (!ps->vocab_tokens[i]) continue;
        int slen = (int)strlen(ps->vocab_tokens[i]);
        uint32_t idx = tok_hash(ps->vocab_tokens[i], slen) & mask;
        while (ps->tok_ht_ids[idx] != -1) idx = (idx + 1) & mask;
        ps->tok_ht_ids[idx] = i;
    }
}

/* Find token ID by string. Returns -1 if not found. O(1) average. */
static int tok_lookup(GGUFIndex *ps, const char *s, int len) {
    if (!ps->tok_ht_ids) {
        /* fallback linear scan */
        for (int i = 0; i < ps->vocab_size; i++) {
            if (ps->vocab_tokens[i] && strlen(ps->vocab_tokens[i]) == (size_t)len
                && memcmp(ps->vocab_tokens[i], s, len) == 0)
                return i;
        }
        return -1;
    }
    int mask = ps->tok_ht_cap - 1;
    uint32_t idx = tok_hash(s, len) & mask;
    while (ps->tok_ht_ids[idx] != -1) {
        int id = ps->tok_ht_ids[idx];
        const char *t = ps->vocab_tokens[id];
        if (t && strlen(t) == (size_t)len && memcmp(t, s, len) == 0) return id;
        idx = (idx + 1) & mask;
    }
    return -1;
}

/* Score-based BPE merge on an array of token IDs */
static int bpe_merge(GGUFIndex *ps, int *ids, int n) {
    if (!ps->vocab_scores) return n;
    while (n > 1) {
        float best_score = -1e30f;
        int best_idx = -1, best_id = -1;
        for (int i = 0; i < n - 1; i++) {
            /* Concatenate token strings */
            const char *a = ps->vocab_tokens[ids[i]];
            const char *b = ps->vocab_tokens[ids[i+1]];
            if (!a || !b) continue;
            int la = strlen(a), lb = strlen(b);
            if (la + lb > 128) continue;
            char merged[130];
            memcpy(merged, a, la);
            memcpy(merged + la, b, lb);
            int mid = tok_lookup(ps, merged, la + lb);
            if (mid >= 0 && ps->vocab_scores[mid] > best_score) {
                best_score = ps->vocab_scores[mid];
                best_idx = i;
                best_id = mid;
            }
        }
        if (best_idx < 0) break;
        ids[best_idx] = best_id;
        /* Remove ids[best_idx+1] by shifting */
        for (int i = best_idx + 1; i < n - 1; i++) ids[i] = ids[i+1];
        n--;
    }
    return n;
}

/* GPT-2 byte-to-unicode table: maps each byte to a unicode codepoint */
static int gpt2_byte_to_rune(int b) {
    /* Printable ASCII + Latin-1 supplement range → identity */
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
        return b;
    /* Everything else → 256 + offset */
    static int table_built = 0;
    static int table[256];
    if (!table_built) {
        int n = 0;
        for (int i = 0; i < 256; i++) {
            if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))
                table[i] = i;
            else
                table[i] = 256 + n++;
        }
        table_built = 1;
    }
    return table[(unsigned char)b];
}

/* Encode a unicode codepoint as UTF-8, return length */
static int rune_to_utf8(int r, char *out) {
    if (r < 0x80) { out[0] = (char)r; return 1; }
    if (r < 0x800) { out[0] = 0xC0 | (r >> 6); out[1] = 0x80 | (r & 0x3F); return 2; }
    out[0] = 0xE0 | (r >> 12); out[1] = 0x80 | ((r >> 6) & 0x3F); out[2] = 0x80 | (r & 0x3F); return 3;
}

/* Try to match a special token at position i in text. Returns token id and advances *len. */
static int try_special_token(GGUFIndex *ps, const char *text, int tlen, int i, int *consumed) {
    static const char *specials[] = {
        "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|end|>",
        "<start_of_turn>", "<end_of_turn>", "<|user|>", "<|assistant|>",
        "[INST]", "[/INST]", "<s>", "</s>",
        "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
        "<|bos|>", "<|eot_id|>", NULL
    };
    if (text[i] != '<' && text[i] != '[') return -1;
    for (int s = 0; specials[s]; s++) {
        int slen = (int)strlen(specials[s]);
        if (i + slen <= tlen && memcmp(text + i, specials[s], slen) == 0) {
            int id = tok_lookup(ps, specials[s], slen);
            if (id >= 0) { *consumed = slen; return id; }
        }
    }
    return -1;
}

static int tokenize_input(GGUFIndex *ps, const char *text, int *tokens, int max_tokens) {
    if (!ps->vocab_tokens) {
        int n = 0, len = strlen(text);
        for (int i = 0; i < len && n < max_tokens; i++) tokens[n++] = (unsigned char)text[i];
        return n;
    }

    int tlen = strlen(text);
    int *ids = malloc((tlen + 16) * sizeof(int));
    int n = 0;

    if (ps->is_gpt2_bpe) {
        /* GPT-2: check special tokens first, then byte-level BPE */
        for (int i = 0; i < tlen && n < max_tokens; ) {
            int consumed = 0;
            int sid = try_special_token(ps, text, tlen, i, &consumed);
            if (sid >= 0) { ids[n++] = sid; i += consumed; continue; }
            int r = gpt2_byte_to_rune((unsigned char)text[i]);
            char u8[4]; int u8len = rune_to_utf8(r, u8);
            int id = tok_lookup(ps, u8, u8len);
            ids[n++] = (id >= 0) ? id : 0;
            i++;
        }
    } else {
        /* SentencePiece: split on special tokens first, then ▁-encode segments */
        int i = 0;
        while (i < tlen && n < max_tokens) {
            /* Check special tokens at raw text level */
            int consumed = 0;
            int sid = try_special_token(ps, text, tlen, i, &consumed);
            if (sid >= 0) { ids[n++] = sid; i += consumed; continue; }

            /* Find next special token boundary (or end) */
            int seg_end = i + 1;
            while (seg_end < tlen) {
                int c2 = 0;
                if (try_special_token(ps, text, tlen, seg_end, &c2) >= 0) break;
                seg_end++;
            }

            /* Encode segment [i, seg_end) with SentencePiece ▁ */
            int slen = seg_end - i;
            char *sp = malloc(slen * 3 + 4);
            int sp_len = 0;
            if (ps->add_space_prefix && i == 0 && text[i] != ' ') {
                sp[sp_len++] = 0xE2; sp[sp_len++] = 0x96; sp[sp_len++] = 0x81;
            }
            for (int j = i; j < seg_end; j++) {
                if (text[j] == ' ') {
                    sp[sp_len++] = 0xE2; sp[sp_len++] = 0x96; sp[sp_len++] = 0x81;
                } else {
                    sp[sp_len++] = text[j];
                }
            }
            sp[sp_len] = '\0';
            int k = 0;
            while (k < sp_len && n < max_tokens) {
                int clen = 1;
                unsigned char c = (unsigned char)sp[k];
                if (c >= 0xC0 && c < 0xE0) clen = 2;
                else if (c >= 0xE0 && c < 0xF0) clen = 3;
                else if (c >= 0xF0) clen = 4;
                if (k + clen > sp_len) clen = 1;
                int id = tok_lookup(ps, sp + k, clen);
                if (id >= 0) { ids[n++] = id; k += clen; }
                else {
                    char hex[7]; snprintf(hex, 7, "<0x%02X>", (unsigned char)sp[k]);
                    id = tok_lookup(ps, hex, 6);
                    ids[n++] = (id >= 0) ? id : 0; k++;
                }
            }
            free(sp);
            i = seg_end;
        }
    }

    n = bpe_merge(ps, ids, n);
    int out = (n < max_tokens) ? n : max_tokens;
    memcpy(tokens, ids, out * sizeof(int));
    free(ids);
    return out;
}

static void chat(GGUFIndex *ps) {
    int max_seq = 512;
    InferState is = alloc_infer(ps, max_seq);
    CalendarDrift cd; drift_init(&cd);
    MetaTrack meta; meta_init(&meta);
    HarmonicState hs = {0};

    char input[1024];
    printf("\n[doe] the parliament is in session. type your message (Ctrl+C to dissipate):\n");
    printf("[doe] host: %s (%s, %dM params)\n\n",
           ps->host_path, ps->host_arch,
           (int)(ps->host_vocab * ps->host_dim * 2 / 1000000)); /* rough estimate */

    float debt_sum = 0; int debt_count = 0;

    while (1) {
        printf("> "); fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        int len = strlen(input);
        while (len > 0 && (input[len-1]=='\n' || input[len-1]=='\r')) input[--len] = '\0';
        if (!len) continue;
        if (strcmp(input,"quit")==0 || strcmp(input,"exit")==0) break;
        if (strcmp(input,"status")==0) {
            printf("[field] step=%d debt=%.3f entropy=%.3f resonance=%.3f emergence=%.3f\n",
                   F.step, F.debt, F.entropy, F.resonance, F.emergence);
            printf("[field] season=%s health=%.3f temp=%.3f velocity=%s\n",
                   (const char*[]){"spring","summer","autumn","winter"}[F.season],
                   F.field_health, F.effective_temp,
                   (const char*[]){"nomove","walk","run","backward"}[F.velocity_mode]);
            printf("[drift] d=%.3f stability=%.3f accel=%.4f snapshots=%d\n",
                   cd.drift, cd.stability, cd.drift_accel, cd.n_snapshots);
            int te = 0;
            for (int l = 0; l < ps->n_field_layers; l++) te += ps->field_layers[l].n_alive;
            printf("[experts] alive=%d consensus=%.2f elections=%d\n",
                   te, ps->field_layers[0].parliament.consensus,
                   ps->field_layers[0].parliament.election_count);
            if (debt_count > 0)
                printf("[prophecy] avg_debt=%.4f total_debt=%.4f\n", debt_sum/debt_count, F.debt);
            continue;
        }

        /* Reset KV cache */
        int kd = ps->host_kv_heads * ps->host_head_dim;
        memset(is.key_cache, 0, ps->host_n_layers * max_seq * kd * 4);
        memset(is.value_cache, 0, ps->host_n_layers * max_seq * kd * 4);

        /* Wrap input in chat template (auto-detected from GGUF chat_template) */
        char wrapped[2048];
        /* Only use chat template if the key special tokens exist in vocab */
        int use_template = 0;
        switch (ps->chat_style) {
        case 1: /* ChatML */
            if (tok_lookup(ps, "<|im_start|>", 12) >= 0) {
                snprintf(wrapped, sizeof(wrapped),
                    "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", input);
                use_template = 1;
            }
            break;
        case 2: /* [INST] */
            if (tok_lookup(ps, "[INST]", 6) >= 0) {
                snprintf(wrapped, sizeof(wrapped), "[INST] %s [/INST]", input);
                use_template = 1;
            }
            break;
        case 3: /* Zephyr */
            if (tok_lookup(ps, "<|user|>", 8) >= 0) {
                snprintf(wrapped, sizeof(wrapped),
                    "<|user|>\n%s\n<|assistant|>\n", input);
                use_template = 1;
            }
            break;
        case 4: /* Phi */
            if (tok_lookup(ps, "<|end|>", 7) >= 0) {
                snprintf(wrapped, sizeof(wrapped),
                    "<|user|>\n%s<|end|>\n<|assistant|>\n", input);
                use_template = 1;
            }
            break;
        case 5: /* Gemma */
            if (tok_lookup(ps, "<start_of_turn>", 15) >= 0) {
                snprintf(wrapped, sizeof(wrapped),
                    "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", input);
                use_template = 1;
            }
            break;
        case 6: /* nanollama — <|user_start|>...<|user_end|><|assistant_start|> */
            snprintf(wrapped, sizeof(wrapped),
                "<|user_start|>%s<|user_end|><|assistant_start|>", input);
            use_template = 1;
            break;
        }
        if (!use_template) snprintf(wrapped, sizeof(wrapped), "%s", input);

        /* Tokenize wrapped input */
        int input_tokens[512];
        int n_input = 0;
        if (ps->bos_id >= 0) input_tokens[n_input++] = ps->bos_id;
        n_input += tokenize_input(ps, wrapped, input_tokens + n_input, 512 - n_input);

        int pos = 0;
        for (int i = 0; i < n_input && pos < max_seq - 1; i++, pos++)
            doe_forward(ps, &is, input_tokens[i], pos);

        int prev = input_tokens[n_input - 1];
        printf("  ");
        int total_births = 0, total_deaths = 0;

        for (int i = 0; i < 200 && pos < max_seq; i++, pos++) {
            float *lg = doe_forward(ps, &is, prev, pos);

            /* Field modulation on logits */
            field_step(1.0f);
            apply_field_to_logits(lg, ps->host_vocab);

            int next = sample(lg, ps->host_vocab, F.effective_temp, 40);

            /* Stop on EOS or chat-template end tokens */
            if (next == ps->eos_id) break;
            if (ps->vocab_tokens && next >= 0 && next < ps->vocab_size && ps->vocab_tokens[next]) {
                const char *ts = ps->vocab_tokens[next];
                if (strcmp(ts, "<|im_end|>") == 0 || strcmp(ts, "<|end|>") == 0 ||
                    strcmp(ts, "<|endoftext|>") == 0 || strcmp(ts, "<end_of_turn>") == 0 ||
                    strcmp(ts, "<|user|>") == 0 || strcmp(ts, "<|assistant_end|>") == 0 ||
                    strcmp(ts, "<|eot_id|>") == 0)
                    break;
            }

            /* Prophecy debt — retroactive conscience */
            float pd = compute_prophecy_debt(lg, next, ps->host_vocab);
            F.debt += pd;
            debt_sum += pd; debt_count++;

            /* NOTORCH Hebbian update — debt drives learning */
            float learn_signal = pd > 0.3f ? -pd : (1.0f - pd) * 0.1f;
            for (int l = 0; l < ps->n_field_layers; l++) {
                FieldLayer *fl = &ps->field_layers[l];
                for (int e = 0; e < MAX_EXPERTS; e++) {
                    if (!fl->experts[e].alive || fl->experts[e].tokens_seen == 0) continue;
                    notorch_step(fl->experts[e].lora_A, fl->experts[e].lora_B,
                                ps->host_dim, ps->host_dim, ps->lora_rank,
                                is.x, is.xb, learn_signal);
                }
            }

            /* Vitality + mitosis + apoptosis */
            if (i % 10 == 0) {
                /* Harmonic decomposition */
                float lh[16]; int lhl = 0;
                for (int j = 0; j < 16 && j < i; j++) lh[lhl++] = F.entropy;
                if (lhl > 2) harmonic_decompose(&is.hs, lh, lhl);

                for (int l = 0; l < ps->n_field_layers; l++) {
                    update_expert_vitality(&ps->field_layers[l], 10);
                    if (try_mitosis(&ps->field_layers[l], ps->host_dim, ps->lora_rank)) total_births++;
                    if (try_apoptosis(&ps->field_layers[l])) total_deaths++;
                }
            }

            /* Drift snapshot */
            if (i % DRIFT_INTERVAL == 0 && i > 0)
                drift_snapshot(&cd, F.debt, ps, &is.hs);

            token_decode_print(ps, next);
            fflush(stdout);
            prev = next;
        }
        printf("\n");

        /* Meta record */
        int te = 0;
        for (int l = 0; l < ps->n_field_layers; l++) te += ps->field_layers[l].n_alive;
        meta_record(&meta, F.step, te, ps->field_layers[0].parliament.consensus,
                    F.debt, F.field_health, debt_count > 0 ? debt_sum/debt_count : 0,
                    cd.drift, F.debt);

        if (total_births > 0 || total_deaths > 0)
            printf("  [life] births=%d deaths=%d\n", total_births, total_deaths);
        printf("\n");
    }
    free_infer(&is);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * HTTP SERVE MODE — minimal HTTP server for doe_ui.html and doe.html
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int g_serve_port = 0; /* 0 = disabled */

/* JSON-escape a string into buf. Returns bytes written (not counting NUL). */
static int json_escape(const char *src, char *buf, int bufsz) {
    int p = 0;
    for (; *src && p < bufsz - 2; src++) {
        switch (*src) {
        case '"':  if(p+2<bufsz){buf[p++]='\\';buf[p++]='"';}  break;
        case '\\': if(p+2<bufsz){buf[p++]='\\';buf[p++]='\\';} break;
        case '\n': if(p+2<bufsz){buf[p++]='\\';buf[p++]='n';}  break;
        case '\r': if(p+2<bufsz){buf[p++]='\\';buf[p++]='r';}  break;
        case '\t': if(p+2<bufsz){buf[p++]='\\';buf[p++]='t';}  break;
        default:   buf[p++] = *src; break;
        }
    }
    buf[p] = '\0';
    return p;
}

/* Read full HTTP request into buf, return total bytes. */
static int http_read_request(int fd, char *buf, int bufsz) {
    int total = 0;
    int content_length = -1;
    int header_end = -1;
    while (total < bufsz - 1) {
        int n = (int)read(fd, buf + total, bufsz - 1 - total);
        if (n <= 0) break;
        total += n;
        buf[total] = '\0';
        /* Find end of headers */
        if (header_end < 0) {
            char *hdr_end = strstr(buf, "\r\n\r\n");
            if (hdr_end) {
                header_end = (int)(hdr_end - buf) + 4;
                /* Parse Content-Length */
                char *cl = strcasestr(buf, "content-length:");
                if (cl) content_length = atoi(cl + 15);
                else content_length = 0;
            }
        }
        if (header_end >= 0 && total >= header_end + content_length) break;
    }
    return total;
}

/* Send full buffer over socket */
static void http_send(int fd, const char *data, int len) {
    int sent = 0;
    while (sent < len) {
        int n = (int)write(fd, data + sent, len - sent);
        if (n <= 0) break;
        sent += n;
    }
}

/* Send HTTP response header */
static void http_send_header(int fd, int status, const char *content_type, int content_length) {
    char hdr[512];
    const char *status_text = status == 200 ? "OK" : status == 404 ? "Not Found" : "Bad Request";
    int hlen;
    if (content_length >= 0) {
        hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %d\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Connection: close\r\n\r\n",
            status, status_text, content_type, content_length);
    } else {
        /* Streaming (SSE) — no content-length */
        hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Cache-Control: no-cache\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Connection: keep-alive\r\n\r\n",
            status, status_text, content_type);
    }
    http_send(fd, hdr, hlen);
}

/* Serve a static file (doe_ui.html, doe.html) */
static int http_serve_file(int fd, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *data = malloc(sz);
    if (!data) { fclose(f); return 0; }
    fread(data, 1, sz, f); fclose(f);
    http_send_header(fd, 200, "text/html; charset=utf-8", (int)sz);
    http_send(fd, data, (int)sz);
    free(data);
    return 1;
}

/* Extract JSON string value for a key from body. Simple parser. */
static int json_get_string(const char *json, const char *key, char *out, int outsz) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return 0;
    p = strchr(p + strlen(needle), ':');
    if (!p) return 0;
    while (*p && (*p == ':' || *p == ' ' || *p == '\t')) p++;
    if (*p != '"') return 0;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < outsz - 1) {
        if (*p == '\\' && p[1]) { p++; /* skip escape */ }
        out[i++] = *p++;
    }
    out[i] = '\0';
    return i;
}

/* Extract last user message from messages array in chat/completions body */
static int json_get_last_user_message(const char *body, char *out, int outsz) {
    /* Find last "role":"user" ... "content":"..." */
    const char *last_user = NULL;
    const char *p = body;
    while ((p = strstr(p, "\"role\"")) != NULL) {
        const char *rv = strstr(p, "\"user\"");
        if (rv && rv - p < 30) last_user = p;
        p++;
    }
    if (!last_user) return 0;
    return json_get_string(last_user, "content", out, outsz);
}

static float json_get_float(const char *json, const char *key, float def) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return def;
    p = strchr(p + strlen(needle), ':');
    if (!p) return def;
    return (float)atof(p + 1);
}

/* Run inference and stream SSE tokens */
static void http_stream_inference(int fd, GGUFIndex *ps, const char *user_msg, float temperature) {
    int max_seq = 512;
    InferState is = alloc_infer(ps, max_seq);

    /* Reset KV cache */
    int kd = ps->host_kv_heads * ps->host_head_dim;
    memset(is.key_cache, 0, ps->host_n_layers * max_seq * kd * 4);
    memset(is.value_cache, 0, ps->host_n_layers * max_seq * kd * 4);

    /* Wrap input in chat template */
    char wrapped[2048];
    switch (ps->chat_style) {
    case 1: snprintf(wrapped, sizeof(wrapped), "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_msg); break;
    case 2: snprintf(wrapped, sizeof(wrapped), "[INST] %s [/INST]", user_msg); break;
    case 3: snprintf(wrapped, sizeof(wrapped), "<|user|>\n%s\n<|assistant|>\n", user_msg); break;
    case 4: snprintf(wrapped, sizeof(wrapped), "<|user|>\n%s<|end|>\n<|assistant|>\n", user_msg); break;
    case 5: snprintf(wrapped, sizeof(wrapped), "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", user_msg); break;
    case 6: snprintf(wrapped, sizeof(wrapped), "<|user_start|>%s<|user_end|><|assistant_start|>", user_msg); break;
    default: snprintf(wrapped, sizeof(wrapped), "%s", user_msg); break;
    }

    /* Tokenize */
    int input_tokens[512];
    int n_input = 0;
    if (ps->bos_id >= 0) input_tokens[n_input++] = ps->bos_id;
    n_input += tokenize_input(ps, wrapped, input_tokens + n_input, 512 - n_input);

    /* Prefill */
    int pos = 0;
    for (int i = 0; i < n_input && pos < max_seq - 1; i++, pos++)
        doe_forward(ps, &is, input_tokens[i], pos);

    int prev = input_tokens[n_input - 1];

    /* Generate tokens, stream as SSE */
    for (int i = 0; i < 256 && pos < max_seq; i++, pos++) {
        float *lg = doe_forward(ps, &is, prev, pos);
        field_step(1.0f);
        apply_field_to_logits(lg, ps->host_vocab);

        float temp = temperature > 0.01f ? temperature : F.effective_temp;
        int next = sample(lg, ps->host_vocab, temp, 40);

        /* Stop on EOS */
        if (next == ps->eos_id) break;
        if (ps->vocab_tokens && next >= 0 && next < ps->vocab_size && ps->vocab_tokens[next]) {
            const char *ts = ps->vocab_tokens[next];
            if (strcmp(ts, "<|im_end|>") == 0 || strcmp(ts, "<|end|>") == 0 ||
                strcmp(ts, "<|endoftext|>") == 0 || strcmp(ts, "<end_of_turn>") == 0 ||
                strcmp(ts, "<|user|>") == 0 || strcmp(ts, "<|assistant_end|>") == 0 ||
                strcmp(ts, "<|eot_id|>") == 0) break;
        }

        /* Prophecy debt + Hebbian update */
        float pd = compute_prophecy_debt(lg, next, ps->host_vocab);
        F.debt += pd;
        float learn_signal = pd > 0.3f ? -pd : (1.0f - pd) * 0.1f;
        for (int l = 0; l < ps->n_field_layers; l++) {
            FieldLayer *fl = &ps->field_layers[l];
            for (int e = 0; e < MAX_EXPERTS; e++) {
                if (!fl->experts[e].alive || fl->experts[e].tokens_seen == 0) continue;
                notorch_step(fl->experts[e].lora_A, fl->experts[e].lora_B,
                            ps->host_dim, ps->host_dim, ps->lora_rank,
                            is.x, is.xb, learn_signal);
            }
        }

        /* Decode token to buffer */
        char tokbuf[256], escaped[512];
        token_decode_buf(ps, next, tokbuf, sizeof(tokbuf));
        json_escape(tokbuf, escaped, sizeof(escaped));

        /* Send SSE event */
        char sse[1024];
        int slen = snprintf(sse, sizeof(sse), "data: {\"token\":\"%s\"}\n\n", escaped);
        int wr = (int)write(fd, sse, slen);
        if (wr <= 0) break; /* client disconnected */

        prev = next;
    }

    /* Send done event */
    write(fd, "data: {\"done\":true}\n\n", 20);
    free_infer(&is);
}

/* Main HTTP serve loop */
static void serve_loop(GGUFIndex *ps, const char *exe_dir) {
    signal(SIGPIPE, SIG_IGN); /* ignore broken pipes */

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("[serve] socket"); return; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(g_serve_port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("[serve] bind"); close(server_fd); return;
    }
    if (listen(server_fd, 8) < 0) {
        perror("[serve] listen"); close(server_fd); return;
    }

    /* Resolve HTML file paths relative to executable */
    char ui_path[512], vis_path[512];
    snprintf(ui_path, sizeof(ui_path), "%sdoe_ui.html", exe_dir);
    snprintf(vis_path, sizeof(vis_path), "%sdoe.html", exe_dir);

    printf("[serve] parliament listening on http://0.0.0.0:%d\n", g_serve_port);
    printf("[serve]   /         → chat UI\n");
    printf("[serve]   /visual   → parliament terminal\n");
    printf("[serve]   /health   → status\n");
    printf("[serve]   POST /chat/completions → inference stream\n\n");

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client < 0) continue;

        char req[8192];
        int reqlen = http_read_request(client, req, sizeof(req));
        if (reqlen <= 0) { close(client); continue; }

        /* Parse method and path */
        char method[8] = "", path[256] = "";
        sscanf(req, "%7s %255s", method, path);

        /* Handle CORS preflight */
        if (strcmp(method, "OPTIONS") == 0) {
            const char *cors = "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type\r\n"
                "Content-Length: 0\r\n"
                "Connection: close\r\n\r\n";
            http_send(client, cors, (int)strlen(cors));
            close(client);
            continue;
        }

        if (strcmp(method, "GET") == 0) {
            if (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0) {
                if (!http_serve_file(client, ui_path)) {
                    const char *msg = "doe_ui.html not found";
                    http_send_header(client, 404, "text/plain", (int)strlen(msg));
                    http_send(client, msg, (int)strlen(msg));
                }
            } else if (strcmp(path, "/visual") == 0) {
                if (!http_serve_file(client, vis_path)) {
                    const char *msg = "doe.html not found";
                    http_send_header(client, 404, "text/plain", (int)strlen(msg));
                    http_send(client, msg, (int)strlen(msg));
                }
            } else if (strcmp(path, "/health") == 0) {
                char body[512];
                int blen = snprintf(body, sizeof(body),
                    "{\"status\":\"ok\",\"model\":\"%s\",\"arch\":\"%s\","
                    "\"params\":\"%dM\",\"vocab\":%d,\"layers\":%d,"
                    "\"experts\":%d,\"debt\":%.4f,\"health\":%.4f}",
                    ps->host_path, ps->host_arch,
                    (int)(ps->host_vocab * ps->host_dim * 2 / 1000000),
                    ps->host_vocab, ps->host_n_layers,
                    ps->n_field_layers > 0 ? ps->field_layers[0].n_alive : 0,
                    F.debt, F.field_health);
                http_send_header(client, 200, "application/json", blen);
                http_send(client, body, blen);
            } else {
                const char *msg = "not found";
                http_send_header(client, 404, "text/plain", (int)strlen(msg));
                http_send(client, msg, (int)strlen(msg));
            }
        } else if (strcmp(method, "POST") == 0 &&
                   (strcmp(path, "/chat/completions") == 0 || strcmp(path, "/v1/chat/completions") == 0)) {
            /* Find body after \r\n\r\n */
            char *body = strstr(req, "\r\n\r\n");
            if (!body) { close(client); continue; }
            body += 4;

            char user_msg[2048] = "";
            json_get_last_user_message(body, user_msg, sizeof(user_msg));

            if (user_msg[0] == '\0') {
                const char *err = "{\"error\":\"no user message\"}";
                http_send_header(client, 400, "application/json", (int)strlen(err));
                http_send(client, err, (int)strlen(err));
            } else {
                float temp = json_get_float(body, "temperature", 0.0f);
                printf("[serve] inference: \"%.*s\" temp=%.2f\n",
                       (int)(strlen(user_msg) > 60 ? 60 : strlen(user_msg)), user_msg, temp);
                http_send_header(client, 200, "text/event-stream", -1);
                http_stream_inference(client, ps, user_msg, temp);
            }
        } else {
            const char *msg = "method not allowed";
            http_send_header(client, 400, "text/plain", (int)strlen(msg));
            http_send(client, msg, (int)strlen(msg));
        }

        close(client);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN — the field manifests.
 * ═══════════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    printf("\n  doe.c — Democracy of Experts\n");
    printf("  θ = ε + γ + αδ — the parliament awakens.\n\n");

    char gguf_path[256] = "";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) snprintf(gguf_path, 256, "%s", argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) { g_n_threads = atoi(argv[++i]); if (g_n_threads < 1) g_n_threads = 1; }
        else if (strcmp(argv[i], "--prophecy") == 0 && i+1 < argc) { /* will be set after field_init */ }
        else if (strcmp(argv[i], "--destiny") == 0 && i+1 < argc) { /* will be set after field_init */ }
        else if (strcmp(argv[i], "--serve") == 0 && i+1 < argc) { g_serve_port = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("doe.c — DOE: inference architecture over any GGUF\n\n");
            printf("  --model PATH    GGUF to index (or auto-detect)\n");
            printf("  --serve PORT    start HTTP server for UI (doe_ui.html, doe.html)\n");
            printf("  --threads N     CPU threads for matvec (default: all cores)\n");
            printf("  --prophecy N    prediction horizon (default: 7)\n");
            printf("  --destiny F     destiny bias strength (default: 0.35)\n");
            printf("  --lora-rank N   LoRA rank (default: 16)\n");
            printf("  --lora-alpha F  LoRA injection strength (default: 0.1)\n\n");
            printf("  BLAS: cc doe.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o doe\n");
            printf("  GPU:  cc doe.c -O3 -lm -lpthread -DUSE_CUBLAS -lcublas -lcudart -o doe\n");
            return 0;
        }
    }

    /* ── Thread count for matvec ── */
    g_n_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (g_n_threads < 1) g_n_threads = 1;
    if (g_n_threads > 32) g_n_threads = 32;

    /* ── Field awakens ── */
    field_init();

    /* Parse field overrides */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prophecy") == 0 && i+1 < argc) F.prophecy = atoi(argv[++i]);
        else if (strcmp(argv[i], "--destiny") == 0 && i+1 < argc) F.destiny = atof(argv[++i]);
        else if (strcmp(argv[i], "--lora-rank") == 0 && i+1 < argc) { /* handled in index_load */ }
        else if (strcmp(argv[i], "--lora-alpha") == 0 && i+1 < argc) F.lora_alpha = atof(argv[++i]);
    }

    /* ── Environment scan ── */
    Environment env;
    env_scan(&env, __FILE__);

    /* ── PHASE 1: Search for DOE identity + gamma FIRST ── */
    char identity_path[256] = "";
    char gamma_path[256] = "";
    int weightless = 1;
    {
        static const char *wdirs[] = { "weights/", "doe_w/", "./", "../weights/", NULL };
        struct stat st;
        /* Search for doe_identity*.gguf (any variant: _micro, _mini, _q8, etc.) */
        for (int d = 0; wdirs[d] && identity_path[0] == '\0'; d++) {
            DIR *dir = opendir(wdirs[d]);
            if (!dir) continue;
            struct dirent *ent;
            int64_t best_size = 0;
            while ((ent = readdir(dir)) != NULL) {
                if (strncmp(ent->d_name, "doe_identity", 12) != 0) continue;
                int nlen = (int)strlen(ent->d_name);
                if (nlen < 5 || strcmp(ent->d_name + nlen - 5, ".gguf") != 0) continue;
                char tmp[256];
                snprintf(tmp, 256, "%s%s", wdirs[d], ent->d_name);
                if (stat(tmp, &st) == 0 && st.st_size > best_size) {
                    snprintf(identity_path, 256, "%s", tmp);
                    best_size = st.st_size;
                }
            }
            closedir(dir);
            if (identity_path[0] != '\0') {
                stat(identity_path, &st);
                printf("[identity] found: %s (%.1fMB)\n", identity_path, (float)st.st_size/(1024*1024));
                weightless = 0;
            }
        }
        /* Search for doe_gamma*.bin or doe_gamma*.npz */
        for (int d = 0; wdirs[d] && gamma_path[0] == '\0'; d++) {
            DIR *dir = opendir(wdirs[d]);
            if (!dir) continue;
            struct dirent *ent;
            while ((ent = readdir(dir)) != NULL) {
                if (strncmp(ent->d_name, "doe_gamma", 9) == 0 ||
                    strncmp(ent->d_name, "gamma_", 6) == 0) {
                    char tmp[256];
                    snprintf(tmp, 256, "%s%s", wdirs[d], ent->d_name);
                    if (stat(tmp, &st) == 0 && st.st_size > 0) {
                        snprintf(gamma_path, 256, "%s", tmp);
                        printf("[gamma] found: %s (%.1fMB)\n", tmp, (float)st.st_size/(1024*1024));
                        break;
                    }
                }
            }
            closedir(dir);
        }
        if (weightless)
            printf("[identity] no doe_identity.gguf — weightless mode.\n");
        if (gamma_path[0] == '\0')
            printf("[gamma] no doe_gamma.bin — parliament self-organizes.\n");
    }

    /* ── PHASE 2: Find host GGUF (external knowledge substrate) ── */
    if (gguf_path[0] == '\0') {
        if (identity_path[0] != '\0') {
            snprintf(gguf_path, 256, "%s", identity_path);
            printf("[host] using identity as host model.\n");
        } else {
            /* Also check all discovered GGUFs for doe.identity metadata */
            int identity_idx = -1, external_idx = -1;
            for (int i = 0; i < env.n_ggufs; i++) {
                if (strstr(env.ggufs[i].path, "mycelium/")) continue;
                if (strstr(env.ggufs[i].path, "doe_gamma")) continue;
                /* Quick sniff for doe.identity key in this GGUF */
                if (strstr(env.ggufs[i].path, "doe_identity")) {
                    identity_idx = i; continue;
                }
                if (external_idx < 0) external_idx = i;
            }
            /* Identity GGUF by name takes priority */
            if (identity_idx >= 0) {
                snprintf(gguf_path, 256, "%s", env.ggufs[identity_idx].path);
                printf("[host] found identity GGUF: %s\n", gguf_path);
                weightless = 0;
            } else if (external_idx >= 0) {
                snprintf(gguf_path, 256, "%s", env.ggufs[external_idx].path);
                printf("[host] indexing external: %s (%.1fMB)\n", gguf_path, (float)env.ggufs[external_idx].file_size/(1024*1024));
            } else {
                fprintf(stderr, "[error] no GGUF found. use --model PATH or place a .gguf nearby.\n");
                return 1;
            }
        }
    }

    /* ── Index GGUF ── */
    GGUFIndex idx;
    if (!index_load(&idx, gguf_path)) {
        fprintf(stderr, "[error] failed to index %s\n", gguf_path);
        return 1;
    }
    idx.weightless = weightless;

    /* If GGUF has doe.identity metadata — it's ours regardless of filename */
    if (idx.identity_tag[0] != '\0') {
        idx.weightless = 0;
        printf("[identity] verified via metadata: \"%s\"\n", idx.identity_tag);
    }

    /* ── Load gamma if found ── */
    if (gamma_path[0] != '\0') {
        FILE *gf = fopen(gamma_path, "rb");
        if (gf) {
            fseek(gf, 0, SEEK_END); long gsz = ftell(gf); fseek(gf, 0, SEEK_SET);
            idx.gamma_data = malloc(gsz);
            idx.gamma_size = (int)gsz;
            if (fread(idx.gamma_data, 1, gsz, gf) == (size_t)gsz)
                printf("[gamma] loaded %ld bytes — personality active.\n", gsz);
            else { free(idx.gamma_data); idx.gamma_data = NULL; idx.gamma_size = 0; }
            fclose(gf);
        }
    }

    /* ── Mycelium — check for existing LoRA spores ── */
    MyceliumState mycelium;
    mycelium_init(&mycelium);
    if (mycelium_load(&idx, idx.profile.fingerprint))
        printf("[mycelium] resumed adaptation for this index.\n");

    /* ── Chat or Serve ── */
    if (g_serve_port > 0) {
        /* Resolve directory of the executable for HTML files */
        char exe_dir[512] = "./";
        {
            /* Try to find doe_ui.html relative to argv[0] */
            char *slash = strrchr(argv[0], '/');
            if (slash) { int dlen = (int)(slash - argv[0]) + 1; if (dlen < 500) { memcpy(exe_dir, argv[0], dlen); exe_dir[dlen] = '\0'; } }
        }
        serve_loop(&idx, exe_dir);
    } else {
        chat(&idx);
    }

    /* ── Save spore on exit ── */
    mycelium_save(&idx, F.step, F.field_health);

    /* ── Cleanup ── */
    index_free(&idx);
    printf("[doe] the parliament adjourns. θ persists.\n");
    return 0;
}
