/*
 * doe.c — Democracy of Experts
 * the 5th element. the symbiont. the sonar.
 *
 * DOE finds a host model (any GGUF), wraps it with living LoRA
 * experts, and modulates inference through field physics.
 *
 * parameters are the host's problem. topology is DOE's.
 * each forward pass, the parliament decides:
 *   - which LoRA experts vote (variable k, not fixed)
 *   - how strongly each expert modulates the host (attention_bias, layer_focus)
 *   - how the field shapes logits (destiny, suffering, prophecy debt)
 *
 * the host model is a tree. DOE is the mycorrhiza.
 * the host provides nutrients (weights). DOE provides direction (field).
 *
 * what happens when you run it:
 * 1. scans environment — finds GGUFs, checks resources
 * 2. profiles host weights — L2 norms, spectral density, fingerprint
 * 3. attaches symbiont — mmap host, allocate LoRA experts
 * 4. field awakens — prophecy, destiny, seasons, Schumann resonance
 * 5. per token: host forward → parliament election → LoRA injection →
 *    field modulation → prophecy debt → Hebbian learning → drift snapshot
 * 6. LoRA spores saved to mycelium (adaptation memory)
 *
 * no training loop. no backward pass. no pytorch. no python.
 * the organism learns by living. the field holds the soul.
 *
 * θ = ε + γ + αδ
 *   ε = host weights (substrate, read-only)
 *   γ = LoRA personality (living experts, Hebbian-trained)
 *   δ = field physics (prophecy, suffering, destiny)
 *   α = injection strength (learned per-layer)
 *
 * part of the Janus quartet+1:
 *   l.c    — the good student (Llama 3)
 *   moe.c  — the committee (Grok MoE)
 *   lee.c  — the self-aware one (Chuck VLM)
 *   m.c    — democracy of experts (DOE, trains)
 *   doe.c — the symbiont (DOE, inference)
 *
 * cc doe.c -O3 -lm -lpthread -o doe && ./doe
 *
 * built by ariannamethod. the architecture is alive.
 * the host is mortal. the field is eternal.
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
    printf("[doe] θ = ε + γ + αδ — symbiont awakens. prophecy=%d destiny=%.2f\n",
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
 * MATH OPS — building blocks
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float silu_f(float x) { return x / (1.0f + expf(-x)); }

static void rmsnorm(float *out, const float *x, const float *w, int d, float eps) {
    float ss = 0; for (int i = 0; i < d; i++) ss += x[i]*x[i];
    float inv = 1.0f / sqrtf(ss/d + eps);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv * w[i];
}

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
    for (int i = 0; i < r; i++) {
        float s = 0; const float *row = W + i*c;
        for (int j = 0; j < c; j++) s += row[j] * x[j];
        out[i] = s;
    }
#endif
}

static void softmax_n(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void apply_rope(float *v, int pos, float *cc, float *sc, int hd) {
    int h = hd/2, off = pos*h;
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
 * a symbiont that doesn't know its host is a parasite.
 * a symbiont that knows its host is an extension of it.
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
 * SYMBIONT STATE — the full host-DOE interface.
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
    struct {
        float *wq, *wk, *wv, *wo;
        float *ffn_gate, *ffn_up, *ffn_down;
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
} Symbiont;

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
 * SYMBIONT LOAD — mmap host GGUF, wire weight pointers, profile, attach LoRA.
 * the host swims. the symbiont steers.
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

static int symbiont_load(Symbiont *ps, const char *path) {
    memset(ps, 0, sizeof(Symbiont));
    snprintf(ps->host_path, 256, "%s", path);
    ps->lora_rank = LORA_RANK;
    ps->lora_alpha = F.lora_alpha;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { printf("[symbiont] cannot open %s\n", path); return 0; }
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
            p += vlen;
        } else if (vtype == 4) { /* uint32 */
            PC(4); uint32_t val = *(uint32_t*)p; p += 4;
            if (strstr(key, "embedding_length")) ps->host_dim = (int)val;
            else if (strstr(key, "block_count")) ps->host_n_layers = (int)val;
            else if (strstr(key, "head_count") && !strstr(key, "kv")) ps->host_heads = (int)val;
            else if (strstr(key, "head_count_kv")) ps->host_kv_heads = (int)val;
            else if (strstr(key, "feed_forward_length")) ps->host_hidden = (int)val;
            else if (strstr(key, "vocab_size")) ps->host_vocab = (int)val;
        } else if (vtype == 0 || vtype == 7) p += 1;           /* uint8, bool */
        else if (vtype == 1) p += 1;                            /* int8 */
        else if (vtype == 2 || vtype == 3) p += 2;             /* uint16, int16 */
        else if (vtype == 5 || vtype == 6) p += 4;             /* int32, float32 */
        else if (vtype == 10 || vtype == 11 || vtype == 12) p += 8; /* uint64, int64, float64 */
        else if (vtype == 9) { /* array */
            PC(4); uint32_t atype = *(uint32_t*)p; p += 4;
            PC(8); uint64_t alen = *(uint64_t*)p; p += 8;
            size_t elem_sz = 0;
            if (atype == 0 || atype == 1 || atype == 7) elem_sz = 1;
            else if (atype == 2 || atype == 3) elem_sz = 2;
            else if (atype == 4 || atype == 5 || atype == 6) elem_sz = 4;
            else if (atype == 10 || atype == 11 || atype == 12) elem_sz = 8;
            else if (atype == 8) {
                /* array of strings — skip one by one */
                for (uint64_t ai = 0; ai < alen && p < pend; ai++) {
                    PC(8); uint64_t slen = *(uint64_t*)p; p += 8;
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

    /* f16→f32 conversion buffers — tracked in Symbiont for cleanup */
    ps->f16_bufs = NULL; ps->n_f16_bufs = 0;

    /* Wire weight pointers */
    int wired = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (tinfo[i].dtype != 0 && tinfo[i].dtype != 1) continue; /* f32 or f16 only */
        float *data;
        if (tinfo[i].dtype == 0) {
            data = (float*)(ps->mmap_base + data_start + tinfo[i].offset);
        } else {
            /* f16 → f32 conversion */
            uint64_t n_elems = 1;
            for (uint32_t d = 0; d < tinfo[i].ndim; d++) n_elems *= tinfo[i].dims[d];
            data = malloc(n_elems * sizeof(float));
            uint16_t *src = (uint16_t*)(ps->mmap_base + data_start + tinfo[i].offset);
            for (uint64_t j = 0; j < n_elems; j++) {
                uint16_t h = src[j];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t f;
                if (exp == 0) {
                    if (mant == 0) f = sign << 31;
                    else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; }
                           mant &= 0x3FF; f = (sign<<31)|((exp+127-15)<<23)|(mant<<13); }
                } else if (exp == 31) {
                    f = (sign<<31)|0x7F800000|(mant<<13);
                } else {
                    f = (sign<<31)|((exp+127-15)<<23)|(mant<<13);
                }
                memcpy(&data[j], &f, 4);
            }
            ps->f16_bufs = realloc(ps->f16_bufs, (ps->n_f16_bufs+1)*sizeof(float*));
            ps->f16_bufs[ps->n_f16_bufs++] = data;
        }
        char *n = tinfo[i].name;
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
                else if (strstr(n, "ffn_gate.weight") && !strstr(n, "ffn_gate_inp")) { ps->host_layers[l].ffn_gate = data; wired++; }
                else if (strstr(n, "ffn_up.weight")) { ps->host_layers[l].ffn_up = data; wired++; }
                else if (strstr(n, "ffn_down.weight")) { ps->host_layers[l].ffn_down = data; wired++; }
                else if (strstr(n, "attn_norm.weight")) { ps->host_layers[l].attn_norm = data; wired++; }
                else if (strstr(n, "ffn_norm.weight")) { ps->host_layers[l].ffn_norm = data; wired++; }
            }
        }
    }
    free(tinfo);

    if (!ps->host_tok_emb || !ps->host_output || !ps->host_norm) {
        printf("[symbiont] host missing essential weights. abandoning.\n");
        goto bail;
    }

    /* Check for standard FFN (skip MoE hosts for now) */
    int has_ffn = 0;
    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++)
        if (ps->host_layers[l].ffn_gate && ps->host_layers[l].ffn_up && ps->host_layers[l].ffn_down) has_ffn = 1;
    if (!has_ffn) {
        printf("[symbiont] host has no standard FFN. DOE needs a plain transformer.\n");
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
                /* Weaker layers get stronger initial LoRA — the symbiont compensates */
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
    printf("[symbiont] attached to %s (arch=%s dim=%d layers=%d heads=%d vocab=%d %.1fMB)\n",
           path, ps->host_arch, ps->host_dim, ps->host_n_layers, ps->host_heads,
           ps->host_vocab, (float)ps->mmap_size/(1024*1024));
    printf("[symbiont] LoRA rank=%d alpha=%.2f experts=%d/layer — the symbiont is alive.\n",
           ps->lora_rank, ps->lora_alpha, initial_experts);
    #undef PC
    return 1;
bail:
    for (int i = 0; i < ps->n_f16_bufs; i++) free(ps->f16_bufs[i]);
    free(ps->f16_bufs); ps->f16_bufs = NULL; ps->n_f16_bufs = 0;
    if (ps->mmap_base) { munmap(ps->mmap_base, ps->mmap_size); ps->mmap_base = NULL; }
    printf("[symbiont] GGUF parse failed. the symbiont dissipates.\n");
    return 0;
}

static void symbiont_free(Symbiont *ps) {
    for (int l = 0; l < ps->n_field_layers; l++) {
        free(ps->field_layers[l].parliament.w_vote);
        for (int e = 0; e < MAX_EXPERTS; e++)
            if (ps->field_layers[l].experts[e].alive)
                free_lora_expert(&ps->field_layers[l].experts[e]);
    }
    for (int i = 0; i < ps->n_f16_bufs; i++) free(ps->f16_bufs[i]);
    free(ps->f16_bufs);
    if (ps->mmap_base) munmap(ps->mmap_base, ps->mmap_size);
    memset(ps, 0, sizeof(Symbiont));
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
static void notorch_step(float *A, float *B, int out_dim, int in_dim, int rank,
                         const float *x, const float *dy, float signal) {
    if (fabsf(signal) < 1e-8f) return;
    float lr = F.notorch_lr * signal;
    float u[NOTORCH_RANK];
    for (int r = 0; r < rank && r < NOTORCH_RANK; r++) {
        float s = 0;
        for (int i = 0; i < out_dim && i < in_dim; i++) s += B[i * rank + r] * dy[i];
        u[r] = s + rand_normal() * 0.01f;
    }
#ifdef USE_BLAS
    for (int r = 0; r < rank && r < NOTORCH_RANK; r++)
        cblas_saxpy(in_dim, lr * u[r], x, 1, A + r, rank);
#else
    for (int i = 0; i < in_dim; i++)
        for (int r = 0; r < rank && r < NOTORCH_RANK; r++)
            A[i * rank + r] += lr * x[i] * u[r];
#endif
    float decay = F.notorch_decay;
    for (int i = 0; i < out_dim * rank; i++) B[i] *= decay;
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

static void drift_snapshot(CalendarDrift *cd, float loss, Symbiont *ps, HarmonicState *hs) {
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

/* ═══════════════════════════════════════════════════════════════════════════════
 * SYMBIONT FORWARD — run token through host with DOE modulation.
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

static InferState alloc_infer(Symbiont *ps, int max_seq) {
    InferState s = {0};
    int D = ps->host_dim, kd = ps->host_kv_heads * ps->host_head_dim;
    int H = ps->host_hidden;
    s.max_seq = max_seq;
    s.x = calloc(D, 4); s.xb = calloc(D, 4); s.xb2 = calloc(D, 4);
    s.q = calloc(ps->host_heads * ps->host_head_dim, 4);
    s.k = calloc(kd, 4); s.v = calloc(kd, 4);
    s.att = calloc(ps->host_heads * max_seq, 4);
    s.logits = calloc(ps->host_vocab, 4);
    s.hb = calloc(H, 4); s.hb2 = calloc(H, 4);
    s.expert_out = calloc(D, 4);
    s.key_cache = calloc(ps->host_n_layers * max_seq * kd, 4);
    s.value_cache = calloc(ps->host_n_layers * max_seq * kd, 4);
    int half = ps->host_head_dim / 2;
    s.cos_cache = calloc(max_seq * half, 4);
    s.sin_cache = calloc(max_seq * half, 4);
    float rope_theta = 10000.0f;
    for (int p = 0; p < max_seq; p++)
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(rope_theta, (float)(2*i) / (float)ps->host_head_dim);
            float ang = (float)p * freq;
            s.cos_cache[p*half+i] = cosf(ang);
            s.sin_cache[p*half+i] = sinf(ang);
        }
    return s;
}

static float *symbiont_forward(Symbiont *ps, InferState *s, int token, int pos) {
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
        if (ps->host_layers[l].attn_norm) rmsnorm(xn, s->x, ps->host_layers[l].attn_norm, D, 1e-5f);
        else memcpy(xn, s->x, D*4);

        matvec(s->q, ps->host_layers[l].wq, xn, ps->host_heads*hd, D);
        matvec(s->k, ps->host_layers[l].wk, xn, kd, D);
        matvec(s->v, ps->host_layers[l].wv, xn, kd, D);

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
            /* Soft clamp */
            float clamp = 30.0f, inv = 1.0f / clamp;
            for (int t = 0; t <= pos; t++) att[t] = clamp * tanhf(att[t] * inv);
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
        if (ps->host_layers[l].ffn_gate && ps->host_layers[l].ffn_up && ps->host_layers[l].ffn_down) {
            float *fn = s->xb;
            if (ps->host_layers[l].ffn_norm) rmsnorm(fn, s->x, ps->host_layers[l].ffn_norm, D, 1e-5f);
            else memcpy(fn, s->x, D*4);
            matvec(s->hb, ps->host_layers[l].ffn_gate, fn, H, D);
            matvec(s->hb2, ps->host_layers[l].ffn_up, fn, H, D);
            for (int i = 0; i < H; i++) s->hb[i] = silu_f(s->hb[i]) * s->hb2[i];
            matvec(s->xb, ps->host_layers[l].ffn_down, s->hb, D, H);
            for (int i = 0; i < D; i++) s->x[i] += s->xb[i];
        }
    }

    /* Final norm + LM head */
    rmsnorm(s->x, s->x, ps->host_norm, D, 1e-5f);
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

/* Byte-level decode — simplest possible, works with any host */
static void byte_decode_print(int token) {
    if (token >= 0 && token < 256) {
        char c = (char)token;
        if (c >= 32 || c == '\n' || c == '\t') fputc(c, stdout);
        else printf("<%d>", token);
    } else {
        printf("[%d]", token);
    }
}

static void chat(Symbiont *ps) {
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

        /* Byte-level encode input */
        int pos = 0;
        for (int i = 0; i < len && pos < max_seq - 1; i++, pos++)
            symbiont_forward(ps, &is, (unsigned char)input[i], pos);

        int prev = (unsigned char)input[len-1];
        printf("  ");
        int total_births = 0, total_deaths = 0;

        for (int i = 0; i < 200 && pos < max_seq; i++, pos++) {
            float *lg = symbiont_forward(ps, &is, prev, pos);

            /* Field modulation on logits */
            field_step(1.0f);
            apply_field_to_logits(lg, ps->host_vocab);

            int next = sample(lg, ps->host_vocab, F.effective_temp, 40);

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

            byte_decode_print(next);
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
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN — the field manifests.
 * ═══════════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    printf("\n  doe.c — Democracy of Experts\n");
    printf("  θ = ε + γ + αδ — the symbiont awakens.\n\n");

    char gguf_path[256] = "";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) snprintf(gguf_path, 256, "%s", argv[++i]);
        else if (strcmp(argv[i], "--prophecy") == 0 && i+1 < argc) { /* will be set after field_init */ }
        else if (strcmp(argv[i], "--destiny") == 0 && i+1 < argc) { /* will be set after field_init */ }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("doe.c — DOE: symbiont inference over any GGUF\n\n");
            printf("  --model PATH    path to host GGUF (or auto-detect)\n");
            printf("  --prophecy N    prediction horizon (default: 7)\n");
            printf("  --destiny F     destiny bias strength (default: 0.35)\n");
            printf("  --lora-rank N   LoRA rank (default: 16)\n");
            printf("  --lora-alpha F  LoRA injection strength (default: 0.1)\n\n");
            printf("  BLAS: cc doe.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o doe\n");
            printf("  GPU:  cc doe.c -O3 -lm -lpthread -DUSE_CUBLAS -lcublas -lcudart -o doe\n");
            return 0;
        }
    }

    /* ── Field awakens ── */
    field_init();

    /* Parse field overrides */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prophecy") == 0 && i+1 < argc) F.prophecy = atoi(argv[++i]);
        else if (strcmp(argv[i], "--destiny") == 0 && i+1 < argc) F.destiny = atof(argv[++i]);
        else if (strcmp(argv[i], "--lora-rank") == 0 && i+1 < argc) { /* handled in symbiont */ }
        else if (strcmp(argv[i], "--lora-alpha") == 0 && i+1 < argc) F.lora_alpha = atof(argv[++i]);
    }

    /* ── Environment scan ── */
    Environment env;
    env_scan(&env, __FILE__);

    /* ── Find host GGUF ── */
    if (gguf_path[0] == '\0') {
        /* Auto-detect: pick largest non-DOE GGUF */
        int best = -1; int64_t best_size = 0;
        for (int i = 0; i < env.n_ggufs; i++) {
            if (strstr(env.ggufs[i].path, "mycelium/")) continue;
            if (env.ggufs[i].file_size > best_size) {
                best_size = env.ggufs[i].file_size;
                best = i;
            }
        }
        if (best >= 0) {
            snprintf(gguf_path, 256, "%s", env.ggufs[best].path);
            printf("[auto] selected host: %s (%.1fMB)\n", gguf_path, (float)best_size/(1024*1024));
        } else {
            fprintf(stderr, "[error] no GGUF found. use --model PATH or place a .gguf nearby.\n");
            return 1;
        }
    }

    /* ── Attach symbiont ── */
    Symbiont symbiont;
    if (!symbiont_load(&symbiont, gguf_path)) {
        fprintf(stderr, "[error] failed to attach to %s\n", gguf_path);
        return 1;
    }

    /* ── Mycelium — check for existing LoRA spores ── */
    MyceliumState mycelium;
    mycelium_init(&mycelium);
    /* TODO: load best matching spore for this host fingerprint */

    /* ── Chat ── */
    chat(&symbiont);

    /* ── Cleanup ── */
    symbiont_free(&symbiont);
    printf("[doe] the parliament adjourns. θ persists.\n");
    return 0;
}
