#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/sysctl.h>

#define Q UINT64_C(18446744073709551557)

#define READ_LIMB(tile, line) svread_hor_za64_u64_m(svdup_u64(0), all, tile, line)

#define MERGE(tile, offset) do { \
    svuint64_t degree0 = svmla_n_u64_x(all, READ_LIMB(tile, 0), READ_LIMB(tile, 4), 59); \
    svuint64_t degree1 = svmla_n_u64_x(all, READ_LIMB(tile, 1), READ_LIMB(tile, 5), 59); \
    svuint64_t degree2 = svmla_n_u64_x(all, READ_LIMB(tile, 2), READ_LIMB(tile, 6), 59); \
    svuint64_t degree3 = READ_LIMB(tile, 3); \
    degree1 = svadd_u64_x(all, degree1, svlsr_n_u64_x(all, degree0, 16)); \
    degree2 = svadd_u64_x(all, degree2, svlsr_n_u64_x(all, degree1, 16)); \
    degree3 = svadd_u64_x(all, degree3, svlsr_n_u64_x(all, degree2, 16)); \
    svuint64_t lo = svorr_u64_x(all, svand_n_u64_x(all, degree0, 65535), \
        svlsl_n_u64_x(all, svand_n_u64_x(all, degree1, 65535), 16)); \
    lo = svorr_u64_x(all, lo, svlsl_n_u64_x(all, svand_n_u64_x(all, degree2, 65535), 32)); \
    lo = svorr_u64_x(all, lo, svlsl_n_u64_x(all, degree3, 48)); \
    svuint64_t value = svmla_n_u64_x(all, lo, svlsr_n_u64_x(all, degree3, 16), 59); \
    svbool_t reduced = svorr_b_z(all, svcmplt_u64(all, value, lo), svcmpge_n_u64(all, value, Q)); \
    (offset) = svsub_n_u64_m(reduced, value, Q); \
} while (0)


static const uint16_t packing_indices[2][32] = {
    {0, 64, 64, 64, 1, 0, 64, 64, 2, 1, 0, 64, 3, 2, 1, 0, 64, 3, 2, 1, 64, 64, 3, 2, 64, 64, 64, 3, 64, 64, 64, 64},
    {4, 64, 64, 64, 5, 4, 64, 64, 6, 5, 4, 64, 7, 6, 5, 4, 64, 7, 6, 5, 64, 64, 7, 6, 64, 64, 64, 7, 64, 64, 64, 64},
};

#define ACCUMULATE(REAL, IMAG, OFFSET) do { \
    svuint16_t input = svreinterpret_u16_u64(svld1_u64(all, source + OFFSET)); \
    svmopa_za64_u16_m(REAL, digits, digits, real_weight, input); \
    svmopa_za64_u16_m(IMAG, digits, digits, imag_weight, input); \
} while (0)

#define STORE(REAL, IMAG, OFFSET) do { \
    svuint64_t real, imag; \
    MERGE(REAL, real); \
    MERGE(IMAG, imag); \
    svst1_u64(all, output + stripe + OFFSET, real); \
    svst1_u64(all, output + 128 + stripe + OFFSET, imag); \
} while (0)

__arm_locally_streaming __arm_new("za")
int akita_packing_sme_striped(const uint64_t *source, const uint64_t *weights,
                    size_t terms, uint64_t *output) {
    if (svcntd() != 8 || terms == 0 || terms > 16 * 1024) return 0;
    svbool_t all = svptrue_b64();
    svbool_t digits = svptrue_b16();
    svuint16_t real_indices = svld1_u16(digits, packing_indices[0]);
    svuint16_t imaginary_indices = svld1_u16(digits, packing_indices[1]);
    for (size_t stripe = 0; stripe < 128; stripe += 16) {
        svzero_za();
        for (size_t term = 0; term < terms; ++term, source += 16) {
            svuint16_t weight = svreinterpret_u16_u64(svld1rq_u64(all, weights + term * 2));
            svuint16_t real_weight = svtbl_u16(weight, real_indices);
            svuint16_t imag_weight = svtbl_u16(weight, imaginary_indices);
            ACCUMULATE(0, 1, 0);
            ACCUMULATE(2, 3, 8);
        }
        STORE(0, 1, 0);
        STORE(2, 3, 8);
    }
    return 1;
}

__arm_locally_streaming
static int full_streaming_length(void) {
    return svcntd() == 8;
}

int akita_packing_sme_available(void) {
    int feature = 0;
    size_t size = sizeof(feature);
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &feature, &size, NULL, 0) || !feature)
        return 0;
    feature = 0;
    size = sizeof(feature);
    return !sysctlbyname("hw.optional.arm.FEAT_SME_I16I64", &feature, &size, NULL, 0) &&
        feature && full_streaming_length();
}
