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

#define ACCUMULATE_COLUMN(source_tile, input_row, weight) do { \
    svuint64_t raw = svread_ver_za64_u64_m(svdup_u64(0), all, source_tile, input_row); \
    svuint16_t source = svreinterpret_u16_u64(raw); \
    svuint16_t packed = svreinterpret_u16_u64(svld1rq_u64(all, weight)); \
    svuint16_t real = svtbl_u16(packed, real_indices); \
    svuint16_t imaginary = svtbl_u16(packed, imaginary_indices); \
    svmopa_za64_u16_m(0, digits, digits, real, source); \
    svmopa_za64_u16_m(1, digits, digits, imaginary, source); \
} while (0)

__arm_locally_streaming __arm_new("za")
int akita_packing_sme_source(const uint64_t *source, const uint64_t *weights,
                             size_t terms, uint64_t *output) {
    const size_t stride = 16;
    const size_t panel = 16;
    const size_t lanes = 8;
    const size_t d = 2048;
    const size_t positions = terms / stride;
    if (svcntd() != lanes || terms == 0 || terms > 16 * 1024 || terms % stride != 0)
        return 0;
    uint64_t scratch[128 * 14];
    svbool_t all = svptrue_b64();
    svbool_t digits = svptrue_b16();
    svuint16_t real_indices = svld1_u16(digits, packing_indices[0]);
    svuint16_t imaginary_indices = svld1_u16(digits, packing_indices[1]);
    const size_t subring = d / stride;
    for (size_t first = 0; first < positions; first += panel) {
        const size_t end = positions - first < panel ? positions : first + panel;
        for (size_t j = 0; j < subring; j += lanes) {
            uint64_t *saved = scratch + j * 14;
            svzero_za();
            if (first != 0) {
                for (uint32_t k = 0; k < 7; k++) {
                    svld1_hor_za64(0, k, all, saved + lanes * k);
                    svld1_hor_za64(1, k, all, saved + lanes * (k + 7));
                }
            }
            for (size_t position = first; position < end; position++) {
                const uint64_t *row = source + position * d + j * stride;
                for (uint32_t k = 0; k < lanes; k++, row += stride) {
                    svwrite_hor_za64_u64_m(6, k, all, svld1_u64(all, row));
                    svwrite_hor_za64_u64_m(7, k, all, svld1_u64(all, row + lanes));
                }
                const uint64_t *by_position = weights + position * stride * 2;
                for (uint32_t k = 0; k < lanes; k++, by_position += 2) {
                    ACCUMULATE_COLUMN(6, k, by_position);
                    ACCUMULATE_COLUMN(7, k, by_position + lanes * 2);
                }
            }
            if (end != positions) {
                for (uint32_t k = 0; k < 7; k++) {
                    svst1_hor_za64(0, k, all, saved + lanes * k);
                    svst1_hor_za64(1, k, all, saved + lanes * (k + 7));
                }
            } else {
                svuint64_t real, imaginary;
                MERGE(0, real);
                MERGE(1, imaginary);
                svst1_u64(all, output + j, real);
                svst1_u64(all, output + subring + j, imaginary);
            }
        }
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
