#include "matrix.h"
#include <stdlib.h>
#include <immintrin.h>

struct avx_priv {
    float values[4][4];
};

#define PRIV(x) \
    ((struct avx_priv *) ((x)->priv))

static void assign(Matrix *thiz, Mat4x4 data)
{
    /* FIXME: don't hardcode row & col */
    thiz->row = thiz->col = 4;

    thiz->priv = malloc(4 * 4 * sizeof(float));
    for (int i = 0; i < 4; i += 4){
        for(int j = 0; j < 4; j += 4){
            PRIV(thiz)->values[i][j] = data.values[i][j];
        }
    }
}

static const float epsilon = 1 / 10000.0;

static bool equal(const Matrix *l, const Matrix *r)
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (PRIV(l)->values[i][j] + epsilon < PRIV(r)->values[i][j] ||
                    PRIV(r)->values[i][j] + epsilon < PRIV(l)->values[i][j])
                return false;
    return true;
}

bool avx_mul(Matrix *dst, const Matrix *l, const Matrix *r)
{
    /* FIXME: error hanlding */
    __m256 ymm0, ymm1, ymm2;
    dst->priv = malloc(4 * 4 * sizeof(float));
    float *mul = malloc(sizeof(float) * 8);
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            PRIV(dst)->values[i][j] = 0;
            for (int k = 0; k < 4; k+=8){
                ymm0 = _mm256_set_ps(PRIV(l)->values[i][k+7],PRIV(l)->values[i][k+6],PRIV(l)->values[i][k+5],PRIV(l)->values[i][k+4], PRIV(l)->values[i][k+3], PRIV(l)->values[i][k+2], PRIV(l)->values[i][k+1], PRIV(l)->values[i][k]);
                ymm1 = _mm256_set_ps(PRIV(r)->values[k+7][j],PRIV(r)->values[k+6][j],PRIV(r)->values[k+5][j],PRIV(r)->values[k+4][j], PRIV(r)->values[k+3][j], PRIV(r)->values[k+2][j], PRIV(r)->values[k+1][j], PRIV(r)->values[k][j]);
                ymm2 = _mm256_mul_ps(ymm0, ymm1);
                _mm256_storeu_ps(mul, ymm2);
                PRIV(dst)->values[i][j] += mul[0] + mul[1] + mul[2] + mul[3] + mul[4] + mul[5] + mul[6] + mul[7];
            }
        }
    }
    return true;
}

MatrixAlgo AVXMatrixProvider = {
    .assign = assign,
    .equal = equal,
    .mul = avx_mul,
};
