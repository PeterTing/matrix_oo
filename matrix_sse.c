#include "matrix.h"
#include <stdlib.h>
#include <emmintrin.h>

struct sse_priv {
    float values[4][4];
};

#define PRIV(x) \
    ((struct sse_priv *) ((x)->priv))

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

bool sse_mul(Matrix *dst, const Matrix *l, const Matrix *r)
{
    /* FIXME: error hanlding */
    __m128 xmm0, xmm1, xmm2;
    dst->priv = malloc(4 * 4 * sizeof(float));
    float *mul = malloc(sizeof(float) * 4);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++){
                xmm0 = _mm_setr_ps(PRIV(l)->values[i][k],PRIV(l)->values[i][k+1],PRIV(l)->values[i][k+2],PRIV(l)->values[i][k+3]);
                xmm1 = _mm_setr_ps(PRIV(r)->values[k][j],PRIV(r)->values[k+1][j],PRIV(r)->values[k+2][j],PRIV(r)->values[k+3][j]);
                xmm2 = _mm_mul_ps(xmm0, xmm1);
                _mm_store_ps(mul, xmm2);
                PRIV(dst)->values[i][j] += mul[0] + mul[1] + mul[2] + mul[3];
            }
    return true;
}

MatrixAlgo SSEMatrixProvider = {
    .assign = assign,
    .equal = equal,
    .mul = sse_mul,
};
