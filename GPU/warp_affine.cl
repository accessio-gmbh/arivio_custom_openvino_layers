// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define _CAT(a,b) a##b
#define CAT(a,b) _CAT(a,b)

inline void warpAffine(const int N, const int C, const __global INPUT0_TYPE* src, const int IH, const int IW,
                       __global OUTPUT0_TYPE* dst, const int OH, const int OW,
                       const __global INPUT1_TYPE* matrix)
{
    int y = get_global_id(1);
    int x = get_global_id(2);
    int bc = get_global_id(0);
    int b = bc / C;
    int c = b % C;

    if (x >= OW || y >= OH || c >= C)
        return;

    int matrix_idx = (b)*INPUT1_PITCHES[0] + (c)*INPUT1_PITCHES[1];


// This part is for warping normalized (positions in interval [-1,1]), which is same as BilinearInterpolation layer
//    INPUT1_TYPE m0 = IW/(OW-1.f)*matrix[matrix_idx+0];
//    INPUT1_TYPE m1 = IW/(OH-1.f)*matrix[matrix_idx+1];
//    INPUT1_TYPE m2 = 0.5f*IW*(-matrix[matrix_idx+0]-matrix[matrix_idx+1]+matrix[matrix_idx+2]+1.f);
//    INPUT1_TYPE m3 = IH/(OW-1.f)*matrix[matrix_idx+3];
//    INPUT1_TYPE m4 = IH/(OH-1.f)*matrix[matrix_idx+4];
//    INPUT1_TYPE m5 = 0.5f*IH*(-matrix[matrix_idx+3]-matrix[matrix_idx+4]+matrix[matrix_idx+5]+1.f);
//
//    INPUT0_TYPE xi = x*m0 + y*m1 + m2;
//    INPUT0_TYPE yi = x*m3 + y*m4 + m5;

// This part is for unnormalized (positions in pixel) input
    INPUT0_TYPE xi = x*matrix[matrix_idx] + y*matrix[matrix_idx+1] + matrix[matrix_idx+2];
    INPUT0_TYPE yi = x*matrix[matrix_idx+3] + y*matrix[matrix_idx+4] + matrix[matrix_idx+5];


//    INPUT0_TYPE xo_n = (2*x) / (INPUT0_TYPE)(OW-1) - (INPUT0_TYPE)(1.f);
//    INPUT0_TYPE yo_n = (2*y) / (INPUT0_TYPE)(OH-1) - (INPUT0_TYPE)(1.f);
//    INPUT0_TYPE xi_n = xo_n*(INPUT0_TYPE)(matrix[0]) + yo_n*(INPUT0_TYPE)(matrix[1]) + (INPUT0_TYPE)(matrix[2]);
//    INPUT0_TYPE yi_n = xo_n*(INPUT0_TYPE)(matrix[3]) + yo_n*(INPUT0_TYPE)(matrix[4]) + (INPUT0_TYPE)(matrix[5]);
//    INPUT0_TYPE xi = (xi_n + (INPUT0_TYPE)(1.f)) * (INPUT0_TYPE)(0.5f) * (INPUT0_TYPE)(IW);
//    INPUT0_TYPE yi = (yi_n + (INPUT0_TYPE)(1.f)) * (INPUT0_TYPE)(0.5f) * (INPUT0_TYPE)(IH);

    int out_idx = (b)*OUTPUT0_PITCHES[0] + (c)*OUTPUT0_PITCHES[1] + (y)*OUTPUT0_PITCHES[2] + (x)*OUTPUT0_PITCHES[3];
    if (xi < (INPUT0_TYPE)(0.f) || yi < (INPUT0_TYPE)(0.f) || xi > IW-(INPUT0_TYPE)(1.f) || yi > IH-(INPUT0_TYPE)(1.f)){
        dst[out_idx] = (OUTPUT0_TYPE)(84.f);
    }
    else {
        int iy0 = (int)(yi);
        int iy1 = iy0 + 1;
        INPUT0_TYPE y_lambda0 = yi - iy0;
        INPUT0_TYPE y_lambda1 = (INPUT0_TYPE)(1.0f) - y_lambda0;

        int ix0 = (int)(xi);
        int ix1 = ix0 + 1;
        INPUT0_TYPE x_lambda0 = xi - ix0;
        INPUT0_TYPE x_lambda1 = (INPUT0_TYPE)(1.0f) - x_lambda0;

        int in_idx_base = (b)*INPUT0_PITCHES[0] + (c)*INPUT0_PITCHES[1];
        int in_idx00 = in_idx_base + (iy0)*INPUT0_PITCHES[2] + (ix0)*INPUT0_PITCHES[3];
        int in_idx01 = in_idx_base + (iy0)*INPUT0_PITCHES[2] + (ix1)*INPUT0_PITCHES[3];
        int in_idx10 = in_idx_base + (iy1)*INPUT0_PITCHES[2] + (ix0)*INPUT0_PITCHES[3];
        int in_idx11 = in_idx_base + (iy1)*INPUT0_PITCHES[2] + (ix1)*INPUT0_PITCHES[3];

        dst[out_idx] = (OUTPUT0_TYPE)(y_lambda1 * (x_lambda1 * src[in_idx00] + x_lambda0 * src[in_idx01]) +
                                      y_lambda0 * (x_lambda1 * src[in_idx10] + x_lambda0 * src[in_idx11]));
    }
}


__kernel void warp_affine(const __global INPUT0_TYPE*  input,
                          const __global INPUT1_TYPE*  matrix,
                           __global OUTPUT0_TYPE* output)
{
    int IB = OUTPUT0_DIMS[0];
    int IF = OUTPUT0_DIMS[1];
    int OY = OUTPUT0_DIMS[2];
    int OX = OUTPUT0_DIMS[3];
    int IY = INPUT0_DIMS[2];
    int IX = INPUT0_DIMS[3];

    warpAffine(IB, IF, input, IY, IX, output, OY, OX, matrix);
}
