/* Derived from Mesh2Splat by Electronic Arts Inc.
 * Original: Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
 * Licensed under BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
 *
 * Modifications: Copyright (c) 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#version 430 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform vec2 metallicRoughnessFactors;
uniform vec3 u_bboxMin;
uniform vec3 u_bboxMax;


in VS_OUT{
    vec3 position;
    vec3 normal;
    vec4 tangent;
    vec2 uv;
    vec2 normalizedUv;
    vec3 scale;
    vec4 vertexColor;
} gs_in[];

out vec3 Position;
flat out vec3 Scale;
out vec2 UV;
out vec4 Tangent;
out vec4 VertexColor;
out vec3 Normal;
flat out vec4 Quaternion;

void transpose2x3(in mat2x3 m, out mat3x2 outputMat) {
    outputMat[0][0] = m[0][0];
    outputMat[1][0] = m[0][1];
    outputMat[2][0] = m[0][2];

    outputMat[0][1] = m[1][0];
    outputMat[1][1] = m[1][1];
    outputMat[2][1] = m[1][2];
}

vec4 quat_cast(mat3 m) {
    float fourXSquaredMinus1 = m[0][0] - m[1][1] - m[2][2];
    float fourYSquaredMinus1 = m[1][1] - m[0][0] - m[2][2];
    float fourZSquaredMinus1 = m[2][2] - m[0][0] - m[1][1];
    float fourWSquaredMinus1 = m[0][0] + m[1][1] + m[2][2];

    int biggestIndex = 0;
    float fourBiggestSquaredMinus1 = fourWSquaredMinus1;
    if (fourXSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourXSquaredMinus1;
        biggestIndex = 1;
    }
    if (fourYSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourYSquaredMinus1;
        biggestIndex = 2;
    }
    if (fourZSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourZSquaredMinus1;
        biggestIndex = 3;
    }

    float biggestVal = sqrt(fourBiggestSquaredMinus1 + 1.0f) * 0.5f;
    float mult = 0.25f / biggestVal;

    vec4 q;

    if (biggestIndex == 0) {
        q.w = biggestVal;
        q.x = (m[1][2] - m[2][1]) * mult;
        q.y = (m[2][0] - m[0][2]) * mult;
        q.z = (m[0][1] - m[1][0]) * mult;
    }
    else if (biggestIndex == 1) {
        q.w = (m[1][2] - m[2][1]) * mult;
        q.x = biggestVal;
        q.y = (m[0][1] + m[1][0]) * mult;
        q.z = (m[2][0] + m[0][2]) * mult;
    }
    else if (biggestIndex == 2) {
        q.w = (m[2][0] - m[0][2]) * mult;
        q.x = (m[0][1] + m[1][0]) * mult;
        q.y = biggestVal;
        q.z = (m[1][2] + m[2][1]) * mult;
    }
    else {
        q.w = (m[0][1] - m[1][0]) * mult;
        q.x = (m[2][0] + m[0][2]) * mult;
        q.y = (m[1][2] + m[2][1]) * mult;
        q.z = biggestVal;
    }

    return q;
}

mat2 inverse2x2(mat2 m) {
    float determinant = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if (determinant == 0.0) {
        return mat2(0.0); // Handle non-invertible case if necessary
    }
    float invDet = 1.0 / determinant;
    mat2 inverse;
    inverse[0][0] = m[1][1] * invDet;
    inverse[1][0] = -m[1][0] * invDet;

    inverse[0][1] = -m[0][1] * invDet;
    inverse[1][1] = m[0][0] * invDet;

    return inverse;
}

mat2x3 multiplyMat2x3WithMat2x2(mat2x3 matA, mat2 matB) {
    mat2x3 result;

    result[0][0] = matA[0][0] * matB[0][0] + matA[1][0] * matB[0][1];
    result[1][0] = matA[0][0] * matB[1][0] + matA[1][0] * matB[1][1];

    result[0][1] = matA[0][1] * matB[0][0] + matA[1][1] * matB[0][1];
    result[1][1] = matA[0][1] * matB[1][0] + matA[1][1] * matB[1][1];

    result[0][2] = matA[0][2] * matB[0][0] + matA[1][2] * matB[0][1];
    result[1][2] = matA[0][2] * matB[1][0] + matA[1][2] * matB[1][1];

    return result;
}

mat2x3 computeUv3DJacobian(vec3 verticesTriangle3D[3], vec2 verticesTriangleUV[3]) {
    // 3D positions of the triangle's vertices
    vec3 pos0 = verticesTriangle3D[0];
    vec3 pos1 = verticesTriangle3D[1];
    vec3 pos2 = verticesTriangle3D[2];

    // UV coordinates of the triangle's vertices
    vec2 uv0 = verticesTriangleUV[0];
    vec2 uv1 = verticesTriangleUV[1];
    vec2 uv2 = verticesTriangleUV[2];

    mat2 UVMatrix;
    UVMatrix[0][0] = uv1.x - uv0.x;
    UVMatrix[1][0] = uv2.x - uv0.x;

    UVMatrix[0][1] = uv1.y - uv0.y;
    UVMatrix[1][1] = uv2.y - uv0.y;


    mat2x3 VMatrix;
    VMatrix[0][0] = pos1.x - pos0.x;
    VMatrix[1][0] = pos2.x - pos0.x;

    VMatrix[0][1] = pos1.y - pos0.y;
    VMatrix[1][1] = pos2.y - pos0.y;

    VMatrix[0][2] = pos1.z - pos0.z;
    VMatrix[1][2] = pos2.z - pos0.z;

    // Compute the Jacobian matrix
    return multiplyMat2x3WithMat2x2(VMatrix, inverse2x2(UVMatrix));
}

void main() {
    vec3 edge1 = gs_in[1].position - gs_in[0].position;
    vec3 edge2 = gs_in[2].position - gs_in[0].position;
    vec3 edge3 = gs_in[2].position - gs_in[1].position;

    vec3 bboxSize = u_bboxMax - u_bboxMin;

    if (length(edge2) > length(edge1) && length(edge2) > length(edge3)) {
        vec3 temp = edge1;
        edge1 = edge2;
        edge2 = temp;
    }
    else if (length(edge3) > length(edge1) && length(edge3) > length(edge2)) {
        vec3 temp = edge1;
        edge1 = edge3;
        edge3 = temp;
    }

    // Normalize the edge vectors
    edge1 = normalize(edge1);
    
    vec3 normal = normalize(cross(edge1, edge2));

    float absX = abs(normal.x);
    float absY = abs(normal.y);
    float absZ = abs(normal.z);

    vec2 orthogonalUvs[3];
    //longest dimension to 0-1, and scaling down the smaller one proportionally
    for (int i = 0; i < 3; i++)
    {
        float u, v;
        vec3 pos = gs_in[i].position;

        if (absX > absY && absX > absZ)
        {
            float rangeY = u_bboxMax.y - u_bboxMin.y;
            float rangeZ = u_bboxMax.z - u_bboxMin.z;
            float range  = max(rangeY, rangeZ);

            float relY = pos.y - u_bboxMin.y;
            float relZ = pos.z - u_bboxMin.z;

            u = relY / range;  
            v = relZ / range;

        }
        else if (absY > absZ)
        {
            float rangeX = u_bboxMax.x - u_bboxMin.x;
            float rangeZ = u_bboxMax.z - u_bboxMin.z;
            float range  = max(rangeX, rangeZ);

            float relX = pos.x - u_bboxMin.x;
            float relZ = pos.z - u_bboxMin.z;

            u = relX / range;
            v = relZ / range;
        }
        else
        {
            float rangeX = u_bboxMax.x - u_bboxMin.x;
            float rangeY = u_bboxMax.y - u_bboxMin.y;
            float range  = max(rangeX, rangeY);

            float relX = pos.x - u_bboxMin.x;
            float relY = pos.y - u_bboxMin.y;

            u = relX / range;
            v = relY / range;
        }

        orthogonalUvs[i] = vec2(u, v);
    }

    vec3 xAxis = edge1;
    vec3 yAxis = normalize(cross(normal, xAxis));
    vec3 zAxis = normal;

    mat3 rotationMatrix = mat3(xAxis, yAxis, zAxis);
    vec4 q = quat_cast(rotationMatrix);
    vec4 quaternion = vec4(q.w, q.x, q.y, q.z);

    mat2x3 J;

    vec3 true_vertices3D[3] = { gs_in[0].position, gs_in[1].position, gs_in[2].position };
    vec2 true_normalized_vertices2D[3] = { orthogonalUvs[0], orthogonalUvs[1], orthogonalUvs[2] };

    J = computeUv3DJacobian(true_vertices3D, true_normalized_vertices2D);
      
    mat3x2 J_T;
    transpose2x3(J, J_T);

    vec3 Ju = vec3(J_T[0][0], J_T[1][0], J_T[2][0]); 
    vec3 Jv = vec3(J_T[0][1], J_T[1][1], J_T[2][1]); 

    float gaussian_scale_x = length(Ju);
    float gaussian_scale_y = length(Jv);

    float packed_s_x    = gaussian_scale_x;
    float packed_s_y    = gaussian_scale_y;
    float packed_s_z    = 1e-7;

    Scale = vec3(packed_s_x, packed_s_y, packed_s_z);

    for (int i = 0; i < 3; i++)
    {
        Tangent                 = gs_in[i].tangent;
        Position                = gs_in[i].position;
        Normal                  = gs_in[i].normal;
        UV                      = gs_in[i].uv;
        VertexColor             = gs_in[i].vertexColor;
        Quaternion              = quaternion;
        gl_Position             = vec4(orthogonalUvs[i] * 2.0 - 1.0, 0.0, 1.0);
        EmitVertex();
    }
    EndPrimitive();
}
