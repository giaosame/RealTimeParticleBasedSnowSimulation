#pragma once
#include <vector>
#include "vertex.h"

namespace PointsGenerator {
    void createCube(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, int& idxForWholeVertices,
                    const int N_SIDE, const glm::vec3& OFFSET, const glm::vec3& COLOR)
    {
        float l = (float)N_SIDE / 10.f;

        int idx = idxForWholeVertices;
        for (int i = 0; i < N_SIDE; ++i)
        {
            for (int j = 0; j < N_SIDE; ++j)
            {
                for (int k = 0; k < N_SIDE; ++k)
                {
                    glm::vec3 position;
                    position = glm::vec3(k * l / (float)N_SIDE,
                        j * l / (float)N_SIDE,
                        i * l / (float)N_SIDE);
                    position += OFFSET;

                    Vertex v;
                    v.position = glm::vec4(position, 1.f);
                    v.color = glm::vec4(COLOR, 1.f);
                    verts.push_back(v);
                    indices.push_back(idx);
                    idx++;
                }
            }
        }

        idxForWholeVertices = idx;
    }

    // 14112 points when N_SIDE = 30
    void createSphere(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, int& idxForWholeVertices,
                      const int N_SIDE, const glm::vec3& OFFSET, const glm::vec3& COLOR, 
                      const glm::vec3& InitialVel = glm::vec3(0.f))
    {
        float l = float(N_SIDE) / 10.f;
        const float MID = float(N_SIDE) / 2;

        int idx = idxForWholeVertices;
        for (int i = 0; i < N_SIDE; ++i)  // z
        {
            for (int j = 0; j < N_SIDE; ++j)  // y
            {
                float y = j * l / float(N_SIDE);
                float mid_l = MID * l / float(N_SIDE);
                float dy = j < MID ? mid_l - y : y - mid_l;
                float r2 = mid_l * mid_l - dy * dy;
                glm::vec3 center = glm::vec3(mid_l, y, mid_l) + OFFSET;
                for (int k = 0; k < N_SIDE; ++k)  // x
                {
                    glm::vec3 position;
                    position = glm::vec3(k * l / float(N_SIDE),
                        y,
                        i * l / float(N_SIDE));
                    position += OFFSET;

                    glm::vec3 diff = position - center;
                    if (diff.x * diff.x + diff.z * diff.z > r2)
                        continue;

                    Vertex v;
                    v.position = glm::vec4(position, 1.f);
                    v.color = glm::vec4(COLOR, 1.f);
                    v.velocity = glm::vec4(InitialVel, 1.f);
                    verts.push_back(v);
                    indices.push_back(idx);
                    idx++;
                }
            }
        }

        idxForWholeVertices = idx;
    }

    bool inTanglecube(const glm::vec3& p) {
        const float x2 = p.x * p.x;
        const float y2 = p.y * p.y;
        const float z2 = p.z * p.z;
        return x2* x2 - 5.f * x2 + y2 * y2 - 5.f * y2 + z2 * z2 - 5.f * z2 + 11.8f <= 0;
    }

    void createTanglecube(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, int& idxForWholeVertices,
                          const int N_SIDE, const glm::vec3& OFFSET, const glm::vec3& COLOR)
    {
        float l = float(N_SIDE) / 10.f;
        const float MID = float(N_SIDE) / 2;
        const float MID_L = MID * l / float(N_SIDE);
        const glm::vec3 center = glm::vec3(MID_L, MID_L, MID_L);

        int idx = idxForWholeVertices;
        for (int i = 0; i < N_SIDE; ++i)  // z
        {
            float z = i * l / float(N_SIDE);
            for (int j = 0; j < N_SIDE; ++j)  // y
            {
                float y = j * l / float(N_SIDE);
                for (int k = 0; k < N_SIDE; ++k)  // x
                {
                    glm::vec3 position = glm::vec3(k * l / float(N_SIDE), y, z);
                    glm::vec3 diff = position - center;

                    if (!inTanglecube(diff))
                        continue;

                    Vertex v;
                    v.position = glm::vec4(position + OFFSET, 1.f);
                    v.color = glm::vec4(COLOR, 1.f);
                    verts.push_back(v);
                    indices.push_back(idx);
                    idx++;
                }
            }
        }

        idxForWholeVertices = idx;
    }

    bool inHeart(const glm::vec3& p) {
        const float h = 1.0f, r1 = 1.0f, r2 = 0.0f;
        const float x2 = p.x * p.x;
        const float y2 = p.y * p.y;
        const float z2 = p.z * p.z;
        const float z3 = p.z * z2;
        const float temp = x2 + 9.f * y2 / 4.f + z2 - 1.f;
        return temp * temp * temp - x2 * z3 - 9.f * y2 * z3 / 80.f <= 0;
    }

    void createHeart(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, int& idxForWholeVertices,
                     const int N_SIDE, const glm::vec3& OFFSET, const glm::vec3& COLOR)
    {
        float l = float(N_SIDE) / 10.f;
        const float MID = float(N_SIDE) / 2;
        const float MID_L = MID * l / float(N_SIDE);
        const glm::vec3 center = glm::vec3(MID_L, MID_L, MID_L);

        int idx = idxForWholeVertices;
        for (int i = 0; i < N_SIDE; ++i)  // z
        {
            float z = i * l / float(N_SIDE);
            for (int j = 0; j < N_SIDE; ++j)  // y
            {
                float y = j * l / float(N_SIDE);
                for (int k = 0; k < N_SIDE; ++k)  // x
                {
                    glm::vec3 position = glm::vec3(k * l / float(N_SIDE), y, z);
                    glm::vec3 diff = position - center;

                    if (!inHeart(diff))
                        continue;

                    Vertex v;
                    position = glm::vec3(position.y, position.z, position.x);
                    v.position = glm::vec4(position + OFFSET, 1.f);
                    v.color = glm::vec4(COLOR, 1.f);
                    verts.push_back(v);
                    indices.push_back(idx);
                    idx++;
                }
            }
        }

        idxForWholeVertices = idx;
    }

    bool inTorus(const glm::vec3& p) {
        const float r1 = 1.0f, r2 = 0.5f;
        const float q = sqrt(p.x * p.x + p.z * p.z) - r1;
        const float len = sqrt(q * q + p.y * p.y);
        return len - r2 <= 0;
    }

    void createTorus(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, int& idxForWholeVertices,
                     const int N_SIDE, const glm::vec3& OFFSET, const glm::vec3& COLOR)
    {
        float l = float(N_SIDE) / 10.f;
        const float MID = float(N_SIDE) / 2;
        const float MID_L = MID * l / float(N_SIDE);
        const glm::vec3 center = glm::vec3(MID_L, MID_L, MID_L);

        int idx = idxForWholeVertices;
        for (int i = 0; i < N_SIDE; ++i)  // z
        {
            float z = i * l / float(N_SIDE);
            for (int j = 0; j < N_SIDE; ++j)  // y
            {
                float y = j * l / float(N_SIDE);
                for (int k = 0; k < N_SIDE; ++k)  // x
                {
                    glm::vec3 position = glm::vec3(k * l / float(N_SIDE), y, z);
                    glm::vec3 diff = position - center;

                    if (!inTorus(diff))
                        continue;

                    Vertex v;
                    position = glm::vec3(position.y, position.z, position.x);
                    v.color = glm::vec4(COLOR, 1.f);
                    v.position = glm::vec4(position + OFFSET, 1.f);
                    verts.push_back(v);
                    indices.push_back(idx);
                    idx++;
                }
            }
        }
        idxForWholeVertices = idx;
    }
}