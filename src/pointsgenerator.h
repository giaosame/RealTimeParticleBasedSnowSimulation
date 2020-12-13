#pragma once
#include <vector>
#include "vertex.h"

namespace PointsGenerator {
    void createCube(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, const int N_SIDE, const glm::vec3& OFFSET) {
        float l = (float)N_SIDE / 10.f;

        int idx = 0;
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
                    verts.push_back(v);
                    indices.push_back(idx);
                    idx++;
                }
            }
        }
    }

    // 14112 points when N_SIDE = 30
    void createSphere(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, const int N_SIDE, const glm::vec3& OFFSET) {
        float l = float(N_SIDE) / 10.f;
        const float MID = float(N_SIDE) / 2;

        int idx = 0;
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
                    verts.push_back(v);
                    indices.push_back(idx);
                    idx++;
                }
            }
        }
    }
}