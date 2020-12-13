#pragma once
#include <vector>
#include "vertex.h"

namespace PointsGenerator {
	void createCube(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, const int N_SIDE) {
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
                    position += glm::vec3(0.05f, 0.05f, 0.05f);

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