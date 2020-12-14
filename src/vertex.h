#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>

struct Vertex {
    glm::vec4 position = glm::vec4(0.f, 0.f, 0.f, 1.f);
    glm::vec4 velocity = glm::vec4(0.f, 0.f, 0.f, 1.f);
    glm::vec4 attr1 = glm::vec4(0.05f, 0.0125f, -1.f, 1.f);;  // radius, mass, isFixed, snowPortion
    glm::vec4 attr2 = glm::vec4(-1.f, -1.f, 1.f, 1.f);;  // neighborMax, hasBrokenBond, d, (null)
    glm::vec4 color = glm::vec4(1.f, 1.f, 1.f, 1.f);

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 5> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 5> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, velocity);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[2].offset = offsetof(Vertex, attr1);

        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[3].offset = offsetof(Vertex, attr2);

        attributeDescriptions[4].binding = 0;
        attributeDescriptions[4].location = 4;
        attributeDescriptions[4].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[4].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }

    bool operator==(const Vertex& other) const {
        return position == other.position && 
            velocity == other.velocity && 
            attr1 == other.attr1 && 
            attr2 == other.attr2 &&
            color == other.color;
    }
};

//namespace std {
//    template<> struct std::hash<Vertex> {
//        size_t operator()(Vertex const& vertex) const {
//            return ((std::hash<glm::vec3>()(vertex.pos) ^
//                (std::hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
//                (std::hash<glm::vec2>()(vertex.texCoord) << 1);
//        }
//    };
//}
