#pragma once
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>  // exposes functions to do precise timekeeping
#include <optional>
#include <vector>
#include "../external/tiny_obj_loader.h"
#include "../external/stb_image.h"

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};


const int N_GRID_CELLS = 1000000;
const int N_SIDE = 30;
const int N_FOR_VIS = N_SIDE * N_SIDE * N_SIDE;
const float DT = 0.0017f;
const int WIDTH = 1000;
const int HEIGHT = 800;
const float BALL_SCALE_FACTOR = 0.01f;
const glm::mat4 BALL_SCALE_MAT = glm::scale(glm::mat4(1), glm::vec3(BALL_SCALE_FACTOR, BALL_SCALE_FACTOR, BALL_SCALE_FACTOR));

const int MAX_FRAMES_IN_FLIGHT = 2;
const std::string BALL_PATH = "../assets/models/sphere.obj";
const std::string TEXTURE_PATH = "../assets/images/viking_room.png";
const std::string MODEL_PATH = "../assets/models/viking_room.obj";

bool leftMouseDown = false;
bool rightMouseDown = false;
double previousX = 0.0;
double previousY = 0.0;

float r = 10.0f;
float theta = 1.0f;
float phi = -0.7f;

glm::vec3 eye = glm::vec3(5.0f, 5.0f, r);
glm::mat4 viewMat = glm::lookAt(eye, glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 1.0f, 0.0f));

static void updateOrbit(float deltaX, float deltaY, float deltaZ) {
    theta += deltaX;
    phi += deltaY;
    r = glm::clamp(r - deltaZ, 1.0f, 50.0f);

    float radTheta = glm::radians(theta);
    float radPhi = glm::radians(phi);

    glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), radTheta, glm::vec3(0.0f, 1.0f, 0.0f)) * glm::rotate(glm::mat4(1.0f), radPhi, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 finalTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f)) * rotation * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.0f, r));
    viewMat = glm::inverse(finalTransform);
    //update pos
    eye = glm::vec3(-viewMat[3][0], -viewMat[3][1], -viewMat[3][2]);
}