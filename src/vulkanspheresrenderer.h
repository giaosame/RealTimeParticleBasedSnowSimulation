#include "utils.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <array>
#include <optional>
#include <set>

#include "vertex.h"
#include "pointsgenerator.h"


class VulkanSpheresRenderer {
public:
    void run() {
        initWindow();
        initParticles();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    vk::UniqueInstance instance;
    // VkDebugUtilsMessengerEXT callback;
    vk::SurfaceKHR surface;

    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;

    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::Queue computeQueue;

    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::ImageView> swapChainImageViews;
    std::vector<vk::Framebuffer> swapChainFramebuffers;

    // Texture image and view
    vk::Image textureImage;
    vk::DeviceMemory textureImageMemory;
    vk::ImageView textureImageView;
    vk::Sampler textureSampler;

    // Depth image and view
    vk::Image depthImage;
    vk::DeviceMemory depthImageMemory;
    vk::ImageView depthImageView;

    vk::RenderPass renderPass;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;

    vk::CommandPool commandPool;

    std::vector<Vertex> raw_verts;
    std::vector<uint32_t> raw_indices;
    std::vector<Vertex> sphere_verts;
    std::vector<uint32_t> sphere_indices;
    // Vertex* raw_verts = new Vertex[N_FOR_VIS];
    // uint32_t* raw_indices = new uint32_t[N_FOR_VIS];

    int* cellVertArray = new int[N_GRID_CELLS * 6]{ 0 };
    int* cellVertCount = new int[N_GRID_CELLS]{ 0 };

    std::vector<Vertex> model_verts;
    std::vector<uint32_t> model_indices;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // Old vertices data
    vk::Buffer vertexBuffer1;
    vk::DeviceMemory vertexBufferMemory1;

    // New vertices data
    vk::Buffer vertexBuffer2;
    vk::DeviceMemory vertexBufferMemory2;

    vk::Buffer numVertsBuffer;
    vk::DeviceMemory numVertsBufferMemory;

    // storge buffer 
    vk::Buffer cellVertArrayBuffer;
    vk::DeviceMemory cellVertArrayBufferMemory;
    vk::Buffer cellVertCountBuffer;
    vk::DeviceMemory cellVertCountBufferMemory;

    // sphere vertices storge buffer
    vk::Buffer sphereVertsBuffer;
    vk::DeviceMemory sphereVertsBufferMemory;

    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;

    std::vector<vk::Buffer> uniformUboBuffers;
    std::vector<vk::DeviceMemory> uniformUboBuffersMemory;

    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;
    std::vector<vk::CommandBuffer, std::allocator<vk::CommandBuffer>> commandBuffers;

    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    size_t currentFrame = 0;

    // for compute pipeline
    vk::PipelineLayout computePipelineLayout;
    vk::Pipeline computePipelinePhysics;
    vk::Pipeline computePipelineFillCellVertex;
    vk::Pipeline computePipelineResetCellVertex;
    vk::Pipeline computePipelineSphereVertex;
    vk::DescriptorSetLayout computeDescriptorSetLayout;
    vk::DescriptorPool computeDescriptorPool;
    std::vector<vk::DescriptorSet> computeDescriptorSet;

    bool framebufferResized = false;

    //call back for the mouse down
    static void mouseDownCallback(GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                leftMouseDown = true;
                glfwGetCursorPos(window, &previousX, &previousY);
            }
            else if (action == GLFW_RELEASE) {
                leftMouseDown = false;
            }
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                rightMouseDown = true;
                glfwGetCursorPos(window, &previousX, &previousY);
            }
            else if (action == GLFW_RELEASE) {
                rightMouseDown = false;
            }
        }
    }


    static void mouseMoveCallback(GLFWwindow* window, double xPosition, double yPosition) {
        if (leftMouseDown) {
            double sensitivity = 0.5;
            float deltaX = static_cast<float>((previousX - xPosition) * sensitivity);
            float deltaY = static_cast<float>((previousY - yPosition) * sensitivity);

            updateOrbit(deltaX, deltaY, 0.0f);

            previousX = xPosition;
            previousY = yPosition;
        }
        else if (rightMouseDown) {
            double deltaZ = static_cast<float>((previousY - yPosition) * 0.05);

            updateOrbit(0.0f, 0.0f, deltaZ);
            previousY = yPosition;
        }
    }

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Snow Simulator", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

        glfwSetMouseButtonCallback(window, mouseDownCallback);
        glfwSetCursorPosCallback(window, mouseMoveCallback);
        updateOrbit(0.0f, 0.0f, 0.0f);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<VulkanSpheresRenderer*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initParticles() {
        loadModel();

        int idxForWholeVertices = 0;
        const glm::vec3 OFFSET(0.05f, 0.05f, 0.05f);
        PointsGenerator::createSphere(raw_verts, raw_indices, idxForWholeVertices, 30, OFFSET, glm::vec3(1.f, 1.f, 1.f));
        // std::cout << "Number of vertices: " << raw_verts.size() << std::endl;

        int sphereIdx = 0;
        for (int i = 0; i < raw_verts.size(); i++) {// 27000
            auto translation = raw_verts[i].position;
            translation.w = 0.f;
            for (int j = 0; j < model_verts.size(); j++) { // 180
                Vertex v = model_verts[j];
                v.position += translation;
                sphere_verts.push_back(v);
                sphere_indices.push_back(sphereIdx);
                sphereIdx++;
            }
        }
        std::cout << "Number of raw_verts: " << raw_verts.size() << std::endl;
        std::cout << "Number of sphereIdx: " << sphereIdx << std::endl;
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createDepthResources();
        createFramebuffers();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();

        // loadModel();
        createVertexBuffers();
        createIndexBuffer();
        createNumVertsBuffer(); 

        createCellVertArrayBuffer();
        createCellVertCountBuffer();
        createSphereVertsBuffer();


        createComputePipeline("../src/shaders/physicsCompute.spv", computePipelinePhysics);
        createComputePipeline("../src/shaders/fillCellVertexInfo.spv", computePipelineFillCellVertex);
        createComputePipeline("../src/shaders/resetCellVertexInfo.spv", computePipelineResetCellVertex); 
        createComputePipeline("../src/shaders/sphereVertexCompute.spv", computePipelineSphereVertex);

        createuniformUboBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void createComputePipeline(std::string computeShaderPath, vk::Pipeline& pipelineIdx)
    {
        auto computeShaderCode = readFile(computeShaderPath);
        auto computeShaderModule = createShaderModule(computeShaderCode);

        vk::PipelineShaderStageCreateInfo computeShaderStageInfo = {};
        computeShaderStageInfo.flags = vk::PipelineShaderStageCreateFlags();
        computeShaderStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
        computeShaderStageInfo.module = *computeShaderModule;
        computeShaderStageInfo.pName = "main";

        vk::DescriptorSetLayoutBinding computeLayoutBindingVertices1{};
        computeLayoutBindingVertices1.binding = 0;
        computeLayoutBindingVertices1.descriptorCount = 1;
        computeLayoutBindingVertices1.descriptorType = vk::DescriptorType::eStorageBuffer;
        computeLayoutBindingVertices1.pImmutableSamplers = nullptr;
        computeLayoutBindingVertices1.stageFlags = vk::ShaderStageFlagBits::eCompute;

        vk::DescriptorSetLayoutBinding computeLayoutBindingVertices2{};
        computeLayoutBindingVertices2.binding = 1;
        computeLayoutBindingVertices2.descriptorCount = 1;
        computeLayoutBindingVertices2.descriptorType = vk::DescriptorType::eStorageBuffer;
        computeLayoutBindingVertices2.pImmutableSamplers = nullptr;
        computeLayoutBindingVertices2.stageFlags = vk::ShaderStageFlagBits::eCompute;

        vk::DescriptorSetLayoutBinding computeLayoutBindingCellVertexArray{};
        computeLayoutBindingCellVertexArray.binding = 2;
        computeLayoutBindingCellVertexArray.descriptorCount = 1;
        computeLayoutBindingCellVertexArray.descriptorType = vk::DescriptorType::eStorageBuffer;
        computeLayoutBindingCellVertexArray.pImmutableSamplers = nullptr;
        computeLayoutBindingCellVertexArray.stageFlags = vk::ShaderStageFlagBits::eCompute;

        vk::DescriptorSetLayoutBinding computeLayoutBindingCellVertexCount{};
        computeLayoutBindingCellVertexCount.binding = 3;
        computeLayoutBindingCellVertexCount.descriptorCount = 1;
        computeLayoutBindingCellVertexCount.descriptorType = vk::DescriptorType::eStorageBuffer;
        computeLayoutBindingCellVertexCount.pImmutableSamplers = nullptr;
        computeLayoutBindingCellVertexCount.stageFlags = vk::ShaderStageFlagBits::eCompute;

        vk::DescriptorSetLayoutBinding computeLayoutBindingNumVerts{};
        computeLayoutBindingNumVerts.binding = 4;
        computeLayoutBindingNumVerts.descriptorCount = 1;
        computeLayoutBindingNumVerts.descriptorType = vk::DescriptorType::eStorageBuffer;
        computeLayoutBindingNumVerts.pImmutableSamplers = nullptr;
        computeLayoutBindingNumVerts.stageFlags = vk::ShaderStageFlagBits::eCompute;

        vk::DescriptorSetLayoutBinding computeLayoutBindingSphereVerts{};
        computeLayoutBindingSphereVerts.binding = 5;
        computeLayoutBindingSphereVerts.descriptorCount = 1;
        computeLayoutBindingSphereVerts.descriptorType = vk::DescriptorType::eStorageBuffer;
        computeLayoutBindingSphereVerts.pImmutableSamplers = nullptr;
        computeLayoutBindingSphereVerts.stageFlags = vk::ShaderStageFlagBits::eCompute;

        std::vector<vk::DescriptorSetLayoutBinding> bindings = { computeLayoutBindingVertices1, computeLayoutBindingVertices2,
            computeLayoutBindingCellVertexArray, computeLayoutBindingCellVertexCount, computeLayoutBindingNumVerts, computeLayoutBindingSphereVerts };

        vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        //descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.flags = vk::DescriptorSetLayoutCreateFlags();
        descriptorSetLayoutCreateInfo.pNext = nullptr;
        descriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        descriptorSetLayoutCreateInfo.pBindings = bindings.data();

        try {
            computeDescriptorSetLayout = device->createDescriptorSetLayout(descriptorSetLayoutCreateInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }

        //VkDescriptorPoolSize poolSizes[1];
        std::array<vk::DescriptorPoolSize, 1> poolSizes{};
        poolSizes[0].type = vk::DescriptorType::eStorageBuffer;
        poolSizes[0].descriptorCount = 10;

        vk::DescriptorPoolCreateInfo descriptorPoolInfo = {};
        //descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolInfo.flags = vk::DescriptorPoolCreateFlags();
        descriptorPoolInfo.pNext = nullptr;
        descriptorPoolInfo.poolSizeCount = 1;
        descriptorPoolInfo.pPoolSizes = poolSizes.data();
        descriptorPoolInfo.maxSets = 1;

        try {
            computeDescriptorPool = device->createDescriptorPool(descriptorPoolInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create compute descriptor pool!");
        }

        vk::DescriptorSetAllocateInfo allocInfo = {};
        // allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = computeDescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &computeDescriptorSetLayout;

        //computeDescriptorSet.resize(1);
        try {
            computeDescriptorSet = device->allocateDescriptorSets(allocInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create compute descriptor sets!");
        }

        // Set descriptor set for the old vertices
        vk::DescriptorBufferInfo computeBufferInfoVertices1 = {};
        computeBufferInfoVertices1.buffer = vertexBuffer1;
        computeBufferInfoVertices1.offset = 0;
        computeBufferInfoVertices1.range = static_cast<uint32_t>(raw_verts.size() * sizeof(Vertex));

        vk::WriteDescriptorSet writeComputeInfoVertices1 = {};
        writeComputeInfoVertices1.dstSet = computeDescriptorSet[0];
        writeComputeInfoVertices1.dstBinding = 0;
        writeComputeInfoVertices1.descriptorCount = 1;
        writeComputeInfoVertices1.dstArrayElement = 0;
        writeComputeInfoVertices1.descriptorType = vk::DescriptorType::eStorageBuffer;
        writeComputeInfoVertices1.pBufferInfo = &computeBufferInfoVertices1;

        // Set descriptor set for the new vertices
        vk::DescriptorBufferInfo computeBufferInfoVertices2 = {};
        computeBufferInfoVertices2.buffer = vertexBuffer2;
        computeBufferInfoVertices2.offset = 0;
        computeBufferInfoVertices2.range = static_cast<uint32_t>(raw_verts.size() * sizeof(Vertex));

        vk::WriteDescriptorSet writeComputeInfoVertices2 = {};
        writeComputeInfoVertices2.dstSet = computeDescriptorSet[0];
        writeComputeInfoVertices2.dstBinding = 1;
        writeComputeInfoVertices2.descriptorCount = 1;
        writeComputeInfoVertices2.dstArrayElement = 0;
        writeComputeInfoVertices2.descriptorType = vk::DescriptorType::eStorageBuffer;
        writeComputeInfoVertices2.pBufferInfo = &computeBufferInfoVertices2;

        // Set descriptor set for the cell vertex array 
        vk::DescriptorBufferInfo computeBufferInfoCellVertexArray = {};
        computeBufferInfoCellVertexArray.buffer = cellVertArrayBuffer;
        computeBufferInfoCellVertexArray.offset = 0;
        computeBufferInfoCellVertexArray.range = static_cast<uint32_t>(N_GRID_CELLS * 6 * sizeof(int));

        vk::WriteDescriptorSet writeComputeInfoCellVertexArray = {};
        writeComputeInfoCellVertexArray.dstSet = computeDescriptorSet[0];
        writeComputeInfoCellVertexArray.dstBinding = 2;
        writeComputeInfoCellVertexArray.descriptorCount = 1;
        writeComputeInfoCellVertexArray.dstArrayElement = 0;
        writeComputeInfoCellVertexArray.descriptorType = vk::DescriptorType::eStorageBuffer;
        writeComputeInfoCellVertexArray.pBufferInfo = &computeBufferInfoCellVertexArray;

        // Set descriptor set for the cell vertex count 
        vk::DescriptorBufferInfo computeBufferInfoCellVertexCount = {};
        computeBufferInfoCellVertexCount.buffer = cellVertCountBuffer;
        computeBufferInfoCellVertexCount.offset = 0;
        computeBufferInfoCellVertexCount.range = static_cast<uint32_t>(N_GRID_CELLS * sizeof(int));

        vk::WriteDescriptorSet writeComputeInfoCellVertexCount = {};
        writeComputeInfoCellVertexCount.dstSet = computeDescriptorSet[0];
        writeComputeInfoCellVertexCount.dstBinding = 3;
        writeComputeInfoCellVertexCount.descriptorCount = 1;
        writeComputeInfoCellVertexCount.dstArrayElement = 0;
        writeComputeInfoCellVertexCount.descriptorType = vk::DescriptorType::eStorageBuffer;
        writeComputeInfoCellVertexCount.pBufferInfo = &computeBufferInfoCellVertexCount;

        // Set descriptor set for the buffer representing the number of vertices
        vk::DescriptorBufferInfo computeBufferInfoNumVerts = {};
        computeBufferInfoNumVerts.buffer = numVertsBuffer;
        computeBufferInfoNumVerts.offset = 0;
        computeBufferInfoNumVerts.range = static_cast<uint32_t>(sizeof(int));  

        vk::WriteDescriptorSet writeComputeInfoNumVerts = {};
        writeComputeInfoNumVerts.dstSet = computeDescriptorSet[0];
        writeComputeInfoNumVerts.dstBinding = 4;
        writeComputeInfoNumVerts.descriptorCount = 1;
        writeComputeInfoNumVerts.dstArrayElement = 0;
        writeComputeInfoNumVerts.descriptorType = vk::DescriptorType::eStorageBuffer;
        writeComputeInfoNumVerts.pBufferInfo = &computeBufferInfoNumVerts;

        // Set descriptor set for the buffer representing the sphere vertices
        vk::DescriptorBufferInfo computeBufferInfoSphereVerts = {};
        computeBufferInfoSphereVerts.buffer = sphereVertsBuffer;
        computeBufferInfoSphereVerts.offset = 0;
        computeBufferInfoSphereVerts.range = static_cast <uint32_t>(sizeof(Vertex) * sphere_verts.size());

        vk::WriteDescriptorSet writeComputeInfoSphereVerts = {};
        writeComputeInfoSphereVerts.dstSet = computeDescriptorSet[0];
        writeComputeInfoSphereVerts.dstBinding = 5;
        writeComputeInfoSphereVerts.descriptorCount = 1;
        writeComputeInfoSphereVerts.dstArrayElement = 0;
        writeComputeInfoSphereVerts.descriptorType = vk::DescriptorType::eStorageBuffer;
        writeComputeInfoSphereVerts.pBufferInfo = &computeBufferInfoSphereVerts;

        std::array<vk::WriteDescriptorSet, 6> writeDescriptorSets = { writeComputeInfoVertices1, writeComputeInfoVertices2, 
            writeComputeInfoCellVertexArray, writeComputeInfoCellVertexCount, writeComputeInfoNumVerts,  writeComputeInfoSphereVerts };
        device->updateDescriptorSets(static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
        std::array<vk::DescriptorSetLayout, 1> descriptorSetLayouts = { computeDescriptorSetLayout };

        vk::PipelineLayoutCreateInfo computePipelineLayoutInfo = {};
        computePipelineLayoutInfo.flags = vk::PipelineLayoutCreateFlags();
        computePipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
        computePipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
        computePipelineLayoutInfo.pushConstantRangeCount = 0;
        computePipelineLayoutInfo.pPushConstantRanges = 0;

        try {
            computePipelineLayout = device->createPipelineLayout(computePipelineLayoutInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }

        vk::ComputePipelineCreateInfo computePipelineInfo = {};
        computePipelineInfo.flags = vk::PipelineCreateFlags();
        computePipelineInfo.stage = computeShaderStageInfo;
        computePipelineInfo.layout = computePipelineLayout;

        try {
            pipelineIdx = (vk::Pipeline)device->createComputePipeline(nullptr, computePipelineInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create compute pipeline!");
        }
    }

    VkShaderModule createShaderModule2(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(*device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    void mainLoop() {
        std::cout << sizeof(glm::vec3) << ", " << sizeof(glm::vec4) << ", " << sizeof(Vertex) << std::endl;
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        device->waitIdle();
    }

    void cleanupSwapChain() {
        device->destroyImageView(depthImageView);
        device->destroyImage(depthImage);
        device->freeMemory(depthImageMemory);

        for (auto framebuffer : swapChainFramebuffers) {
            device->destroyFramebuffer(framebuffer);
        }

        device->freeCommandBuffers(commandPool, commandBuffers);

        device->destroyPipeline(graphicsPipeline);
        device->destroyPipelineLayout(pipelineLayout);
        device->destroyRenderPass(renderPass);

        for (auto imageView : swapChainImageViews) {
            device->destroyImageView(imageView);
        }

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            device->destroyBuffer(uniformUboBuffers[i]);
            device->freeMemory(uniformUboBuffersMemory[i]);
        }

        device->destroyDescriptorPool(descriptorPool);
        device->destroySwapchainKHR(swapChain);
    }

    void cleanup() {
        cleanupSwapChain();

        device->destroyPipeline(computePipelinePhysics);
        device->destroyPipeline(computePipelineFillCellVertex);
        device->destroyPipeline(computePipelineResetCellVertex);
        device->destroyPipeline(computePipelineSphereVertex);
        device->destroyPipelineLayout(computePipelineLayout);
        device->destroyDescriptorPool(computeDescriptorPool);
        device->destroyDescriptorSetLayout(computeDescriptorSetLayout);

        // The main texture image is used until the end of the program:
        device->destroySampler(textureSampler);
        device->destroyImageView(textureImageView);
        device->destroyImage(textureImage);
        device->freeMemory(textureImageMemory);

        device->destroyBuffer(numVertsBuffer);
        device->freeMemory(numVertsBufferMemory);

        device->destroyBuffer(vertexBuffer1);
        device->freeMemory(vertexBufferMemory1);
        device->destroyBuffer(vertexBuffer2);
        device->freeMemory(vertexBufferMemory2);

        device->destroyBuffer(indexBuffer);
        device->freeMemory(indexBufferMemory);

        device->destroyBuffer(cellVertArrayBuffer);
        device->freeMemory(cellVertArrayBufferMemory);
        device->destroyBuffer(cellVertCountBuffer);
        device->freeMemory(cellVertCountBufferMemory);
        device->destroyBuffer(sphereVertsBuffer);
        device->freeMemory(sphereVertsBufferMemory);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            device->destroySemaphore(renderFinishedSemaphores[i]);
            device->destroySemaphore(imageAvailableSemaphores[i]);
            device->destroyFence(inFlightFences[i]);
        }

        device->destroyCommandPool(commandPool);

        // surface is created by glfw, therefore not using a Unique handle
        instance->destroySurfaceKHR(surface);

        /*if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(*instance, callback, nullptr);
        }*/

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device->waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createDepthResources();
        createFramebuffers();
        createuniformUboBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        auto appInfo = vk::ApplicationInfo(
            "Hello Triangle",
            VK_MAKE_VERSION(1, 0, 0),
            "No Engine",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_0
        );

        auto extensions = getRequiredExtensions();

        auto createInfo = vk::InstanceCreateInfo(
            vk::InstanceCreateFlags(),
            &appInfo,
            0, nullptr, // enabled layers
            static_cast<uint32_t>(extensions.size()), extensions.data() // enabled extensions
        );

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }

        try {
            instance = vk::createInstanceUnique(createInfo, nullptr);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void createSurface() {
        VkSurfaceKHR rawSurface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &rawSurface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }

        surface = rawSurface;
    }

    void pickPhysicalDevice() {
        auto devices = instance->enumeratePhysicalDevices();
        if (devices.size() == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (!physicalDevice) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value(), indices.computeFamily.value() };

        float queuePriority = 1.0f;

        for (uint32_t queueFamily : uniqueQueueFamilies) {
            queueCreateInfos.push_back({
                vk::DeviceQueueCreateFlags(),
                queueFamily,
                1, // queueCount
                &queuePriority
                });
        }

        auto deviceFeatures = vk::PhysicalDeviceFeatures();
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        auto createInfo = vk::DeviceCreateInfo(
            vk::DeviceCreateFlags(),
            static_cast<uint32_t>(queueCreateInfos.size()),
            queueCreateInfos.data()
        );
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }

        try {
            device = physicalDevice.createDeviceUnique(createInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create logical device!");
        }

        graphicsQueue = device->getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device->getQueue(indices.presentFamily.value(), 0);
        computeQueue = device->getQueue(indices.computeFamily.value(), 0);
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo(
            vk::SwapchainCreateFlagsKHR(),
            surface,
            imageCount,
            surfaceFormat.format,
            surfaceFormat.colorSpace,
            extent,
            1, // imageArrayLayers
            vk::ImageUsageFlagBits::eColorAttachment
        );

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        createInfo.oldSwapchain = vk::SwapchainKHR(nullptr);

        try {
            swapChain = device->createSwapchainKHR(createInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create swap chain!");
        }

        swapChainImages = device->getSwapchainImagesKHR(swapChain);

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    vk::ImageView createImageView(vk::Image image, vk::Format format,
        vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor)) {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        try {
            imageView = device->createImageView(viewInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            try {
                swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat);
            }
            catch (vk::SystemError err) {
                std::cerr << "failed to create image views!" << std::endl;
            }
        }
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment = {};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = vk::SampleCountFlagBits::e1;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
        colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

        vk::AttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::AttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = vk::SampleCountFlagBits::e1;
        depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
        depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        vk::AttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        vk::SubpassDescription subpass = {};
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;  // Unlike color attachments, a subpass can only use a single depth (+stencil) attachment.

        vk::SubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
        //dependency.srcAccessMask = 0;
        dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
        dependency.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite | vk::AccessFlagBits::eColorAttachmentWrite;

        std::array<vk::AttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

        vk::RenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());;
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        try {
            renderPass = device->createRenderPass(renderPassInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding{};
        // Specify the binding used in the shader 
        uboLayoutBinding.binding = 0;
        // Specify the type of descriptor, which is a uniform buffer object
        uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
        // Our MVP transformation is in a single uniform buffer object
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
        uboLayoutBinding.pImmutableSamplers = nullptr; // Optional, used for image sampling related descriptors

        vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        // Indicate that we intend to use the combined image sampler descriptor in the fragment shader.
        samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
        // Create this->descriptorSetLayout
        vk::DescriptorSetLayoutCreateInfo descriptorLayoutInfo{};
        descriptorLayoutInfo.flags = vk::DescriptorSetLayoutCreateFlags();
        descriptorLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        descriptorLayoutInfo.pBindings = bindings.data();

        try {
            descriptorSetLayout = device->createDescriptorSetLayout(descriptorLayoutInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("../src/shaders/vert.spv");
        auto fragShaderCode = readFile("../src/shaders/frag.spv");

        auto vertShaderModule = createShaderModule(vertShaderCode);
        auto fragShaderModule = createShaderModule(fragShaderCode);

        vk::PipelineShaderStageCreateInfo shaderStages[] = {
            {
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eVertex,
                *vertShaderModule,
                "main"
            },
            {
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eFragment,
                *fragShaderModule,
                "main"
            }
        };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;  
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        vk::Viewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        vk::Rect2D scissor = {};
        scissor.offset = vk::Offset2D{ 0, 0 };
        scissor.extent = swapChainExtent;

        vk::PipelineViewportStateCreateInfo viewportState = {};
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        vk::PipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = vk::CullModeFlagBits::eBack;
        rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
        rasterizer.depthBiasEnable = VK_FALSE;

        vk::PipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineDepthStencilStateCreateInfo depthStencil{};
        // The depthTestEnable field specifies if the depth of new fragments should be compared to the depth buffer to see if they should be discarded.
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        // Specify the comparison that is performed to keep or discard fragments.
        depthStencil.depthCompareOp = vk::CompareOp::eLess;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f; // Optional
        depthStencil.maxDepthBounds = 1.0f; // Optional
        depthStencil.stencilTestEnable = VK_FALSE;
        // depthStencil.front = {}; // Optional
        // depthStencil.back = {}; // Optional

        vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendAttachment.blendEnable = VK_FALSE;

        vk::PipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = vk::LogicOp::eCopy;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        try {
            pipelineLayout = device->createPipelineLayout(pipelineLayoutInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        vk::GraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = nullptr;

        try {
            graphicsPipeline = (vk::Pipeline)device->createGraphicsPipeline(nullptr, pipelineInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<vk::ImageView, 2> attachments = {
                swapChainImageViews[i],
                depthImageView
            };

            vk::FramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            try {
                swapChainFramebuffers[i] = device->createFramebuffer(framebufferInfo);
            }
            catch (vk::SystemError err) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        vk::CommandPoolCreateInfo poolInfo = {};
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        try {
            commandPool = device->createCommandPool(poolInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    // Handle layout transitions
    void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        // Use an image memory barrier to perform layout transitions, which is primary for synchronization purposes
        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;

        // Not want to use the barrier to transfer queue family ownership
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        // barrier.image and barrier.subresourceRange specify the image that is affected and the specific part of the image
        barrier.image = image;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = vk::AccessFlags(0);
            barrier.dstAccessMask = vk::AccessFlags(vk::AccessFlagBits::eTransferWrite);

            sourceStage = vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTopOfPipe);
            destinationStage = vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTransfer);
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlags(vk::AccessFlagBits::eTransferWrite);
            barrier.dstAccessMask = vk::AccessFlags(vk::AccessFlagBits::eShaderRead);

            sourceStage = vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTransfer);
            destinationStage = vk::PipelineStageFlags(vk::PipelineStageFlagBits::eFragmentShader);
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        commandBuffer.pipelineBarrier(sourceStage, destinationStage, vk::DependencyFlags(), nullptr, nullptr, barrier);

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::BufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = vk::Offset3D{ 0, 0, 0 };
        region.imageExtent = vk::Extent3D{
            width,
            height,
            1
        };

        // Buffer to image copy operations are enqueued using the following function
        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);

        endSingleTimeCommands(commandBuffer);
    }

    // Load an image and upload it into a Vulkan image object
    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            std::cerr << "failed to load texture image!" << std::endl;
        }
        else {
            std::cout << "load texture image successfully!" << std::endl;
        }

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;

        try {
            createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                stagingBuffer, stagingBufferMemory);
        }
        catch (std::runtime_error err) {
            std::cerr << err.what() << std::endl;
        }

        void* data = device->mapMemory(stagingBufferMemory, 0, imageSize);
        memcpy(data, pixels, (size_t)imageSize);
        device->unmapMemory(stagingBufferMemory);

        // Clean up the original pixel array
        stbi_image_free(pixels);

        // Create the texture image
        try {
            createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eDeviceLocal), textureImage, textureImageMemory);
        }
        catch (std::runtime_error err) {
            std::cerr << err.what() << std::endl;
        }

        // Copy the staging buffer to the texture image:
        // - Transition the texture image to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        // - Execute the buffer to image copy operation
        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

        // To be able to start sampling from the texture image in the shader, use one last transition to prepare it for shader access:
        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        device->destroyBuffer(stagingBuffer);
        device->freeMemory(stagingBufferMemory);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);
    }

    // Set up a sampler object and use the sampler to read colors from the texture in the shader 
    void createTextureSampler() {
        // Samplers are configured through a vk::SamplerCreateInfo structure, 
        // which specifies all filters and transformations that it should apply.
        vk::SamplerCreateInfo samplerInfo{};

        // Specify how to interpolate texels that are magnified or minified
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;

        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;

        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16.0f;  // limits the amount of texel samples that can be used to calculate the final color

        // Return black when sampling beyond the image with clamp to border addressing mode.
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;

        // The texels are addressed using the [0, 1) range on all axes.
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = vk::CompareOp::eAlways;

        // Use mipmapping
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        try {
            // The sampler is a distinct object that provides an interface to extract colors from a texture.
            textureSampler = device->createSampler(samplerInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    void loadModel()
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, BALL_PATH.c_str())) {
            std::cerr << warn + err << std::endl;
            return;
        }

        const float scale = 0.1f;

        // std::unordered_map<Vertex, uint32_t> uniqueVertices;
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                glm::vec3 pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.position = glm::vec4(pos, 1.f) * scale;
                vertex.position.w = 1.f;
                vertex.color = glm::vec4(glm::normalize(pos), 1.f);

                model_verts.push_back(vertex);
                model_indices.push_back(model_indices.size());
            }
        }

        std::cout << "Model vertices number: " << model_verts.size() << std::endl;
    }

    void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory) {
        // vk::Format texFormat = vk::Format::eR8G8B8A8Srgb;
        vk::ImageCreateInfo imgInfo({}, vk::ImageType::e2D, format,
            { width, height, 1 },
            1, 1, vk::SampleCountFlagBits::e1,
            tiling, usage, vk::SharingMode::eExclusive,
            0, nullptr, vk::ImageLayout::eUndefined
        );

        try {
            image = device->createImage(imgInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create image!");
        }

        vk::MemoryRequirements memRequirements;
        device->getImageMemoryRequirements(image, &memRequirements);

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        try {
            imageMemory = device->allocateMemory(allocInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        device->bindImageMemory(image, imageMemory, 0);
    }

    // Take a list of candidate formats in order from most desirable to least desirable, 
    // and checks which is the first one that is supported
    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
        for (vk::Format format : candidates) {
            vk::FormatProperties props;
            physicalDevice.getFormatProperties(format, &props);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    // Select a format with a depth component that supports usage as depth attachment:
    vk::Format findDepthFormat() {
        vk::Format depthFormat;
        try {
            depthFormat = findSupportedFormat(
                { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
                vk::ImageTiling::eOptimal,
                vk::FormatFeatureFlags(vk::FormatFeatureFlagBits::eDepthStencilAttachment)
            );
        }
        catch (std::runtime_error err) {
            std::cerr << err.what() << std::endl;
            depthFormat = vk::Format::eD32Sfloat;
        }

        return depthFormat;
    }

    bool hasStencilComponent(vk::Format format) {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    void createDepthResources() {
        // Find the depth format first
        vk::Format depthFormat = findDepthFormat();
        ;
        vk::ImageCreateInfo depthImgInfo({}, vk::ImageType::e2D, depthFormat,
            { swapChainExtent.width, swapChainExtent.height, 1 },
            1, 1, vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined);

        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlags(vk::ImageUsageFlagBits::eDepthStencilAttachment),
            vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eDeviceLocal),
            depthImage, depthImageMemory);

        depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth));

        // The undefined layout can be used as initial layout
        // transitionImageLayout(depthImage, depthFormat, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }

    void createVertexBuffers() {
        vk::DeviceSize bufferSize = static_cast<uint32_t>(raw_verts.size() * sizeof(Vertex));
        // vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = device->mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, raw_verts.data(), (size_t)bufferSize);
        device->unmapMemory(stagingBufferMemory);

        // Create buffer for the old vertices
        createBuffer(bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer1, vertexBufferMemory1);

        // Create buffer for the new vertices
        createBuffer(bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer2, vertexBufferMemory2);

        copyBuffer(stagingBuffer, vertexBuffer1, bufferSize);
        copyBuffer(stagingBuffer, vertexBuffer2, bufferSize);

        device->destroyBuffer(stagingBuffer);
        device->freeMemory(stagingBufferMemory);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = static_cast<uint32_t>(sphere_indices.size() * sizeof(sphere_indices[0]));
        // vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();;

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = device->mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, sphere_indices.data(), (size_t)bufferSize);
        device->unmapMemory(stagingBufferMemory);

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        device->destroyBuffer(stagingBuffer);
        device->freeMemory(stagingBufferMemory);
    }

    void createNumVertsBuffer() {
        vk::DeviceSize bufferSize = static_cast<uint32_t>(sizeof(int));

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        const int NUM_VERTS = raw_verts.size();
        void* data = device->mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, &NUM_VERTS, (size_t)bufferSize);
        device->unmapMemory(stagingBufferMemory);

        // Create buffer for the old vertices
        createBuffer(bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eDeviceLocal, numVertsBuffer, numVertsBufferMemory);

        copyBuffer(stagingBuffer, numVertsBuffer, bufferSize);

        device->destroyBuffer(stagingBuffer);
        device->freeMemory(stagingBufferMemory);
    }

    void createCellVertArrayBuffer() {
        vk::DeviceSize bufferSize = static_cast < uint32_t>(sizeof(int) * N_GRID_CELLS * 6);
        // vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();;

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = device->mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, cellVertArray, (size_t)bufferSize);
        device->unmapMemory(stagingBufferMemory);

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, cellVertArrayBuffer, cellVertArrayBufferMemory);

        copyBuffer(stagingBuffer, cellVertArrayBuffer, bufferSize);

        device->destroyBuffer(stagingBuffer);
        device->freeMemory(stagingBufferMemory);
    }

 /*   void createSphereVertsBuffer() {
        vk::DeviceSize bufferSize = static_cast <uint32_t>(sizeof(Vertex) * sphere_verts.size());
        // vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();;

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = device->mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, sphere_verts.data(), (size_t)bufferSize);
        device->unmapMemory(stagingBufferMemory);

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst
            | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            , sphereVertsBuffer, sphereVertsBufferMemory);

        copyBuffer(stagingBuffer, sphereVertsBuffer, bufferSize);

        device->destroyBuffer(stagingBuffer);
        device->freeMemory(stagingBufferMemory);
    }*/

    void createSphereVertsBuffer() {
        vk::DeviceSize bufferSize = static_cast < uint32_t>(sizeof(Vertex) * sphere_verts.size());

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = device->mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, sphere_verts.data(), (size_t)bufferSize);  
        device->unmapMemory(stagingBufferMemory);

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, sphereVertsBuffer, sphereVertsBufferMemory);

        copyBuffer(stagingBuffer, sphereVertsBuffer, bufferSize);

        device->destroyBuffer(stagingBuffer);
        device->freeMemory(stagingBufferMemory);
    }

    void createCellVertCountBuffer() {
        vk::DeviceSize bufferSize = static_cast < uint32_t>(sizeof(int) * N_GRID_CELLS);
        // vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();;

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = device->mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, cellVertCount, (size_t)bufferSize);
        device->unmapMemory(stagingBufferMemory);

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, cellVertCountBuffer, cellVertCountBufferMemory);

        copyBuffer(stagingBuffer, cellVertCountBuffer, bufferSize);

        device->destroyBuffer(stagingBuffer);
        device->freeMemory(stagingBufferMemory);
    }

    void createuniformUboBuffers() {
        vk::DeviceSize bufferSize = static_cast < uint32_t>(sizeof(UniformBufferObject));

        uniformUboBuffers.resize(swapChainImages.size());
        uniformUboBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformUboBuffers[i], uniformUboBuffersMemory[i]);
        }
    }

    // Descriptor sets can't be created directly, they must be allocated from a pool like command buffers. 
    // Allocate one of these descriptors for every frame. 
    void createDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 2> descriptorPoolSizes{};
        descriptorPoolSizes[0].type = vk::DescriptorType::eUniformBuffer;
        descriptorPoolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
        // For the allocation of the combined image sampler
        descriptorPoolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
        descriptorPoolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSizes.size());
        poolInfo.pPoolSizes = descriptorPoolSizes.data();

        // Specify the maximum number of descriptor sets that may be allocated
        poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

        try {
            descriptorPool = device->createDescriptorPool(poolInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(swapChainImages.size());
        try {
            descriptorSets = device->allocateDescriptorSets(allocInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vk::DescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformUboBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            // The resources for a combined image sampler structure must be specified in a vk::DescriptorImageInfo struct
            vk::DescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};
            descriptorWrites[0].dstSet = descriptorSets[i];
            // Give our uniform buffer binding index 0
            descriptorWrites[0].dstBinding = 0;
            // Specify the first index in the array of descriptors that we want to update. 
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
            // Specify how many array elements you want to update.
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            device->updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
        vk::BufferCreateInfo bufferInfo = {};
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        //bufferInfo.sharingMode = vk::SharingMode::eExclusive;

        try {
            buffer = device->createBuffer(bufferInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create buffer!");
        }

        vk::MemoryRequirements memRequirements = device->getBufferMemoryRequirements(buffer);

        vk::MemoryAllocateInfo allocInfo = {};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        try {
            bufferMemory = device->allocateMemory(allocInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        device->bindBufferMemory(buffer, bufferMemory, 0);
    }

    vk::CommandBuffer beginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        vk::CommandBuffer commandBuffer;
        device->allocateCommandBuffers(&allocInfo, &commandBuffer);

        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        commandBuffer.begin(&beginInfo);
        return commandBuffer;
    }

    void endSingleTimeCommands(vk::CommandBuffer& commandBuffer) {
        commandBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();

        device->freeCommandBuffers(commandPool, 1, &commandBuffer);
    }

    void copyBuffer(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer, const vk::DeviceSize& size) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::BufferCopy copyRegion{};
        copyRegion.size = size;
        commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        commandBuffers.resize(swapChainFramebuffers.size());

        vk::CommandBufferAllocateInfo allocInfo = {};
        allocInfo.commandPool = commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        try {
            commandBuffers = device->allocateCommandBuffers(allocInfo);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        for (size_t i = 0; i < commandBuffers.size(); i++) {
            vk::CommandBufferBeginInfo beginInfo = {};
            beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;

            try {
                commandBuffers[i].begin(beginInfo);
            }
            catch (vk::SystemError err) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            vk::RenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFramebuffers[i];
            renderPassInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
            renderPassInfo.renderArea.extent = swapChainExtent;

            std::array<vk::ClearValue, 2> clearValues{};
            clearValues[0].color = std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f };
            clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };
            renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
            renderPassInfo.pClearValues = clearValues.data();

            // Bind the compute pipeline
            //vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelinePhysics);
            commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eCompute, computePipelineResetCellVertex); 

            // Bind descriptor sets for compute
            //vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &ComputeDescriptorSet, 0, nullptr);
            commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, 1, computeDescriptorSet.data(), 0, nullptr);

            // Dispatch the compute kernel, with one thread for each vertex
            commandBuffers[i].dispatch(N_GRID_CELLS, 1, 1);

            vk::BufferMemoryBarrier computeToComputeBarrier = {};
            computeToComputeBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead;
            computeToComputeBarrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead;
            computeToComputeBarrier.srcQueueFamilyIndex = queueFamilyIndices.computeFamily.value();
            computeToComputeBarrier.dstQueueFamilyIndex = queueFamilyIndices.computeFamily.value();
            computeToComputeBarrier.buffer = cellVertCountBuffer;
            computeToComputeBarrier.offset = 0;
            computeToComputeBarrier.size = N_GRID_CELLS * sizeof(int);  


            vk::PipelineStageFlags computeShaderStageFlags_1(vk::PipelineStageFlagBits::eComputeShader);
            vk::PipelineStageFlags computeShaderStageFlags_2(vk::PipelineStageFlagBits::eComputeShader);
            commandBuffers[i].pipelineBarrier(computeShaderStageFlags_1,
                computeShaderStageFlags_2,
                vk::DependencyFlags(),
                0, nullptr,
                1, &computeToComputeBarrier,
                0, nullptr);

            // Bind the compute pipeline
            //vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelinePhysics);
            commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eCompute, computePipelineFillCellVertex);

            // Bind descriptor sets for compute
            //vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &ComputeDescriptorSet, 0, nullptr);
            commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, 1, computeDescriptorSet.data(), 0, nullptr);

            // Dispatch the compute kernel, with one thread for each vertex
            commandBuffers[i].dispatch(uint32_t(raw_verts.size()), 1, 1);

            vk::BufferMemoryBarrier computeToComputeBarrier1 = {};
            computeToComputeBarrier1.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead;
            computeToComputeBarrier1.dstAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead;
            computeToComputeBarrier1.srcQueueFamilyIndex = queueFamilyIndices.computeFamily.value();
            computeToComputeBarrier1.dstQueueFamilyIndex = queueFamilyIndices.computeFamily.value();
            computeToComputeBarrier1.buffer = cellVertCountBuffer;
            computeToComputeBarrier1.offset = 0;
            computeToComputeBarrier1.size = N_GRID_CELLS * sizeof(int);  //vertexBufferSize


            vk::PipelineStageFlags computeShaderStageFlags_3(vk::PipelineStageFlagBits::eComputeShader);
            vk::PipelineStageFlags computeShaderStageFlags_4(vk::PipelineStageFlagBits::eComputeShader);
            commandBuffers[i].pipelineBarrier(computeShaderStageFlags_3,
                computeShaderStageFlags_4,
                vk::DependencyFlags(),
                0, nullptr,
                1, &computeToComputeBarrier1,
                0, nullptr);

            // Bind the compute pipeline
            //vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelinePhysics);
            commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eCompute, computePipelinePhysics);

            // Bind descriptor sets for compute
            //vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &ComputeDescriptorSet, 0, nullptr);
            commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, 1, computeDescriptorSet.data(), 0, nullptr);

            // Dispatch the compute kernel, with one thread for each vertex
            commandBuffers[i].dispatch(uint32_t(raw_verts.size()), 1, 1);

            vk::BufferMemoryBarrier computeToComputeBarrier2 = {};
            computeToComputeBarrier2.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead;
            computeToComputeBarrier2.dstAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead;
            computeToComputeBarrier2.srcQueueFamilyIndex = queueFamilyIndices.computeFamily.value();
            computeToComputeBarrier2.dstQueueFamilyIndex = queueFamilyIndices.computeFamily.value();
            computeToComputeBarrier2.buffer = vertexBuffer2;
            computeToComputeBarrier2.offset = 0;
            computeToComputeBarrier2.size = uint32_t(raw_verts.size()) * sizeof(Vertex);  //vertexBufferSize


            vk::PipelineStageFlags computeShaderStageFlags_5(vk::PipelineStageFlagBits::eComputeShader);
            vk::PipelineStageFlags computeShaderStageFlags_6(vk::PipelineStageFlagBits::eComputeShader);
            commandBuffers[i].pipelineBarrier(computeShaderStageFlags_5,
                computeShaderStageFlags_6,
                vk::DependencyFlags(),
                0, nullptr,
                1, &computeToComputeBarrier2,
                0, nullptr);

            // Bind the compute pipeline
            //vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelinePhysics);
            commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eCompute, computePipelineSphereVertex);

            // Bind descriptor sets for compute
            //vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &ComputeDescriptorSet, 0, nullptr);
            commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, 1, computeDescriptorSet.data(), 0, nullptr);

            // Dispatch the compute kernel, with one thread for each vertex
            commandBuffers[i].dispatch(uint32_t(sphere_verts.size()), 1, 1);
            
            // Define a memory barrier to transition the vertex buffer from a compute storage object to a vertex input
            vk::BufferMemoryBarrier computeToVertexBarrier = {};
            computeToVertexBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead;
            computeToVertexBarrier.dstAccessMask = vk::AccessFlagBits::eVertexAttributeRead;
            computeToVertexBarrier.srcQueueFamilyIndex = queueFamilyIndices.computeFamily.value();
            computeToVertexBarrier.dstQueueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
            computeToVertexBarrier.buffer = sphereVertsBuffer;
            computeToVertexBarrier.offset = 0;
            computeToVertexBarrier.size = uint32_t(sphere_verts.size()) * sizeof(Vertex);  //vertexBufferSize


            vk::PipelineStageFlags computeShaderStageFlags(vk::PipelineStageFlagBits::eComputeShader);
            vk::PipelineStageFlags vertexShaderStageFlags(vk::PipelineStageFlagBits::eVertexInput);
            commandBuffers[i].pipelineBarrier(computeShaderStageFlags,
                vertexShaderStageFlags,
                vk::DependencyFlags(),
                0, nullptr,
                1, &computeToVertexBarrier,
                0, nullptr);

            commandBuffers[i].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

            vk::Buffer vertexBuffers[] = { sphereVertsBuffer };
            vk::DeviceSize offsets[] = { 0 };
            commandBuffers[i].bindVertexBuffers(0, 1, vertexBuffers, offsets);
            commandBuffers[i].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);

            commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);
            commandBuffers[i].drawIndexed(static_cast<uint32_t>(uint32_t(sphere_indices.size())), 1, 0, 0, 0);

            commandBuffers[i].endRenderPass();

            try {
                commandBuffers[i].end();
            }
            catch (vk::SystemError err) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        try {
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                imageAvailableSemaphores[i] = device->createSemaphore({});
                renderFinishedSemaphores[i] = device->createSemaphore({});
                inFlightFences[i] = device->createFence({ vk::FenceCreateFlagBits::eSignaled });
            }
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }

    void drawFrame() {
        device->waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

        uint32_t imageIndex;
        try {
            vk::ResultValue result = device->acquireNextImageKHR(swapChain, std::numeric_limits<uint64_t>::max(),
                imageAvailableSemaphores[currentFrame], nullptr);
            imageIndex = result.value;
        }
        catch (vk::OutOfDateKHRError err) {
            recreateSwapChain();
            return;
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        //updateVertexBuffer(imageIndex);

        vk::DeviceSize bufferSize = static_cast<uint32_t>(raw_verts.size() * sizeof(Vertex));
        copyBuffer(vertexBuffer2, vertexBuffer1, bufferSize);
        //std::swap(descriptorSets[0], descriptorSets[1]);
        updateUniformBuffer(imageIndex);
        vk::SubmitInfo submitInfo = {};

        vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        device->resetFences(1, &inFlightFences[currentFrame]);

        try {
            graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        vk::PresentInfoKHR presentInfo = {};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        vk::SwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        vk::Result resultPresent;
        try {
            resultPresent = presentQueue.presentKHR(presentInfo);
        }
        catch (vk::OutOfDateKHRError err) {
            resultPresent = vk::Result::eErrorOutOfDateKHR;
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        if (resultPresent == vk::Result::eSuboptimalKHR || resultPresent == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
            return;
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    }

    // Generate a new transformation every frame to make the geometry spin around. 
    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::mat4(1.f);
        ubo.view = viewMat;
        ubo.proj = glm::perspective(glm::radians(45.f), swapChainExtent.width / (float)swapChainExtent.height, 0.01f, 30.0f);
        ubo.proj[1][1] *= -1;

        void* data = device->mapMemory(uniformUboBuffersMemory[currentImage], 0, sizeof(ubo));
        memcpy(data, &ubo, sizeof(ubo));
        device->unmapMemory(uniformUboBuffersMemory[currentImage]);
    }

    void updateVertexBuffer(uint32_t currentImage) {
       /* void* data = device->mapMemory(vertexBufferMemory1, 0, static_cast<uint32_t>(raw_verts.size() * sizeof(Vertex)));
        memcpy(data, raw_verts.data(), sizeof(raw_verts[0]) * uint32_t(raw_verts.size()));
        device->unmapMemory(vertexBufferMemory1);*/
    }

    vk::UniqueShaderModule createShaderModule(const std::vector<char>& code) {
        try {
            return device->createShaderModuleUnique({
                vk::ShaderModuleCreateFlags(),
                code.size(),
                reinterpret_cast<const uint32_t*>(code.data())
                });
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create shader module!");
        }
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined) {
            return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
        }

        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> availablePresentModes) {
        vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
            else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                bestMode = availablePresentMode;
            }
        }

        return bestMode;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            vk::Extent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

            actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& device) {
        SwapChainSupportDetails details;
        details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
        details.formats = device.getSurfaceFormatsKHR(surface);
        details.presentModes = device.getSurfacePresentModesKHR(surface);

        return details;
    }

    bool isDeviceSuitable(const vk::PhysicalDevice& device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        vk::PhysicalDeviceFeatures supportedFeatures;
        device.getFeatures(&supportedFeatures);

        return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
    }

    bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device) {
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : device.enumerateDeviceExtensionProperties()) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device) {
        QueueFamilyIndices indices;

        auto queueFamilies = device.getQueueFamilyProperties();

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }

            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eCompute) {
                indices.computeFamily = i;
            }

            if (queueFamily.queueCount > 0 && device.getSurfaceSupportKHR(i, surface)) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        auto availableLayers = vk::enumerateInstanceLayerProperties();
        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }
};
