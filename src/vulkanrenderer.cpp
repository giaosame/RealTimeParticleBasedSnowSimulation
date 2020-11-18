#include "vulkanrenderer.h"

void VulkanRenderer::run()
{
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void VulkanRenderer::initWindow()
{
    // Initialize the GLFW library
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);  // disables resizing wingdow for now
    window = glfwCreateWindow(WIDTH, HEIGHT, WINDOW_TITLE, nullptr, nullptr);
}

void VulkanRenderer::initVulkan()
{
    createInstance();
}

void VulkanRenderer::createInstance()
{
    if (enableValidationLayers && !checkValidationLayerSupport())
    {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    // appInfo: a struct providing some useful information to the driver 
    // in order to optimize our specific application 
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // createInfo: tells the Vulkan driver which global extensions and validation layers we want to use.
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;
    
    if (enableValidationLayers) 
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    else 
    {
        createInfo.enabledLayerCount = 0;
    }

    try 
    {
        instance = vk::createInstanceUnique(createInfo, nullptr);
    }
    catch (vk::SystemError err) 
    {
        throw std::runtime_error("failed to create instance!");
    }

    std::cout << "available extensions:" << std::endl;

    for (const auto& extension : vk::enumerateInstanceExtensionProperties()) 
    {
        std::cout << "\t" << extension.extensionName << std::endl;
    }
}

bool VulkanRenderer::checkValidationLayerSupport()
{
    // List all of the available layers
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    // Check if all of the layers in validationLayers exist in the availableLayers list
    for (const char* layerName : validationLayers)
    {
        bool layerFound = false;
        for (const auto& layer : availableLayers)
        {
            if (strcmp(layerName, layer.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            return false;
        }
    }
    return true;
}

void VulkanRenderer::mainLoop()
{
    // Keep the application running until either an error occurs 
    // or the window is closed
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
    }
}

void VulkanRenderer::cleanup()
{
    // All Vulkan resources should be only destroyed right before the program exits.
    
    // Remove the following function because instance destruction is handled by UniqueInstance
    // vkDestroyInstance(instance, nullptr);  

    glfwDestroyWindow(window);
    glfwTerminate();
}