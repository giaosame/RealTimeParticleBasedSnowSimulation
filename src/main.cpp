#include "vulkanpointsrenderer.h"
#include "vulkanspheresrenderer.h"

#define RENDER_USING_POINTS true

int main() {
#if RENDER_USING_POINTS
    VulkanPointsRenderer myRender;
#else
    VulkanSpheresRenderer myRender;
#endif // RENDER_USING_POINTS
    
    try {
        myRender.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}