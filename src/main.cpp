
#include <vulkan/vulkan.h>

#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>

#include "Bitmap.h" // Save bmp file

#include <iostream>

const int WIDTH          = 3200;  // Size of rendered mandelbrot set.
const int HEIGHT         = 2400;  // Size of renderered mandelbrot set.
const int WORKGROUP_SIZE = 32;    // Workgroup size in compute shader.

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

#include "vk_utils.h"


/*
The application launches a compute shader that renders the mandelbrot set,
by rendering it into a storage buffer.
The storage buffer is then read from the GPU, and saved as .png. 
*/
class ComputeApplication
{
private:
    // The pixels of the rendered mandelbrot set are in this format:
    struct Pixel {
        float r, g, b, a;
    };
    
    /*
    In order to use Vulkan, you must create an instance. 
    */
    VkInstance instance;

    VkDebugReportCallbackEXT debugReportCallback;
    /*
    The physical device is some device on the system that supports usage of Vulkan.
    Often, it is simply a graphics card that supports Vulkan. 
    */
    VkPhysicalDevice physicalDevice;
    /*
    Then we have the logical device VkDevice, which basically allows 
    us to interact with the physical device. 
    */
    VkDevice device;

    /*
    The pipeline specifies the pipeline that all graphics and compute commands pass though in Vulkan.

    We will be creating a simple compute pipeline in this application. 
    */
    VkPipeline       pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule   computeShaderModule;

    /*
    The command buffer is used to record commands, that will be submitted to a queue.

    To allocate such command buffers, we use a command pool.
    */
    VkCommandPool   commandPool;
    VkCommandBuffer commandBuffer;

    /*

    Descriptors represent resources in shaders. They allow us to use things like
    uniform buffers, storage buffers and images in GLSL. 

    A single descriptor represents a single resource, and several descriptors are organized
    into descriptor sets, which are basically just collections of descriptors.
    */
    VkDescriptorPool      descriptorPool;
    VkDescriptorSet       descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    /*
    The mandelbrot set will be rendered to this buffer.

    The memory that backs the buffer is bufferMemory. 
    */
    VkBuffer       buffer;
    VkDeviceMemory bufferMemory;

    std::vector<const char *> enabledLayers;

    /*
    In order to execute commands on a device(GPU), the commands must be submitted
    to a queue. The commands are stored in a command buffer, and this command buffer
    is given to the queue. 

    There will be different kinds of queues on the device. Not all queues support
    graphics operations, for instance. For this application, we at least want a queue
    that supports compute operations. 
    */
    VkQueue queue; // a queue supporting compute operations.

public:

    void run()
    {
      const int deviceId = 0;

      std::cout << "init vulkan for device " << deviceId << " ... " << std::endl;

      instance = vk_utils::CreateInstance(enableValidationLayers, enabledLayers);

      if(enableValidationLayers)
      {
        vk_utils::InitDebugReportCallback(instance,
                                          &debugReportCallbackFn, &debugReportCallback);
      }

      physicalDevice = vk_utils::FindPhysicalDevice(instance, true, deviceId);

      /*
      Groups of queues that have the same capabilities(for instance, they all supports graphics and computer operations),
      are grouped into queue families.

      When submitting a command buffer, you must specify to which queue in the family you are submitting to.
      This variable keeps track of the index of that queue in its family.
      */
      uint32_t queueFamilyIndex = vk_utils::GetComputeQueueFamilyIndex(physicalDevice);

      device = vk_utils::CreateLogicalDevice(queueFamilyIndex, physicalDevice, enabledLayers);

      vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

      // Buffer size of the storage buffer that will contain the rendered mandelbrot set.
      size_t bufferSize = sizeof(Pixel) * WIDTH * HEIGHT;

      std::cout << "creating resources ... " << std::endl;
      createBuffer(device, physicalDevice, bufferSize,      // very simple example of allocation
                   &buffer, &bufferMemory);                 // (device, bufferSize) ==> (buffer, bufferMemory)

      createDescriptorSetLayout(device, &descriptorSetLayout);                          // here we will create a binding of buffer to shader via descriptorSet
      createDescriptorSetForOurBuffer(device, buffer, bufferSize, &descriptorSetLayout, // (device, buffer, bufferSize, descriptorSetLayout) ==>
                                      &descriptorPool, &descriptorSet);                 // (descriptorPool, descriptorSet)

      std::cout << "compiling shaders  ... " << std::endl;
      createComputePipeline(device, descriptorSetLayout,
                            &computeShaderModule, &pipeline, &pipelineLayout);

      createCommandBuffer(device, queueFamilyIndex, pipeline, pipelineLayout,
                          &commandPool, &commandBuffer);

      recordCommandsTo(commandBuffer, pipeline, pipelineLayout, descriptorSet);

      // Finally, run the recorded command buffer.
      std::cout << "doing computations ... " << std::endl;
      runCommandBuffer(commandBuffer, queue, device);

      // The former command rendered a mandelbrot set to a buffer.
      // Save that buffer as a png on disk.
      std::cout << "saving image       ... " << std::endl;
      saveRenderedImageFromDeviceMemory(device, bufferMemory, 0, WIDTH, HEIGHT);

      // Clean up all vulkan resources.
      std::cout << "destroying all     ... " << std::endl;
      cleanup();
    }

    // assume simple pitch-linear data layout and 'a_bufferMemory' to be a mapped memory.
    //
    static void saveRenderedImageFromDeviceMemory(VkDevice a_device, VkDeviceMemory a_bufferMemory, size_t a_offset, int a_width, int a_height)
    {
      const int a_bufferSize = a_width * a_height * 4;

      void* mappedMemory = nullptr;
      // Map the buffer memory, so that we can read from it on the CPU.
      vkMapMemory(a_device, a_bufferMemory, a_offset, a_bufferSize, 0, &mappedMemory);
      Pixel* pmappedMemory = (Pixel *)mappedMemory;

      // Get the color data from the buffer, and cast it to bytes.
      // We save the data to a vector.
      std::vector<unsigned char> image;
      image.reserve(a_width * a_height * 4);
      for (int i = 0; i < (a_width * a_height); i += 1)
      {
        image.push_back((unsigned char)(255.0f * (pmappedMemory[i].r)));
        image.push_back((unsigned char)(255.0f * (pmappedMemory[i].g)));
        image.push_back((unsigned char)(255.0f * (pmappedMemory[i].b)));
        image.push_back((unsigned char)(255.0f * (pmappedMemory[i].a)));
      }

      // Done reading, so unmap.
      vkUnmapMemory(a_device, a_bufferMemory);

      SaveBMP("mandelbrot.bmp", (const uint32_t*)image.data(), WIDTH, HEIGHT);
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
        VkDebugReportFlagsEXT                       flags,
        VkDebugReportObjectTypeEXT                  objectType,
        uint64_t                                    object,
        size_t                                      location,
        int32_t                                     messageCode,
        const char*                                 pLayerPrefix,
        const char*                                 pMessage,
        void*                                       pUserData)
    {
        printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);
        return VK_FALSE;
    }


    static void createBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, const size_t a_bufferSize,
                             VkBuffer* a_pBuffer, VkDeviceMemory* a_pBufferMemory)
    {
      /*
      We will now create a buffer. We will render the mandelbrot set into this buffer
      in a computer shade later.
      */
      VkBufferCreateInfo bufferCreateInfo = {};
      bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferCreateInfo.size        = a_bufferSize; // buffer size in bytes.
      bufferCreateInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
      bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time.

      VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, a_pBuffer)); // create buffer.

      /*
      But the buffer doesn't allocate memory for itself, so we must do that manually.
      First, we find the memory requirements for the buffer.
      */
      VkMemoryRequirements memoryRequirements;
      vkGetBufferMemoryRequirements(a_device, (*a_pBuffer), &memoryRequirements);
        
      /*
      Now use obtained memory requirements info to allocate the memory for the buffer.
      There are several types of memory that can be allocated, and we must choose a memory type that
      1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
      2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
         with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.

      Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily
      visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
      this flag.
      */
      VkMemoryAllocateInfo allocateInfo = {};
      allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocateInfo.allocationSize  = memoryRequirements.size; // specify required memory.
      allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, a_physDevice);

      VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, a_pBufferMemory)); // allocate memory on device.
        
      // Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory.
      VK_CHECK_RESULT(vkBindBufferMemory(a_device, (*a_pBuffer), (*a_pBufferMemory), 0));
    }

    static void createDescriptorSetLayout(VkDevice a_device, VkDescriptorSetLayout* a_pDSLayout)
    {
       /*
       Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point 0.
       This binds to
         layout(std140, binding = 0) buffer buf
       in the compute shader.
       */
       VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
       descriptorSetLayoutBinding.binding         = 0; // binding = 0
       descriptorSetLayoutBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
       descriptorSetLayoutBinding.descriptorCount = 1;
       descriptorSetLayoutBinding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

       VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
       descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
       descriptorSetLayoutCreateInfo.bindingCount = 1; // only a single binding in this descriptor set layout.
       descriptorSetLayoutCreateInfo.pBindings    = &descriptorSetLayoutBinding;

       // Create the descriptor set layout.
       VK_CHECK_RESULT(vkCreateDescriptorSetLayout(a_device, &descriptorSetLayoutCreateInfo, NULL, a_pDSLayout));
    }

    static void createDescriptorSetForOurBuffer(VkDevice a_device, VkBuffer a_buffer, size_t a_bufferSize, const VkDescriptorSetLayout* a_pDSLayout,
                                                VkDescriptorPool* a_pDSPool, VkDescriptorSet* a_pDS)
    {
      /*
      So we will allocate a descriptor set here.
      But we need to first create a descriptor pool to do that.
      */

      /*
      Our descriptor pool can only allocate a single storage buffer.
      */
      VkDescriptorPoolSize descriptorPoolSize = {};
      descriptorPoolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorPoolSize.descriptorCount = 1;

      VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
      descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      descriptorPoolCreateInfo.maxSets       = 1; // we only need to allocate one descriptor set from the pool.
      descriptorPoolCreateInfo.poolSizeCount = 1;
      descriptorPoolCreateInfo.pPoolSizes    = &descriptorPoolSize;

      // create descriptor pool.
      VK_CHECK_RESULT(vkCreateDescriptorPool(a_device, &descriptorPoolCreateInfo, NULL, a_pDSPool));

      /*
      With the pool allocated, we can now allocate the descriptor set.
      */
      VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
      descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      descriptorSetAllocateInfo.descriptorPool     = (*a_pDSPool); // pool to allocate from.
      descriptorSetAllocateInfo.descriptorSetCount = 1;            // allocate a single descriptor set.
      descriptorSetAllocateInfo.pSetLayouts        = a_pDSLayout;

      // allocate descriptor set.
      VK_CHECK_RESULT(vkAllocateDescriptorSets(a_device, &descriptorSetAllocateInfo, a_pDS));

      /*
      Next, we need to connect our actual storage buffer with the descrptor.
      We use vkUpdateDescriptorSets() to update the descriptor set.
      */

      // Specify the buffer to bind to the descriptor.
      VkDescriptorBufferInfo descriptorBufferInfo = {};
      descriptorBufferInfo.buffer = a_buffer;
      descriptorBufferInfo.offset = 0;
      descriptorBufferInfo.range  = a_bufferSize;

      VkWriteDescriptorSet writeDescriptorSet = {};
      writeDescriptorSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writeDescriptorSet.dstSet          = (*a_pDS); // write to this descriptor set.
      writeDescriptorSet.dstBinding      = 0;        // write to the first, and only binding.
      writeDescriptorSet.descriptorCount = 1;        // update a single descriptor.
      writeDescriptorSet.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
      writeDescriptorSet.pBufferInfo     = &descriptorBufferInfo;

      // perform the update of the descriptor set.
      vkUpdateDescriptorSets(a_device, 1, &writeDescriptorSet, 0, NULL);
    }

    static void createComputePipeline(VkDevice a_device, const VkDescriptorSetLayout& a_dsLayout,
                                      VkShaderModule* a_pShaderModule, VkPipeline* a_pPipeline, VkPipelineLayout* a_pPipelineLayout)
    {
      //Create a shader module. A shader module basically just encapsulates some shader code.
      //
      // the code in comp.spv was created by running the command:
      // glslangValidator.exe -V shader.comp
      std::vector<uint32_t> code = vk_utils::ReadFile("shaders/comp.spv");
      VkShaderModuleCreateInfo createInfo = {};
      createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      createInfo.pCode    = code.data();
      createInfo.codeSize = code.size()*sizeof(uint32_t);
        
      VK_CHECK_RESULT(vkCreateShaderModule(a_device, &createInfo, NULL, a_pShaderModule));

      /*
      Now let us actually create the compute pipeline.
      A compute pipeline is very simple compared to a graphics pipeline.
      It only consists of a single stage with a compute shader.

      So first we specify the compute shader stage, and it's entry point(main).
      */
      VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
      shaderStageCreateInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      shaderStageCreateInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
      shaderStageCreateInfo.module = (*a_pShaderModule);
      shaderStageCreateInfo.pName  = "main";

      /*
      The pipeline layout allows the pipeline to access descriptor sets.
      So we just specify the descriptor set layout we created earlier.
      */
      VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
      pipelineLayoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipelineLayoutCreateInfo.setLayoutCount = 1;
      pipelineLayoutCreateInfo.pSetLayouts    = &a_dsLayout;
      VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &pipelineLayoutCreateInfo, NULL, a_pPipelineLayout));

      VkComputePipelineCreateInfo pipelineCreateInfo = {};
      pipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pipelineCreateInfo.stage  = shaderStageCreateInfo;
      pipelineCreateInfo.layout = (*a_pPipelineLayout);

      // Now, we finally create the compute pipeline.
      //
      VK_CHECK_RESULT(vkCreateComputePipelines(a_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, a_pPipeline));
    }

    static void createCommandBuffer(VkDevice a_device, uint32_t queueFamilyIndex, VkPipeline a_pipeline, VkPipelineLayout a_layout,
                                    VkCommandPool* a_pool, VkCommandBuffer* a_pCmdBuff)
    {
      /*
      We are getting closer to the end. In order to send commands to the device(GPU),
      we must first record commands into a command buffer.
      To allocate a command buffer, we must first create a command pool. So let us do that.
      */
      VkCommandPoolCreateInfo commandPoolCreateInfo = {};
      commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      commandPoolCreateInfo.flags = 0;
      // the queue family of this command pool. All command buffers allocated from this command pool,
      // must be submitted to queues of this family ONLY.
      commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
      VK_CHECK_RESULT(vkCreateCommandPool(a_device, &commandPoolCreateInfo, NULL, a_pool));

      /*
      Now allocate a command buffer from the command pool.
      */
      VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
      commandBufferAllocateInfo.sType       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      commandBufferAllocateInfo.commandPool = (*a_pool); // specify the command pool to allocate from.
      // if the command buffer is primary, it can be directly submitted to queues.
      // A secondary buffer has to be called from some primary command buffer, and cannot be directly
      // submitted to a queue. To keep things simple, we use a primary command buffer.
      commandBufferAllocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      commandBufferAllocateInfo.commandBufferCount = 1; // allocate a single command buffer.
      VK_CHECK_RESULT(vkAllocateCommandBuffers(a_device, &commandBufferAllocateInfo, a_pCmdBuff)); // allocate command buffer.
    }

    static void recordCommandsTo(VkCommandBuffer a_cmdBuff, VkPipeline a_pipeline, VkPipelineLayout a_layout, const VkDescriptorSet& a_ds)
    {
      /*
      Now we shall start recording commands into the newly allocated command buffer.
      */
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the buffer is only submitted and used once in this application.
      VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo)); // start recording commands.

      /*
      We need to bind a pipeline, AND a descriptor set before we dispatch
      The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
      */
      vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, a_pipeline);
      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, a_layout, 0, 1, &a_ds, 0, NULL);

      /*
      Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
      The number of workgroups is specified in the arguments.
      If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
      */
      vkCmdDispatch(a_cmdBuff, (uint32_t)ceil(WIDTH / float(WORKGROUP_SIZE)), (uint32_t)ceil(HEIGHT / float(WORKGROUP_SIZE)), 1);

      VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff)); // end recording commands.
    }


    static void runCommandBuffer(VkCommandBuffer a_cmdBuff, VkQueue a_queue, VkDevice a_device)
    {
      /*
      Now we shall finally submit the recorded command buffer to a queue.
      */
      VkSubmitInfo submitInfo = {};
      submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submitInfo.commandBufferCount = 1; // submit a single command buffer
      submitInfo.pCommandBuffers    = &a_cmdBuff; // the command buffer to submit.

      /*
        We create a fence.
      */
      VkFence fence;
      VkFenceCreateInfo fenceCreateInfo = {};
      fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      fenceCreateInfo.flags = 0;
      VK_CHECK_RESULT(vkCreateFence(a_device, &fenceCreateInfo, NULL, &fence));

      /*
      We submit the command buffer on the queue, at the same time giving a fence.
      */
      VK_CHECK_RESULT(vkQueueSubmit(a_queue, 1, &submitInfo, fence));

      /*
      The command will not have finished executing until the fence is signalled.
      So we wait here.
      We will directly after this read our buffer from the GPU,
      and we will not be sure that the command has finished executing unless we wait for the fence.
      Hence, we use a fence here.
      */
      VK_CHECK_RESULT(vkWaitForFences(a_device, 1, &fence, VK_TRUE, 100000000000));

      vkDestroyFence(a_device, fence, NULL);
    }

    void cleanup() {
        /*
        Clean up all Vulkan Resources. 
        */

        if (enableValidationLayers) {
            // destroy callback.
            auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
            if (func == nullptr) {
                throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
            }
            func(instance, debugReportCallback, NULL);
        }

        vkFreeMemory(device, bufferMemory, NULL);
        vkDestroyBuffer(device, buffer, NULL);	
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);	
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);		
    }
};

int main()
{
  ComputeApplication app;

  try
  {
    app.run();
  }
  catch (const std::runtime_error& e)
  {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }
    
  return EXIT_SUCCESS;
}
