
#include <vulkan/vulkan.h>

#include <vector>
#include <cstring>
#include <cassert>
#include <stdexcept>
#include <cmath>

#include "Bitmap.h" // Save bmp file

#include <iostream>

const int WIDTH          = 3200;  // Size of rendered mandelbrot set.
const int HEIGHT         = 2400;  // Size of renderered mandelbrot set.
const int WORKGROUP_SIZE = 16;    // Workgroup size in compute shader.

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

#include "vk_utils.h"

struct CustomVulkanTexture // in fact this is bad example ... but simple
{
  CustomVulkanTexture() : imagesMemoryGPU(0), imageGPU(0), imageSampler(0), imageView(0) {}

  void Release(VkDevice a_device)
  {
    vkFreeMemory      (a_device, imagesMemoryGPU, NULL);
    vkDestroyImage    (a_device, imageGPU, NULL);
    vkDestroyImageView(a_device, imageView, NULL);
    vkDestroySampler  (a_device, imageSampler, NULL);
  }

  VkDeviceMemory imagesMemoryGPU; // this is bad design in fact, you should store memory somewhere else and/or use memory pools;
  VkImage        imageGPU;
  VkSampler      imageSampler;
  VkImageView    imageView;

  static CustomVulkanTexture Create2DTextureRGBA256(VkDevice a_device, VkPhysicalDevice a_physDevice, int w, int h);

//protected:

  static void CreateTexture(VkDevice a_device, VkPhysicalDevice a_physDevice, const int a_width, const int a_height,
                            VkImage *a_images, VkDeviceMemory *a_pImagesMemory)
  {
    // first create desired objects, but still don't allocate memory for them
    //
    VkImageCreateInfo imgCreateInfo = {};
    imgCreateInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCreateInfo.pNext         = nullptr;
    imgCreateInfo.flags         = 0; // not sure about this ...
    imgCreateInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgCreateInfo.format        = VK_FORMAT_R8G8B8A8_UNORM;
    imgCreateInfo.extent        = VkExtent3D{uint32_t(a_width), uint32_t(a_height), 1};
    imgCreateInfo.mipLevels     = 1;
    imgCreateInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCreateInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgCreateInfo.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; // copy to the texture and read then
    imgCreateInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgCreateInfo.arrayLayers   = 1;
    VK_CHECK_RESULT(vkCreateImage(a_device, &imgCreateInfo, nullptr, a_images + 0));
    // now allocate memory for both images
    //
    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(a_device, a_images[0], &memoryRequirements);
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize  = memoryRequirements.size; // specify required memory.
    allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physDevice);
    VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, a_pImagesMemory)); // allocate memory on device.
    VK_CHECK_RESULT(vkBindImageMemory(a_device, a_images[0], (*a_pImagesMemory), 0));
  }

  void CreateOther(VkDevice a_device)
  {
    VkSamplerCreateInfo samplerInfo = {};
    {
      samplerInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      samplerInfo.pNext        = nullptr;
      samplerInfo.flags        = 0;
      samplerInfo.magFilter    = VK_FILTER_LINEAR;
      samplerInfo.minFilter    = VK_FILTER_LINEAR;
      samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.mipLodBias   = 0.0f;
      samplerInfo.compareOp    = VK_COMPARE_OP_NEVER;
      samplerInfo.minLod           = 0;
      samplerInfo.maxLod           = 0;
      samplerInfo.maxAnisotropy    = 1.0;
      samplerInfo.anisotropyEnable = VK_FALSE;
      samplerInfo.borderColor      = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
      samplerInfo.unnormalizedCoordinates = VK_FALSE;
    }
    VK_CHECK_RESULT(vkCreateSampler(a_device, &samplerInfo, nullptr, &this->imageSampler));

    VkImageViewCreateInfo imageViewInfo = {};
    {
      imageViewInfo.sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      imageViewInfo.flags      = 0;
      imageViewInfo.viewType   = VK_IMAGE_VIEW_TYPE_2D;
      imageViewInfo.format     = VK_FORMAT_R8G8B8A8_UNORM;
      imageViewInfo.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
      // The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
      // It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
      imageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      imageViewInfo.subresourceRange.baseMipLevel   = 0;
      imageViewInfo.subresourceRange.baseArrayLayer = 0;
      imageViewInfo.subresourceRange.layerCount     = 1;
      imageViewInfo.subresourceRange.levelCount     = 1;
      // The view will be based on the texture's image
      imageViewInfo.image = this->imageGPU;
    }
    VK_CHECK_RESULT(vkCreateImageView(a_device, &imageViewInfo, nullptr, &this->imageView));
  }


};

CustomVulkanTexture CustomVulkanTexture::Create2DTextureRGBA256(VkDevice a_device, VkPhysicalDevice a_physDevice, int w, int h)
{
  CustomVulkanTexture res;

  res.CreateTexture(a_device, a_physDevice, w, h,         // (w,h) ==> (imageGPU, imagesMemoryGPU); for R8G8B8A8_UNORM format
                    &res.imageGPU, &res.imagesMemoryGPU); //

  res.CreateOther(a_device);

  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
The application launches a compute shader that renders the mandelbrot set,
by rendering it into a storage bufferStaging.
The storage bufferStaging is then read from the GPU, and saved as .png.
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
    The command bufferStaging is used to record commands, that will be submitted to a queue.

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
    The mandelbrot set will be rendered to 'bufferGPU' and then copied  from 'bufferGPU' to 'bufferStaging'.
    The memory that backs the bufferStaging is bufferMemoryStaging.
    The memory that backs the bufferGPU     is bufferMemoryGPU.
    */
    VkBuffer       bufferGPU, bufferStaging, bufferDynamic;
    VkDeviceMemory bufferMemoryGPU, bufferMemoryStaging, bufferMemoryDynamic;


    // we change this sample to work with textures
    //
    CustomVulkanTexture img;

    std::vector<const char *> enabledLayers;

    /*
    In order to execute commands on a device(GPU), the commands must be submitted
    to a queue. The commands are stored in a command bufferStaging, and this command bufferStaging
    is given to the queue. 

    There will be different kinds of queues on the device. Not all queues support
    graphics operations, for instance. For this application, we at least want a queue
    that supports compute operations. 
    */
    VkQueue queue; // a queue supporting compute operations.

public:
    ComputeApplication() : bufferDynamic(NULL), bufferMemoryDynamic(NULL) { }

    static void GetImageFromGPU(VkDevice a_device, VkDeviceMemory a_stagingMem, int w, int h,
                                uint32_t *imageData)
    {
      void *mappedMemory = nullptr;
      vkMapMemory(a_device, a_stagingMem, 0, w * h * sizeof(float) * 4, 0, &mappedMemory);
      Pixel* pmappedMemory = (Pixel *)mappedMemory;
      for (int i = 0; i < (w * h); i += 1)
      {
        const uint32_t r = ((uint32_t) (255.0f * (pmappedMemory[i].r)));
        const uint32_t g = ((uint32_t) (255.0f * (pmappedMemory[i].g)));
        const uint32_t b = ((uint32_t) (255.0f * (pmappedMemory[i].b)));
        imageData[i] = (r << 0) | (g << 8) | (b << 16);
      }
      // Done reading, so unmap.
      vkUnmapMemory(a_device, a_stagingMem);
    }

    static void PutImageToGPU(VkDevice a_device, VkDeviceMemory a_dynamicMem, int w, int h, const uint32_t *a_imageData)
    {
      void *mappedMemory = nullptr;
      vkMapMemory(a_device, a_dynamicMem, 0, w * h * sizeof(float) * 4, 0, &mappedMemory);
      float* pmappedMemory = (float*)mappedMemory;
      for (int i = 0; i < (w * h); i ++)
      {
        const uint32_t b = (a_imageData[i] & 0x00FF0000) >> 16;
        const uint32_t g = (a_imageData[i] & 0x0000FF00) >> 8;
        const uint32_t r = (a_imageData[i] & 0x000000FF);

        pmappedMemory[i*4+0] = float(r)*(1.0f/255.0f);
        pmappedMemory[i*4+1] = float(g)*(1.0f/255.0f);
        pmappedMemory[i*4+2] = float(b)*(1.0f/255.0f);
        pmappedMemory[i*4+3] = 0.0f;
      }
      // Done reading, so unmap.
      vkUnmapMemory(a_device, a_dynamicMem);
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


    static void CreateStagingBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, const size_t a_bufferSize,
                                    VkBuffer *a_pBuffer, VkDeviceMemory *a_pBufferMemory)
    {
      /*
      We will now create a bufferStaging. We will render the mandelbrot set into this bufferStaging
      in a computer shade later.
      */
      VkBufferCreateInfo bufferCreateInfo = {};
      bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferCreateInfo.size        = a_bufferSize; // bufferStaging size in bytes.
      bufferCreateInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // bufferStaging is used as a storage bufferStaging and we can _copy_to_ it. #NOTE this!
      bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // bufferStaging is exclusive to a single queue family at a time.

      VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, a_pBuffer)); // create bufferStaging.

      /*
      But the bufferStaging doesn't allocate memory for itself, so we must do that manually.
      First, we find the memory requirements for the bufferStaging.
      */
      VkMemoryRequirements memoryRequirements;
      vkGetBufferMemoryRequirements(a_device, (*a_pBuffer), &memoryRequirements);
        
      /*
      Now use obtained memory requirements info to allocate the memory for the bufferStaging.
      There are several types of memory that can be allocated, and we must choose a memory type that
      1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
      2) Satifies our own usage requirements. We want to be able to read the bufferStaging memory from the GPU to the CPU
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
        
      // Now associate that allocated memory with the bufferStaging. With that, the bufferStaging is backed by actual memory.
      VK_CHECK_RESULT(vkBindBufferMemory(a_device, (*a_pBuffer), (*a_pBufferMemory), 0));
    }


    static void CreateDynamicBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, const size_t a_bufferSize,
                                    VkBuffer *a_pBuffer, VkDeviceMemory *a_pBufferMemory)
    {
      VkBufferCreateInfo bufferCreateInfo = {};
      bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferCreateInfo.size        = a_bufferSize;
      bufferCreateInfo.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

      VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, a_pBuffer)); // create bufferStaging.

      VkMemoryRequirements memoryRequirements;
      vkGetBufferMemoryRequirements(a_device, (*a_pBuffer), &memoryRequirements);

      VkMemoryAllocateInfo allocateInfo = {};
      allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocateInfo.allocationSize  = memoryRequirements.size; // specify required memory.
      allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, a_physDevice);

      VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, a_pBufferMemory)); // allocate memory on device.

      VK_CHECK_RESULT(vkBindBufferMemory(a_device, (*a_pBuffer), (*a_pBufferMemory), 0));
    }


    static void CreateWriteOnlyBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, const size_t a_bufferSize,
                                      VkBuffer *a_pBuffer, VkDeviceMemory *a_pBufferMemory)
    {
      /*
      We will now create a bufferStaging. We will render the mandelbrot set into this bufferStaging
      in a computer shade later.
      */
      VkBufferCreateInfo bufferCreateInfo = {};
      bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferCreateInfo.size        = a_bufferSize;                         // bufferStaging size in bytes.
      bufferCreateInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;   // bufferStaging is used as a storage bufferStaging and we can _copy_from_ it. #NOTE this!
      bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;            // bufferStaging is exclusive to a single queue family at a time.

      VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, a_pBuffer)); // create bufferStaging.

      /*
      But the bufferStaging doesn't allocate memory for itself, so we must do that manually.
      First, we find the memory requirements for the bufferStaging.
      */
      VkMemoryRequirements memoryRequirements;
      vkGetBufferMemoryRequirements(a_device, (*a_pBuffer), &memoryRequirements);


      VkMemoryAllocateInfo allocateInfo = {};
      allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocateInfo.allocationSize  = memoryRequirements.size; // specify required memory.
      allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physDevice); // #NOTE VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT

      VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, a_pBufferMemory)); // allocate memory on device.

      // Now associate that allocated memory with the bufferStaging. With that, the bufferStaging is backed by actual memory.
      VK_CHECK_RESULT(vkBindBufferMemory(a_device, (*a_pBuffer), (*a_pBufferMemory), 0));
    }

    static void CreateDescriptorSetLayout(VkDevice a_device, VkDescriptorSetLayout *a_pDSLayout)
    {
       /*
       Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point 0.
       This binds to
         layout(std140, binding = 0) bufferStaging buf
       in the compute shader.

       the second descriptor is for our texture

       */
       VkDescriptorSetLayoutBinding descriptorSetLayoutBinding[2];
       descriptorSetLayoutBinding[0].binding            = 0;
       descriptorSetLayoutBinding[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
       descriptorSetLayoutBinding[0].descriptorCount    = 1;
       descriptorSetLayoutBinding[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
       descriptorSetLayoutBinding[0].pImmutableSamplers = nullptr;

       descriptorSetLayoutBinding[1].binding            = 1;
       descriptorSetLayoutBinding[1].descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
       descriptorSetLayoutBinding[1].descriptorCount    = 1;
       descriptorSetLayoutBinding[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
       descriptorSetLayoutBinding[1].pImmutableSamplers = nullptr;

       VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
       descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
       descriptorSetLayoutCreateInfo.bindingCount = 2; // buffer and texture
       descriptorSetLayoutCreateInfo.pBindings    = descriptorSetLayoutBinding;

       // Create the descriptor set layout.
       VK_CHECK_RESULT(vkCreateDescriptorSetLayout(a_device, &descriptorSetLayoutCreateInfo, NULL, a_pDSLayout));
    }

    void CreateDescriptorSetForOurBufferAndTexture(VkDevice a_device, VkBuffer a_buffer, size_t a_bufferSize, const VkDescriptorSetLayout *a_pDSLayout, VkImage a_image,
                                                   VkDescriptorPool *a_pDSPool, VkDescriptorSet *a_pDS, VkSampler *a_samplers, VkImageView *a_views)
    {
      /*
      So we will allocate a descriptor set here.
      But we need to first create a descriptor pool to do that.
      */

      /*
      Our descriptor pool can only allocate a storage buffer and a texture
      */
      VkDescriptorPoolSize descriptorPoolSize[2];
      descriptorPoolSize[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorPoolSize[0].descriptorCount = 1;
      descriptorPoolSize[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorPoolSize[1].descriptorCount = 1;


      VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
      descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      descriptorPoolCreateInfo.maxSets       = 1; // we need to allocate at least 1 descriptor set
      descriptorPoolCreateInfo.poolSizeCount = 2;
      descriptorPoolCreateInfo.pPoolSizes    = descriptorPoolSize;

      // create descriptor pool.
      VK_CHECK_RESULT(vkCreateDescriptorPool(a_device, &descriptorPoolCreateInfo, NULL, a_pDSPool));

      /*
      With the pool allocated, we can now allocate the descriptor set.
      */
      VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
      descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      descriptorSetAllocateInfo.descriptorPool     = (*a_pDSPool); // pool to allocate from.
      descriptorSetAllocateInfo.descriptorSetCount = 1;            // allocate a descriptor set for buffer and image
      descriptorSetAllocateInfo.pSetLayouts        = a_pDSLayout;

      // allocate descriptor set.
      VK_CHECK_RESULT(vkAllocateDescriptorSets(a_device, &descriptorSetAllocateInfo, a_pDS));

      /*
      Next, we need to connect our actual storage buffer with the descriptor.
      We use vkUpdateDescriptorSets() to update the descriptor set.
      */

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////// bind buffer

      // Specify the bufferStaging to bind to the descriptor.
      VkDescriptorBufferInfo descriptorBufferInfo = {};
      descriptorBufferInfo.buffer = a_buffer;
      descriptorBufferInfo.offset = 0;
      descriptorBufferInfo.range  = a_bufferSize;

      VkWriteDescriptorSet writeDescriptorSet = {};
      writeDescriptorSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writeDescriptorSet.dstSet          = *(a_pDS+0); // write to the first descriptor set.
      writeDescriptorSet.dstBinding      = 0;        // write to the first, and only binding.
      writeDescriptorSet.descriptorCount = 1;        // update a single descriptor.
      writeDescriptorSet.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage bufferStaging.
      writeDescriptorSet.pBufferInfo     = &descriptorBufferInfo;

      // perform the update of the descriptor set.
      vkUpdateDescriptorSets(a_device, 1, &writeDescriptorSet, 0, NULL);

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////// bind texture

      VkDescriptorImageInfo descriptorImageInfo = {};
      descriptorImageInfo.sampler     = img.imageSampler;
      descriptorImageInfo.imageView   = img.imageView;
      descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      VkWriteDescriptorSet writeDescriptorSet2 = {};
      writeDescriptorSet2.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writeDescriptorSet2.dstSet          = *(a_pDS+0); // write to second descriptor set.
      writeDescriptorSet2.dstBinding      = 1;          // second binding
      writeDescriptorSet2.descriptorCount = 1;          // update a single descriptor.
      writeDescriptorSet2.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // image.
      writeDescriptorSet2.pImageInfo      = &descriptorImageInfo;

      vkUpdateDescriptorSets(a_device, 1, &writeDescriptorSet2, 0, NULL);
    }

    static void CreateComputePipeline(VkDevice a_device, const VkDescriptorSetLayout &a_dsLayout,
                                      VkShaderModule *a_pShaderModule, VkPipeline *a_pPipeline,
                                      VkPipelineLayout *a_pPipelineLayout)
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

      //// Allow pass (w,h) inside shader directly from command buffer
      //
      VkPushConstantRange pcRange = {};    // #NOTE: we updated this to pass W/H inside shader
      pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      pcRange.offset     = 0;
      pcRange.size       = 2*sizeof(int);  // #NOTE: (w,h); pleas add more memory if you need more parameters!

      /*
      The pipeline layout allows the pipeline to access descriptor sets.
      So we just specify the descriptor set layout we created earlier.
      */
      VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
      pipelineLayoutCreateInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipelineLayoutCreateInfo.setLayoutCount         = 1;
      pipelineLayoutCreateInfo.pSetLayouts            = &a_dsLayout;
      pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
      pipelineLayoutCreateInfo.pPushConstantRanges    = &pcRange;
      VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &pipelineLayoutCreateInfo, NULL, a_pPipelineLayout));

      VkComputePipelineCreateInfo pipelineCreateInfo = {};
      pipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pipelineCreateInfo.stage  = shaderStageCreateInfo;
      pipelineCreateInfo.layout = (*a_pPipelineLayout);

      // Now, we finally create the compute pipeline.
      //
      VK_CHECK_RESULT(vkCreateComputePipelines(a_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, a_pPipeline));
    }

    static void CreateCommandBuffer(VkDevice a_device, uint32_t queueFamilyIndex, VkPipeline a_pipeline, VkPipelineLayout a_layout,
                                    VkCommandPool *a_pool, VkCommandBuffer *a_pCmdBuff)
    {
      /*
      We are getting closer to the end. In order to send commands to the device(GPU),
      we must first record commands into a command bufferStaging.
      To allocate a command bufferStaging, we must first create a command pool. So let us do that.
      */
      VkCommandPoolCreateInfo commandPoolCreateInfo = {};
      commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

      // the queue family of this command pool. All command buffers allocated from this command pool,
      // must be submitted to queues of this family ONLY.
      commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
      VK_CHECK_RESULT(vkCreateCommandPool(a_device, &commandPoolCreateInfo, NULL, a_pool));

      /*
      Now allocate a command bufferStaging from the command pool.
      */
      VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
      commandBufferAllocateInfo.sType       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      commandBufferAllocateInfo.commandPool = (*a_pool); // specify the command pool to allocate from.
      // if the command bufferStaging is primary, it can be directly submitted to queues.
      // A secondary bufferStaging has to be called from some primary command bufferStaging, and cannot be directly
      // submitted to a queue. To keep things simple, we use a primary command bufferStaging.
      commandBufferAllocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      commandBufferAllocateInfo.commandBufferCount = 1; // allocate a single command bufferStaging.
      VK_CHECK_RESULT(vkAllocateCommandBuffers(a_device, &commandBufferAllocateInfo, a_pCmdBuff)); // allocate command bufferStaging.
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    static VkImageMemoryBarrier imBarTransfer(VkImage a_image, const VkImageSubresourceRange& a_range, VkImageLayout before, VkImageLayout after) // VkImageLayout
    {
      VkImageMemoryBarrier moveToGeneralBar = {};
      moveToGeneralBar.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      moveToGeneralBar.pNext               = nullptr;
      moveToGeneralBar.srcAccessMask       = 0;
      moveToGeneralBar.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
      moveToGeneralBar.oldLayout           = before;
      moveToGeneralBar.newLayout           = after;
      moveToGeneralBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      moveToGeneralBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      moveToGeneralBar.image               = a_image;
      moveToGeneralBar.subresourceRange    = a_range;
      return moveToGeneralBar;
    }

    static VkImageSubresourceRange WholeImageRange()
    {
      VkImageSubresourceRange rangeWholeImage = {};
      rangeWholeImage.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      rangeWholeImage.baseMipLevel   = 0;
      rangeWholeImage.levelCount     = 1;
      rangeWholeImage.baseArrayLayer = 0;
      rangeWholeImage.layerCount     = 1;
      return rangeWholeImage;
    }

    static void RecordCommandsOfExecuteAndTransfer(VkCommandBuffer a_cmdBuff, VkPipeline a_pipeline,VkPipelineLayout a_layout, const VkDescriptorSet &a_ds, VkImage a_image,
                                                   size_t a_bufferSize, VkBuffer a_bufferGPU, VkBuffer a_bufferStaging)
    {
      /*
      Now we shall start recording commands into the newly allocated command bufferStaging.
      */
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the bufferStaging is only submitted and used once in this application.
      VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo)); // start recording commands.

      vkCmdFillBuffer(a_cmdBuff, a_bufferStaging, 0, a_bufferSize, 0); // clear this buffer just for an example and test cases. if we comment 'vkCmdCopyBuffer', we'll get black image

      /*
      We need to bind a pipeline, AND a descriptor set before we dispatch
      The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
      */
      vkCmdBindPipeline      (a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, a_pipeline);
      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, a_layout, 0, 1, &a_ds, 0, NULL);

      int wh[2] = {WIDTH,HEIGHT};
      vkCmdPushConstants(a_cmdBuff, a_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int)*2, wh);

      /*
      Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
      The number of workgroups is specified in the arguments.
      If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
      */
      vkCmdDispatch(a_cmdBuff, (uint32_t)ceil(WIDTH / float(WORKGROUP_SIZE)), (uint32_t)ceil(HEIGHT / float(WORKGROUP_SIZE)), 1);

      // copy data from bufferGPU to bufferStaging. #NOTE: this is new!!!
      // at first, transfer buffer from 'VK_ACCESS_SHADER_WRITE_BIT' to 'VK_ACCESS_TRANSFER_READ_BIT' state

      VkBufferMemoryBarrier bufBarr = {};
      bufBarr.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      bufBarr.pNext = nullptr;
      bufBarr.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bufBarr.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bufBarr.size                = VK_WHOLE_SIZE;
      bufBarr.offset              = 0;
      bufBarr.buffer              = a_bufferGPU;
      bufBarr.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
      bufBarr.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;

      vkCmdPipelineBarrier(a_cmdBuff,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0,
                           0, nullptr,
                           1, &bufBarr,
                           0, nullptr);

      VkBufferCopy copyInfo = {};
      copyInfo.dstOffset = 0;
      copyInfo.srcOffset = 0;
      copyInfo.size      = a_bufferSize;

      vkCmdCopyBuffer(a_cmdBuff, a_bufferGPU, a_bufferStaging, 1, &copyInfo);

      VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff)); // end recording commands.
    }


    static void RecordCommandsOfCopyImageDataToTexture(VkCommandBuffer a_cmdBuff, VkPipeline a_pipeline, int a_width, int a_height, VkBuffer a_bufferDynamic,
                                                       VkImage *a_images, VkBuffer a_bufferStaging)
    {
      //// Now we shall start recording commands into the newly allocated command bufferStaging.
      //
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the bufferStaging is only submitted and used once in this application.
      VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo)); // start recording commands.

      vkCmdFillBuffer(a_cmdBuff, a_bufferStaging, 0, a_width*a_height*sizeof(float)*4, 0); // clear this buffer just for an example and test cases. if we comment 'vkCmdCopyBuffer', we'll get black image

      // we want to work with the whole image
      //
      VkImageSubresourceRange rangeWholeImage = WholeImageRange();


      VkImageSubresourceLayers shittylayers = {};
      shittylayers.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      shittylayers.mipLevel       = 0;
      shittylayers.baseArrayLayer = 0;
      shittylayers.layerCount     = 1;

      VkBufferImageCopy wholeRegion = {};
      wholeRegion.bufferOffset      = 0;
      wholeRegion.bufferRowLength   = uint32_t(a_width);
      wholeRegion.bufferImageHeight = uint32_t(a_height);
      wholeRegion.imageExtent       = VkExtent3D{uint32_t(a_width), uint32_t(a_height), 1};
      wholeRegion.imageOffset       = VkOffset3D{0,0,0};
      wholeRegion.imageSubresource  = shittylayers;

      // at first we must move our images to 'VK_IMAGE_LAYOUT_GENERAL' layout to further clear them
      //
      VkImageMemoryBarrier moveToGeneralBar = imBarTransfer(a_images[0], rangeWholeImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

      vkCmdPipelineBarrier(a_cmdBuff,
                           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0,
                           0, nullptr,           // general memory barriers
                           0, nullptr,           // buffer barriers
                           1, &moveToGeneralBar); // image  barriers

      // now we can clear images
      //
      VkClearColorValue clearVal = {};
      clearVal.float32[0] = 1.0f;
      clearVal.float32[1] = 1.0f;
      clearVal.float32[2] = 1.0f;
      clearVal.float32[3] = 1.0f;

      vkCmdClearColorImage(a_cmdBuff, a_images[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &rangeWholeImage); // clear image with (1,1,1,1)

      vkCmdCopyBufferToImage(a_cmdBuff, a_bufferDynamic, a_images[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &wholeRegion);

      // transfer our texture from VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
      VkImageMemoryBarrier imgBar = {}; //imBarTransfer(a_images[0], rangeWholeImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      {
        imgBar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imgBar.pNext = nullptr;
        imgBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imgBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        imgBar.srcAccessMask       = 0;
        imgBar.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        imgBar.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imgBar.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imgBar.image               = a_images[0];

        imgBar.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        imgBar.subresourceRange.baseMipLevel   = 0;
        imgBar.subresourceRange.levelCount     = 1;
        imgBar.subresourceRange.baseArrayLayer = 0;
        imgBar.subresourceRange.layerCount     = 1;
      };

      vkCmdPipelineBarrier(a_cmdBuff,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0,
                           0, nullptr,
                           0, nullptr,
                           1, &imgBar);

      //VkImageMemoryBarrier barForCopy[2];
      //barForCopy[0] = imBarTransfer(a_images[0], rangeWholeImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      //barForCopy[1] = imBarTransfer(a_images[1], rangeWholeImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      //
      //vkCmdPipelineBarrier(a_cmdBuff,
      //                     VK_PIPELINE_STAGE_TRANSFER_BIT,
      //                     VK_PIPELINE_STAGE_TRANSFER_BIT,
      //                     0,
      //                     0, nullptr,     // general memory barriers
      //                     0, nullptr,     // buffer barriers
      //                     2, barForCopy); // image  barriers
      //
      //vkCmdCopyImageToBuffer(a_cmdBuff, a_images[1], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, a_bufferStaging, 1, &wholeRegion);


      VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff)); // end recording commands.
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    static void RunCommandBuffer(VkCommandBuffer a_cmdBuff, VkQueue a_queue, VkDevice a_device)
    {
      /*
      Now we shall finally submit the recorded command bufferStaging to a queue.
      */
      VkSubmitInfo submitInfo = {};
      submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submitInfo.commandBufferCount = 1; // submit a single command bufferStaging
      submitInfo.pCommandBuffers    = &a_cmdBuff; // the command bufferStaging to submit.

      /*
        We create a fence.
      */
      VkFence fence;
      VkFenceCreateInfo fenceCreateInfo = {};
      fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      fenceCreateInfo.flags = 0;
      VK_CHECK_RESULT(vkCreateFence(a_device, &fenceCreateInfo, NULL, &fence));

      /*
      We submit the command bufferStaging on the queue, at the same time giving a fence.
      */
      VK_CHECK_RESULT(vkQueueSubmit(a_queue, 1, &submitInfo, fence));

      /*
      The command will not have finished executing until the fence is signalled.
      So we wait here.
      We will directly after this read our bufferStaging from the GPU,
      and we will not be sure that the command has finished executing unless we wait for the fence.
      Hence, we use a fence here.
      */
      VK_CHECK_RESULT(vkWaitForFences(a_device, 1, &fence, VK_TRUE, 100000000000));

      vkDestroyFence(a_device, fence, NULL);
    }

    void Cleanup()
    {
      /*
      Clean up all Vulkan Resources.
      */
      if (enableValidationLayers)
      {
          // destroy callback.
          auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
          if (func == nullptr) {
              throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
          }
          func(instance, debugReportCallback, NULL);
      }

      if(bufferDynamic != NULL)
      {
        vkFreeMemory   (device, bufferMemoryDynamic, NULL);
        vkDestroyBuffer(device, bufferDynamic, NULL);
      }

      vkFreeMemory   (device, bufferMemoryStaging, NULL);
      vkDestroyBuffer(device, bufferStaging, NULL);

      vkFreeMemory   (device, bufferMemoryGPU, NULL);
      vkDestroyBuffer(device, bufferGPU, NULL);

      img.Release(device); // we have to explicitly destroy them before vkDestroyInstance and vkDestroyDevice

      vkDestroyShaderModule(device, computeShaderModule, NULL);
      vkDestroyDescriptorPool(device, descriptorPool, NULL);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
      vkDestroyPipelineLayout(device, pipelineLayout, NULL);
      vkDestroyPipeline(device, pipeline, NULL);
      vkDestroyCommandPool(device, commandPool, NULL);
      vkDestroyDevice(device, NULL);
      vkDestroyInstance(instance, NULL);
    }

    void Run()
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
     are grouped into queue families
     When submitting a command bufferStaging, you must specify to which queue in the family you are submitting to.
     This variable keeps track of the index of that queue in its family.
     */
     uint32_t queueFamilyIndex = vk_utils::GetComputeQueueFamilyIndex(physicalDevice);
     device = vk_utils::CreateLogicalDevice(queueFamilyIndex, physicalDevice, enabledLayers);
     vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

     // Buffer size of the storage bufferStaging that will contain the rendered mandelbrot set.
     size_t bufferSize = sizeof(Pixel) * WIDTH * HEIGHT;
     std::cout << "creating resources ... " << std::endl;
     CreateStagingBuffer(device, physicalDevice, bufferSize,      // very simple example of allocation
                         &bufferStaging, &bufferMemoryStaging);   // (device, bufferSize) ==> (bufferStaging, bufferMemoryStaging)

     CreateWriteOnlyBuffer(device, physicalDevice, bufferSize,    // very simple example of allocation
                           &bufferGPU, &bufferMemoryGPU);         // (device, bufferSize) ==> (bufferGPU, bufferMemoryGPU)

     // test image filter pipeline here ... temp solution
     //
     int w,h;
     auto imageData = LoadBMP("texture1.bmp", &w, &h);
     if(imageData.size() == 0)
     {
       std::cout << "can't load texture 'texture1.bmp' " << std::endl;
       return;
     }

     img = CustomVulkanTexture::Create2DTextureRGBA256(device, physicalDevice, w, h);


     CreateDescriptorSetLayout(device, &descriptorSetLayout);                                                 // here we will create a binding of bufferStaging to shader via descriptorSet
     CreateDescriptorSetForOurBufferAndTexture(device, bufferGPU, bufferSize, &descriptorSetLayout, img.imageGPU, // (device, bufferGPU, bufferSize, descriptorSetLayout) ==>  #NOTE: we write now to 'bufferGPU', not 'bufferStaging'
                                               &descriptorPool, &descriptorSet, &img.imageSampler, &img.imageView);   // (descriptorPool, descriptorSet, imageSamplers, imageViews)

     std::cout << "compiling shaders  ... " << std::endl;
     CreateComputePipeline(device, descriptorSetLayout,
                           &computeShaderModule, &pipeline, &pipelineLayout);

     CreateCommandBuffer(device, queueFamilyIndex, pipeline, pipelineLayout,
                         &commandPool, &commandBuffer);

     // load texture data to GPU
     {
       CreateDynamicBuffer(device, physicalDevice, w*h*sizeof(int),
                           &bufferDynamic, &bufferMemoryDynamic);

       //                                                                                  // bufferMemoryDynamic <== imageData.data()
       {
         void *mappedMemory = nullptr;
         vkMapMemory(device, bufferMemoryDynamic, 0, w*h*sizeof(int), 0, &mappedMemory);
         memcpy(mappedMemory, imageData.data(), w*h*sizeof(int));
         vkUnmapMemory(device, bufferMemoryDynamic);
       }

       vkResetCommandBuffer(commandBuffer, 0);
       RecordCommandsOfCopyImageDataToTexture(commandBuffer, pipeline, w, h, bufferDynamic, // bufferDynamic ==> imageGPU
                                              &img.imageGPU, bufferStaging);

       std::cout << "doing some computations ... " << std::endl;
       RunCommandBuffer(commandBuffer, queue, device);
     }

     // main shader
     {
       RecordCommandsOfExecuteAndTransfer(commandBuffer, pipeline, pipelineLayout, descriptorSet, img.imageGPU,
                                          bufferSize, bufferGPU, bufferStaging);
       // Finally, run the recorded command bufferStaging.
       std::cout << "doing computations ... " << std::endl;
       RunCommandBuffer(commandBuffer, queue, device);
       // The former command rendered a mandelbrot set to a bufferStaging.
       // Save that bufferStaging as a png on disk.
       std::cout << "geting image back  ... " << std::endl;
       //saveRenderedImageFromDeviceMemory(device, bufferMemoryStaging, 0, WIDTH, HEIGHT);
       std::vector<uint32_t> resultData(WIDTH*HEIGHT);
       GetImageFromGPU(device, bufferMemoryStaging, WIDTH, HEIGHT,
                       resultData.data());

       std::cout << "saving image       ... " << std::endl;
       SaveBMP("mandelbrot.bmp", resultData.data(), WIDTH, HEIGHT);
       resultData = std::vector<uint32_t>();
       std::cout << std::endl;
     }

     // Clean up all vulkan resources.
     std::cout << "destroying all     ... " << std::endl;
     Cleanup();
   }
};

int main()
{
  ComputeApplication app;

  try
  {
    app.Run();
  }
  catch (const std::runtime_error& e)
  {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }
    
  return EXIT_SUCCESS;
}
