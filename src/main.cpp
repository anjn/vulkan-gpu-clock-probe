#include <vulkan/vulkan.h>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#include <optional>
#include <thread>

#ifndef SHADER_SPV_PATH
#define SHADER_SPV_PATH "shaders/alu.comp.spv"
#endif

struct Args {
    uint32_t iters = 20000;            // per-thread ALU iterations
    uint32_t invocations = 1 << 20;    // total threads = global_invocations
    uint32_t local_size_x = 256;       // must match shader
    uint32_t loops = 120;              // number of samples (dispatches)
    uint32_t interval_ms = 1000;       // wait between samples
    uint32_t seed0 = 0x12345678;       // base seed for push constants
    std::string csv = "out.csv";      // output CSV
};

static Args parse_args(int argc, char** argv) {
    Args a{};
    for (int i = 1; i < argc; ++i) {
        auto eq = std::string(argv[i]);
        auto next = [&](){ return (i+1<argc)? std::string(argv[++i]) : std::string(); };
        if (eq == "--iters") a.iters = std::stoul(next());
        else if (eq == "--invocations") a.invocations = std::stoul(next());
        else if (eq == "--loops") a.loops = std::stoul(next());
        else if (eq == "--interval_ms") a.interval_ms = std::stoul(next());
        else if (eq == "--seed") a.seed0 = std::stoul(next());
        else if (eq == "--csv") a.csv = next();
        else {
            std::fprintf(stderr,
                "Unknown arg: %s\n"
                "Usage: --iters N --invocations N --loops N --interval_ms N --seed N --csv path\n",
                eq.c_str());
            std::exit(2);
        }
    }
    return a;
}

static std::vector<char> read_file(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror("fopen"); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf(sz);
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        std::perror("fread"); std::exit(1);
    }
    std::fclose(f);
    return buf;
}

static void vk_check(VkResult r, const char* where){ if(r!=VK_SUCCESS){ std::fprintf(stderr,"Vulkan error %d at %s\n", r, where); std::exit(1);} }

int main(int argc, char** argv){
    Args args = parse_args(argc, argv);

    // 1) Instance
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "clock-probe";
    ai.applicationVersion = VK_MAKE_VERSION(0,1,0);
    ai.pEngineName = "none";
    ai.engineVersion = VK_MAKE_VERSION(0,0,0);
    ai.apiVersion = VK_API_VERSION_1_2;

    const char* enabledExts[] = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = enabledExts;

    VkInstance instance{}; vk_check(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");

    // 2) Physical device + compute queue
    uint32_t physCount=0; vkEnumeratePhysicalDevices(instance, &physCount, nullptr);
    if(!physCount){ std::fprintf(stderr, "No Vulkan devices found\n"); return 1; }
    std::vector<VkPhysicalDevice> phys(physCount); vkEnumeratePhysicalDevices(instance,&physCount, phys.data());

    uint32_t chosenPhys = UINT32_MAX, chosenQueueFamily = UINT32_MAX;
    VkPhysicalDeviceProperties chosenProps{};

    for (uint32_t i=0;i<physCount;++i){
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys[i], &props);
        uint32_t qfCount=0; vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qfCount); vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &qfCount, qf.data());
        for(uint32_t q=0;q<qfCount;++q){
            if(qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT){
                // Prefer families that support timestamps
                VkPhysicalDeviceLimits lim = props.limits;
                // Note: validBits are per-queue; spec doesn’t expose per-family here, but most support timestamps.
                chosenPhys = i; chosenQueueFamily = q; chosenProps = props; goto found;
            }
        }
    }
found:
    if(chosenPhys==UINT32_MAX){ std::fprintf(stderr, "No compute queue found\n"); return 1; }

    VkPhysicalDevice pdev = phys[chosenPhys];
    std::printf("Using device: %s (api %u.%u)\n", chosenProps.deviceName,
                VK_VERSION_MAJOR(chosenProps.apiVersion), VK_VERSION_MINOR(chosenProps.apiVersion));

    // 3) Device + queue
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = chosenQueueFamily; qci.queueCount = 1; qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;

    VkDevice dev{}; vk_check(vkCreateDevice(pdev, &dci, nullptr, &dev), "vkCreateDevice");
    VkQueue queue{}; vkGetDeviceQueue(dev, chosenQueueFamily, 0, &queue);

    // 4) Resources
    // Storage buffer (device-local). We won’t read back during the run.
    VkDeviceSize bytes = sizeof(uint32_t) * args.invocations;

    auto find_mem_type = [&](uint32_t typeBits, VkMemoryPropertyFlags req) -> uint32_t {
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
        for(uint32_t i=0;i<mp.memoryTypeCount;++i){
            if((typeBits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & req)==req) return i;
        }
        std::fprintf(stderr, "No suitable memory type\n"); std::exit(1);
    };

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buf{}; vk_check(vkCreateBuffer(dev,&bci,nullptr,&buf), "vkCreateBuffer");
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(dev, buf, &mr);

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory mem{}; vk_check(vkAllocateMemory(dev,&mai,nullptr,&mem), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(dev, buf, mem, 0), "vkBindBufferMemory");

    // Descriptor set layout
    VkDescriptorSetLayoutBinding bind0{}; bind0.binding=0; bind0.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bind0.descriptorCount=1; bind0.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount=1; dslci.pBindings=&bind0;
    VkDescriptorSetLayout dsl{}; vk_check(vkCreateDescriptorSetLayout(dev,&dslci,nullptr,&dsl), "vkCreateDescriptorSetLayout");

    // Pipeline layout with push constants (iters, seed)
    VkPushConstantRange pcr{}; pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.offset=0; pcr.size=sizeof(uint32_t)*2;

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount=1; plci.pSetLayouts=&dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VkPipelineLayout pl{}; vk_check(vkCreatePipelineLayout(dev,&plci,nullptr,&pl), "vkCreatePipelineLayout");

    // Shader module
    auto spv = read_file(SHADER_SPV_PATH);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size(); smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule sm{}; vk_check(vkCreateShaderModule(dev,&smci,nullptr,&sm), "vkCreateShaderModule");

    // Pipeline
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipe{}; vk_check(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe), "vkCreateComputePipelines");

    // Descriptor pool + set
    VkDescriptorPoolSize dps{}; dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; dps.descriptorCount=1;
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=1; dpci.poolSizeCount=1; dpci.pPoolSizes=&dps;
    VkDescriptorPool dp{}; vk_check(vkCreateDescriptorPool(dev,&dpci,nullptr,&dp), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool=dp; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
    VkDescriptorSet ds{}; vk_check(vkAllocateDescriptorSets(dev,&dsai,&ds), "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo dbi{}; dbi.buffer=buf; dbi.offset=0; dbi.range=bytes;
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet=ds; wds.dstBinding=0; wds.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds.descriptorCount=1; wds.pBufferInfo=&dbi;
    vkUpdateDescriptorSets(dev,1,&wds,0,nullptr);

    // Command pool/buffer
    VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci2.queueFamilyIndex = chosenQueueFamily; cpci2.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmdPool{}; vk_check(vkCreateCommandPool(dev,&cpci2,nullptr,&cmdPool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=cmdPool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cmd{}; vk_check(vkAllocateCommandBuffers(dev,&cbai,&cmd), "vkAllocateCommandBuffers");

    // Query pool (timestamps)
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = 2;
    VkQueryPool qp{}; vk_check(vkCreateQueryPool(dev,&qpci,nullptr,&qp), "vkCreateQueryPool");

    // Fence for submission
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence{}; vk_check(vkCreateFence(dev,&fci,nullptr,&fence), "vkCreateFence");

    // CSV header
    FILE* csv = std::fopen(args.csv.c_str(), "w");
    if(!csv){ std::perror("fopen csv"); return 1; }
    std::fprintf(csv, "wall_time_iso,duration_ms,rel_freq,iters,invocations\n");
    std::fflush(csv);

    const double tsPeriodNs = chosenProps.limits.timestampPeriod; // ns per tick

    // Compute grid
    uint32_t groups = (args.invocations + args.local_size_x - 1) / args.local_size_x;

    // Warm-up (optional)
    auto record_once = [&](uint32_t iters, uint32_t seed){
        vk_check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vk_check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");

        vkCmdResetQueryPool(cmd, qp, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        struct PC { uint32_t iters; uint32_t seed; } pc{iters, seed};
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups, 1, 1);

        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
        vk_check(vkResetFences(dev,1,&fence), "vkResetFences");
        vk_check(vkQueueSubmit(queue,1,&si,fence), "vkQueueSubmit");
        vk_check(vkWaitForFences(dev,1,&fence, VK_TRUE, UINT64_C(60)*1000*1000*1000), "vkWaitForFences");

        uint64_t ts[2] = {0, 0}; // timestamp ticks
        vk_check(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
        double ns = double(ts[1] - ts[0]) * tsPeriodNs;
        return ns; // nanoseconds
    };

    // Baseline measure for relative frequency estimation
    double baseline_ns = -1;
    //record_once(args.iters, args.seed0);
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    auto iso_now = [](){
        using clock = std::chrono::system_clock;
        auto t = clock::now();
        std::time_t tt = clock::to_time_t(t);
        std::tm tm{};
    #ifdef _WIN32
        localtime_s(&tm, &tt);
    #else
        localtime_r(&tt, &tm);
    #endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
        return std::string(buf);
    };

    // Main sampling loop
    for (uint32_t i=0;i<args.loops;++i){
        uint32_t seed = args.seed0 + i*16699u;
        double ns = record_once(args.iters, seed);
        double ms = ns / 1.0e6;
        if (baseline_ns < 0) baseline_ns = ns;
        double rel_freq = baseline_ns / ns; // >1.0 means faster than baseline
        std::printf("[%4u/%4u] %.3f ms  (rel_freq=%.3f)\n", i+1, args.loops, ms, rel_freq);
        std::fprintf(csv, "%s,%.6f,%.6f,%u,%u\n", iso_now().c_str(), ms, rel_freq, args.iters, args.invocations);
        std::fflush(csv);
        if (i+1<args.loops) {
            std::this_thread::sleep_for(std::chrono::milliseconds(args.interval_ms));
        }
    }

    std::fclose(csv);

    // Cleanup
    vkDeviceWaitIdle(dev);
    vkDestroyFence(dev,fence,nullptr);
    vkDestroyQueryPool(dev,qp,nullptr);
    vkDestroyCommandPool(dev,cmdPool,nullptr);
    vkDestroyDescriptorPool(dev,dp,nullptr);
    vkDestroyPipeline(dev,pipe,nullptr);
    vkDestroyShaderModule(dev,sm,nullptr);
    vkDestroyPipelineLayout(dev,pl,nullptr);
    vkDestroyDescriptorSetLayout(dev,dsl,nullptr);
    vkDestroyBuffer(dev,buf,nullptr);
    vkFreeMemory(dev,mem,nullptr);
    vkDestroyDevice(dev,nullptr);
    vkDestroyInstance(instance,nullptr);
    return 0;
}
