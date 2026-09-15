// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Execute the production LOD selector against trees whose projected sizes are
// non-monotone. Usage: c++ -std=c++20 scripts/verify_lod_gpu_cut.cpp -lvulkan -o
// /tmp/verify_lod_gpu_cut && /tmp/verify_lod_gpu_cut path/to/lod_select_threshold.spv
// Add --single-pass when checking the old shader with its original dispatch sequence.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

static void check(VkResult r) {
    if (r != VK_SUCCESS)
        throw std::runtime_error("Vulkan error " + std::to_string(r));
}
int main(int argc, char** argv) {
    if ((argc != 2 && argc != 3) || (argc == 3 && std::strcmp(argv[2], "--single-pass") != 0))
        return 2;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance;
    check(vkCreateInstance(&ici, nullptr, &instance));
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
    if (!count)
        throw std::runtime_error("No Vulkan device");
    VkPhysicalDevice physical = devices[0];
    for (auto candidate : devices) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(candidate, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            physical = candidate;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical, &props);
    std::cout << props.deviceName << '\n';
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    uint32_t family = 0;
    while (!(families.at(family).queueFlags & VK_QUEUE_COMPUTE_BIT))
        ++family;
    float priority = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice device;
    check(vkCreateDevice(physical, &dci, nullptr, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);
    std::array<VkBuffer, 12> buffers{};
    std::array<VkDeviceMemory, 12> memory{};
    std::array<void*, 12> mapped{};
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physical, &mp);
    for (size_t i = 0; i < buffers.size(); ++i) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = 4096;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        check(vkCreateBuffer(device, &bci, nullptr, &buffers[i]));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, buffers[i], &req);
        uint32_t type = 0;
        constexpr auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        while (!(req.memoryTypeBits & (1u << type)) || (mp.memoryTypes[type].propertyFlags & flags) != flags)
            ++type;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        check(vkAllocateMemory(device, &ai, nullptr, &memory[i]));
        check(vkBindBufferMemory(device, buffers[i], memory[i], 0));
        check(vkMapMemory(device, memory[i], 0, req.size, 0, &mapped[i]));
    }
    std::array<VkDescriptorSetLayoutBinding, 12> bindings{};
    for (uint32_t i = 0; i < 12; ++i)
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    slci.bindingCount = 12;
    slci.pBindings = bindings.data();
    VkDescriptorSetLayout set_layout;
    check(vkCreateDescriptorSetLayout(device, &slci, nullptr, &set_layout));
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 144};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &range;
    VkPipelineLayout layout;
    check(vkCreatePipelineLayout(device, &plci, nullptr, &layout));
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("Cannot open shader");
    std::vector<uint32_t> code(static_cast<size_t>(input.tellg()) / 4);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(code.data()), code.size() * 4);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size() * 4;
    smci.pCode = code.data();
    VkShaderModule shader;
    check(vkCreateShaderModule(device, &smci, nullptr, &shader));
    VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pci.layout = layout;
    pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", nullptr};
    VkPipeline pipeline;
    check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline));
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    VkDescriptorPool pool;
    check(vkCreateDescriptorPool(device, &dpci, nullptr, &pool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &set_layout;
    VkDescriptorSet set;
    check(vkAllocateDescriptorSets(device, &dsai, &set));
    std::array<VkDescriptorBufferInfo, 12> infos{};
    std::array<VkWriteDescriptorSet, 12> writes{};
    for (uint32_t i = 0; i < 12; ++i) {
        infos[i] = {buffers[i], 0, 4096};
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[i], nullptr};
    }
    vkUpdateDescriptorSets(device, 12, writes.data(), 0, nullptr);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = family;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cp;
    check(vkCreateCommandPool(device, &cpci, nullptr, &cp));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cp;
    cbai.commandBufferCount = 1;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VkCommandBuffer command;
    check(vkAllocateCommandBuffers(device, &cbai, &command));
    bool passed = true;
    for (int scenario = 0; scenario < 6; ++scenario) {
        for (auto p : mapped)
            std::memset(p, 0, 4096);
        auto words = [&](int binding) { return static_cast<uint32_t*>(mapped[binding]); };
        // 0 -> {1,2}; 1 -> 3; 3 -> 4. Node 1 stops refinement while node
        // 3's projected size is larger. Node 4 must not form a second cut.
        const uint32_t invalid = 0xffffffffu;
        const uint32_t links[] = {1, 2, invalid, 3, 1, 0, 0, 0, 0, 4, 1, 1, 0, 0, 3};
        std::memcpy(mapped[1], links, sizeof(links));
        const float sizes[] = {scenario == 5 ? 1.0e30f : 20.0f, scenario == 2 ? 10.0f : .1f, .1f, scenario == 2 ? .1f : 10.0f, .1f};
        for (uint32_t i = 0; i < 5; ++i) {
            words(2)[i] = words(11)[i] = i;
            auto frame = static_cast<float*>(mapped[10]) + i * 16;
            frame[6] = -10;
            frame[7] = std::log(sizes[i]);
        }
        if (scenario == 1) {
            words(2)[1] = invalid;
            words(11)[1] = invalid;
        }
        // Exact 144-byte production push-constant layout.
        std::array<uint32_t, 36> uniforms{};
        auto put_float = [&](size_t i, float f) { std::memcpy(&uniforms[i], &f, 4); };
        uniforms[0] = 5;
        uniforms[1] = scenario >= 3 ? 1 : 2;
        uniforms[2] = 1;
        uniforms[3] = invalid;
        put_float(4, scenario >= 4 ? 1.0e-12f : .05f);
        put_float(5, 1);
        put_float(6, 1);
        put_float(7, 1);
        put_float(8, -1);
        put_float(12, 1);
        put_float(17, 1);
        put_float(22, 1);
        uniforms[31] = 5;
        uniforms[32] = 5;
        check(vkResetCommandBuffer(command, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(command, &begin));
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uniforms), uniforms.data());
        vkCmdDispatch(command, 1, 1, 1);
        if (scenario >= 3 && argc == 2) {
            const auto dependency = [&] {
                VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
                b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                     0, 1, &b, 0, nullptr, 0, nullptr);
            };
            for (uint32_t pass = 1; pass <= 16; ++pass) {
                dependency();
                uniforms[35] = pass;
                vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uniforms), uniforms.data());
                vkCmdDispatch(command, 1, 1, 1);
                dependency();
                uniforms[35] = 17;
                vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uniforms), uniforms.data());
                vkCmdDispatchIndirect(command, buffers[3], 3 * sizeof(uint32_t));
            }
        }
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(queue));
        std::set<uint32_t> selected;
        for (uint32_t i = 0; i < std::min(words(3)[0], uniforms[1]); ++i)
            selected.insert(words(5)[i]);
        const std::set<uint32_t> expected = (scenario == 1 || scenario >= 3) ? std::set<uint32_t>{0} : (scenario == 2 ? std::set<uint32_t>{2, 3} : std::set<uint32_t>{1, 2});
        const bool ok = selected == expected && words(3)[0] == expected.size() && words(3)[1] == 0;
        passed &= ok;
        std::cout << "scenario " << scenario << ": candidates=" << words(3)[0] << " overflow=" << words(3)[1] << " selected=";
        for (auto n : selected)
            std::cout << n << ',';
        std::cout << (ok ? " PASS\n" : " FAIL\n");
    }
    vkDestroyCommandPool(device, cp, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
    for (size_t i = 0; i < 12; ++i) {
        vkUnmapMemory(device, memory[i]);
        vkDestroyBuffer(device, buffers[i], nullptr);
        vkFreeMemory(device, memory[i], nullptr);
    }
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return passed ? 0 : 1;
}
