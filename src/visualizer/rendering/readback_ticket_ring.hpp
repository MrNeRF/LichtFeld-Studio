/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "generic_readback_ticket_ring.hpp"
#include <vulkan/vulkan.h>

namespace lfs::vis {
    // Legacy renderer specialization; the state machine carries no native API.
    using ReadbackTicketRing = BasicReadbackTicketRing<VkImage>;
} // namespace lfs::vis
