#pragma once

#include "syntax.h"

namespace Zep {

    class ZepSyntax_Markdown : public ZepSyntax {
    public:
        ZepSyntax_Markdown(ZepBuffer& buffer,
                           const std::unordered_set<std::string>& keywords = std::unordered_set<std::string>{},
                           const std::unordered_set<std::string>& identifiers = std::unordered_set<std::string>{},
                           uint32_t flags = 0);

        virtual void UpdateSyntax() override;
    };

} // namespace Zep
