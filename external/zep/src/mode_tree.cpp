#include "zep/mode_tree.h"

namespace Zep {

    ZepTreeNode::ZepTreeNode(const std::string& strName, uint32_t flags)
        : m_strName(strName), m_flags(flags) {
    }

    ZepFileTree::ZepFileTree() {
        // Root node is 'invisible' and always expanded
        m_spRoot = std::make_shared<ZepFileNode>("Root");
        m_spRoot->Expand(true);
    }

    ZepMode_Tree::ZepMode_Tree(ZepEditor& editor, std::shared_ptr<ZepTree> spTree, ZepWindow& launchWindow, ZepWindow& window)
        : ZepMode_Vim(editor), m_spTree(spTree), m_launchWindow(launchWindow), m_window(window) {
    }

    ZepMode_Tree::~ZepMode_Tree() {
    }

    void ZepMode_Tree::Notify(std::shared_ptr<ZepMessage> message) {
    }

    void ZepMode_Tree::BuildTree() {
        auto& buffer = m_window.GetBuffer();

        std::ostringstream strBuffer;
        std::function<void(ZepTreeNode*, uint32_t indent)> fnVisit;

        fnVisit = [&](ZepTreeNode* pNode, uint32_t indent) {
            for (uint32_t i = 0; i < indent; i++) {
                strBuffer << " ";
            }

            if (pNode->HasChildren()) {
                strBuffer << (pNode->IsExpanded() ? "~ " : "+ ");
            } else {
                strBuffer << "  ";
            }

            strBuffer << pNode->GetName() << std::endl;

            if (pNode->IsExpanded()) {
                for (auto& pChild : pNode->GetChildren()) {
                    fnVisit(pChild.get(), indent + 2);
                }
            }
        };

        if (m_spTree->GetRoot()->IsExpanded()) {
            for (auto pChild : m_spTree->GetRoot()->GetChildren()) {
                fnVisit(pChild.get(), 0);
            }
        }

        ChangeRecord tempRecord;
        buffer.Clear();
        buffer.Insert(buffer.Begin(), strBuffer.str(), tempRecord);
    }

    void ZepMode_Tree::Begin(ZepWindow* pWindow) {
        ZepMode_Vim::Begin(pWindow);

        BuildTree();
    }

} // namespace Zep
