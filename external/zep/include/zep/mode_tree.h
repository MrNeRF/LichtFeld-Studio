#pragma once

#include "mode.h"
#include "zep/mode_vim.h"

namespace Zep {

    struct ZepRepl;

    namespace ZepTreeNodeFlags {
        enum {
            None = (0),
            IsFolder = (1 << 0)
        };
    }

    class ZepTreeNode {
    public:
        using TNode = std::shared_ptr<ZepTreeNode>;
        using TChildren = std::vector<TNode>;

        ZepTreeNode(const std::string& strName, uint32_t flags = ZepTreeNodeFlags::None);

        virtual const std::string& GetName() const {
            return m_strName;
        }
        virtual void SetName(const std::string& name) {
            m_strName = name;
        }
        virtual ZepTreeNode* GetParent() const {
            return m_pParent;
        }
        virtual TChildren GetChildren() const {
            return m_children;
        }
        virtual bool HasChildren() const {
            return !m_children.empty();
        }

        virtual ZepTreeNode* AddChild(const TNode& pNode) {
            m_children.push_back(pNode);
            pNode->m_pParent = this;
            return pNode.get();
        }

        virtual bool RemoveChild(ZepTreeNode* pNode) {
            auto itr = std::find_if(m_children.begin(), m_children.end(), [=](TNode& node) {
                if (node.get() == pNode) {
                    return true;
                }
                return false;
            });

            if (itr != m_children.end()) {
                m_children.erase(itr);
                pNode->SetParent(nullptr);
                return true;
            }
            return false;
        }

        virtual void ClearChildren() {
            m_children.clear();
        }

        virtual bool IsExpanded() const {
            return m_expanded;
        }

        virtual void Expand(bool expand) {
            m_expanded = expand;
        }

        virtual void ExpandAll(bool expand) {
            using fnVisit = std::function<void(ZepTreeNode * pNode, bool ex)>;
            fnVisit visitor = [&](ZepTreeNode* pNode, bool ex) {
                pNode->Expand(ex);
                if (pNode->HasChildren()) {
                    for (auto& pChild : pNode->GetChildren()) {
                        visitor(pChild.get(), ex);
                    }
                }
            };
            visitor(this, expand);
        }

        virtual void SetParent(ZepTreeNode* pParent) {
            m_pParent = pParent;
        }

    protected:
        bool m_expanded = false;
        ZepTreeNode* m_pParent = nullptr;
        TChildren m_children;
        std::string m_strName;
        uint32_t m_flags = ZepTreeNodeFlags::None;
    };

    class ZepTree {
    public:
        virtual ZepTreeNode* GetRoot() const {
            return m_spRoot.get();
        }

    protected:
        ZepTreeNode::TNode m_spRoot;
    };

    class ZepFileNode : public ZepTreeNode {
    public:
        ZepFileNode(const std::string& name, uint32_t flags = ZepTreeNodeFlags::None)
            : ZepTreeNode(name, flags) {
        }
    };

    class ZepFileTree : public ZepTree {
    public:
        ZepFileTree();
    };

    class ZepMode_Tree : public ZepMode_Vim {
    public:
        ZepMode_Tree(ZepEditor& editor, std::shared_ptr<ZepTree> spTree, ZepWindow& launchWindow, ZepWindow& replWindow);
        ~ZepMode_Tree();

        static const char* StaticName() {
            return "TREE";
        }
        virtual void Begin(ZepWindow* pWindow) override;
        virtual void Notify(std::shared_ptr<ZepMessage> message) override;
        virtual const char* Name() const override {
            return StaticName();
        }

    private:
        void BuildTree();

    private:
        std::shared_ptr<ZepTree> m_spTree;
        GlyphIterator m_startLocation = GlyphIterator{0};
        ZepWindow& m_launchWindow;
        ZepWindow& m_window;
    };

} // namespace Zep
