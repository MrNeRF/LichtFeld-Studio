#include "zep/editor.h"
#include "zep/filesystem.h"
#include "zep/syntax.h"
#include "zep/tab_window.h"
#include "zep/window.h"

#include "zep/mcommon/logger.h"
#include "zep/mcommon/threadutils.h"

#include "zep/mode_repl.h"

namespace Zep {

    ZepReplEvaluateOuterCommand::ZepReplEvaluateOuterCommand(ZepEditor& editor, IZepReplProvider* pProvider)
        : ZepExCommand(editor), m_pProvider(pProvider) {
        keymap_add(m_keymap, {"<C-Return>"}, ExCommandId());
    }

    void ZepReplEvaluateOuterCommand::Register(ZepEditor& editor, IZepReplProvider* pProvider) {
        editor.RegisterExCommand(std::make_shared<ZepReplEvaluateOuterCommand>(editor, pProvider));
    }

    void ZepReplEvaluateOuterCommand::Run(const std::vector<std::string>& tokens) {
        ZEP_UNUSED(tokens);
        if (!GetEditor().GetActiveTabWindow()) {
            return;
        }

        auto& buffer = GetEditor().GetActiveTabWindow()->GetActiveWindow()->GetBuffer();
        auto cursor = GetEditor().GetActiveTabWindow()->GetActiveWindow()->GetBufferCursor();

        m_pProvider->ReplParse(buffer, cursor, ReplParseType::OuterExpression);
    }

    ZepReplEvaluateCommand::ZepReplEvaluateCommand(ZepEditor& editor, IZepReplProvider* pProvider)
        : ZepExCommand(editor), m_pProvider(pProvider) {
        keymap_add(m_keymap, {"<S-Return>"}, ExCommandId());
    }

    void ZepReplEvaluateCommand::Register(ZepEditor& editor, IZepReplProvider* pProvider) {
        editor.RegisterExCommand(std::make_shared<ZepReplEvaluateCommand>(editor, pProvider));
    }

    void ZepReplEvaluateCommand::Run(const std::vector<std::string>& tokens) {
        ZEP_UNUSED(tokens);
        if (!GetEditor().GetActiveTabWindow()) {
            return;
        }

        auto& buffer = GetEditor().GetActiveTabWindow()->GetActiveWindow()->GetBuffer();
        auto cursor = GetEditor().GetActiveTabWindow()->GetActiveWindow()->GetBufferCursor();

        m_pProvider->ReplParse(buffer, cursor, ReplParseType::All);
    }

    ZepReplEvaluateInnerCommand::ZepReplEvaluateInnerCommand(ZepEditor& editor, IZepReplProvider* pProvider)
        : ZepExCommand(editor), m_pProvider(pProvider) {
        keymap_add(m_keymap, {"<C-S-Return>"}, ExCommandId());
    }

    void ZepReplEvaluateInnerCommand::Register(ZepEditor& editor, IZepReplProvider* pProvider) {
        editor.RegisterExCommand(std::make_shared<ZepReplEvaluateInnerCommand>(editor, pProvider));
    }

    void ZepReplEvaluateInnerCommand::Run(const std::vector<std::string>& tokens) {
        ZEP_UNUSED(tokens);
        if (!GetEditor().GetActiveTabWindow()) {
            return;
        }

        auto& buffer = GetEditor().GetActiveTabWindow()->GetActiveWindow()->GetBuffer();
        auto cursor = GetEditor().GetActiveTabWindow()->GetActiveWindow()->GetBufferCursor();

        m_pProvider->ReplParse(buffer, cursor, ReplParseType::SubExpression);
    }
    const std::string PromptString = ">> ";
    const std::string ContinuationString = ".. ";

    ZepReplExCommand::ZepReplExCommand(ZepEditor& editor, IZepReplProvider* pProvider)
        : ZepExCommand(editor), m_pProvider(pProvider) {
        keymap_add(m_keymap, {"<c-e>"}, ExCommandId());
    }

    void ZepReplExCommand::Register(ZepEditor& editor, IZepReplProvider* pProvider) {
        editor.RegisterExCommand(std::make_shared<ZepReplExCommand>(editor, pProvider));
    }

    void ZepReplExCommand::Run(const std::vector<std::string>& tokens) {
        ZEP_UNUSED(tokens);
        if (!GetEditor().GetActiveTabWindow()) {
            return;
        }

        // TODO: Modifiable but not saveable buffer
        m_pReplBuffer = GetEditor().GetEmptyBuffer("Repl.lisp"); // , FileFlags::ReadOnly);
        m_pReplBuffer->SetBufferType(BufferType::Repl);
        m_pReplBuffer->GetSyntax()->IgnoreLineHighlight();
        m_pReplBuffer->SetPostKeyNotifier([&](uint32_t key, uint32_t modifier) {
            return AddKeyPress(key, modifier);
        });

        // Adding the window will make it active and begin the mode
        m_pReplWindow = GetEditor().GetActiveTabWindow()->AddWindow(m_pReplBuffer, nullptr, RegionLayoutType::VBox);
        Prompt();
    }

    void ZepReplExCommand::Prompt() {
        // TODO: Repl broken, but when not, need to consider undo
        ChangeRecord changeRecord;
        m_pReplBuffer->Insert(m_pReplBuffer->End(), PromptString, changeRecord);
        MoveToEnd();
    }

    void ZepReplExCommand::MoveToEnd() {
        m_pReplWindow->SetBufferCursor(m_pReplBuffer->End());
        m_startLocation = m_pReplWindow->GetBufferCursor();
    }

    bool ZepReplExCommand::AddKeyPress(uint32_t key, uint32_t modifiers) {
        (void)&modifiers;
        if (key == ExtKeys::RETURN) {
            ChangeRecord record;
            auto& buffer = m_pReplWindow->GetBuffer();
            std::string str = std::string(buffer.GetWorkingBuffer().begin() + m_startLocation.Index(), buffer.GetWorkingBuffer().end());
            if (str.size() <= 1) {
                MoveToEnd();
                buffer.GetMode()->SwitchMode(EditorMode::Insert);
                return false;
            }

            auto stripLineStarts = [](std::string& str) {
                bool newline = true;
                int pos = 0;
                while (pos < str.size()) {
                    if (str[pos] == '\n')
                        newline = true;
                    else if (newline) {
                        if (str.find(PromptString, pos) == pos) {
                            str.erase(pos, PromptString.length());
                            continue;
                        } else if (str.find(ContinuationString, pos) == pos) {
                            str.erase(pos, ContinuationString.length());
                            continue;
                        }
                        newline = false;
                    }
                    pos++;
                }
            };
            stripLineStarts(str);

            std::string ret;
            if (m_pProvider) {
                int indent = 0;
                bool complete = m_pProvider->ReplIsFormComplete(str, indent);
                if (!complete) {
                    // If the indent is < 0, we completed too much of the expression, so don't let the user hit return until they
                    // fix it.  Example in lisp: (+ 2 2))  This expression has 'too many' close brackets.
                    if (indent < 0) {
                        buffer.Delete(buffer.End() - 1, buffer.End(), record);
                        m_pReplWindow->SetBufferCursor(buffer.End());
                        return true;
                    }

                    // New line continuation symbol
                    buffer.Insert(buffer.End(), ContinuationString, record);

                    // Indent by how far the repl suggests
                    if (indent > 0) {
                        for (int i = 0; i < indent; i++) {
                            buffer.Insert(buffer.End(), " ", record);
                        }
                    }
                    m_pReplWindow->SetBufferCursor(buffer.End());
                    return true;
                }
                ret = m_pProvider->ReplParse(str);
            } else {
                ret = str;
            }

            if (!ret.empty() && ret[0] != 0) {
                ret.push_back('\n');
                buffer.Insert(buffer.End(), ret, record);
            }

            Prompt();
            return true;
        }

        return false;
    }

} // namespace Zep
