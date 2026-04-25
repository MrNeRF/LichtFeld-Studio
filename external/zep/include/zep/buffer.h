#pragma once

#include <functional>
#include <set>
#include <stack>

#include "zep/mcommon/logger.h"
#include "zep/mcommon/signals.h"
#include "zep/mcommon/string/stringutils.h"

#include "zep/glyph_iterator.h"

#include "zep/editor.h"
#include "zep/line_widgets.h"
#include "zep/range_markers.h"

namespace Zep {

    class ZepSyntax;
    class ZepTheme;
    class ZepMode;
    class ZepCommand;
    enum class ThemeColor;

    enum class Direction {
        Forward,
        Backward
    };

    namespace SearchType {
        enum : uint32_t {
            WORD = (1 << 0),
            Begin = (1 << 1),
            End = (1 << 2),
            Word = (1 << 3),
            SingleLine = (1 << 4)
        };
    };

    namespace FileFlags {
        enum : uint32_t {
            StrippedCR = (1 << 0),
            TerminatedWithZero = (1 << 1),
            ReadOnly = (1 << 2),
            Locked = (1 << 3), // Can this file path ever be written to?
            Dirty = (1 << 4),  // Has the file been changed?
            HasWarnings = (1 << 6),
            HasErrors = (1 << 7),
            DefaultBuffer = (1 << 8), // Default startup buffer
            HasTabs = (1 << 9),
            HasSpaceTabs = (1 << 10),
            InsertTabs = (1 << 11)
        };
    }

    // Ensure the character is >=0 and <=127 as in the ASCII standard,
    // isalnum, for example will assert on debug build if not in this range.
    inline int ToASCII(const char ch) {
        auto ret = (unsigned int)ch;
        ret = std::min(ret, 127u);
        return ret;
    }

    enum class BufferType {
        Normal,
        Search,
        Repl,
        DataGrid,
        Tree
    };

    // A really big cursor move; which will likely clamp
    // static const iterator MaxCursorMove = iterator(0xFFFFFFF);
    // const long InvalidByteIndex = -1;

    enum class ExpressionType {
        Inner,
        Outer
    };

    // The type of replacement that happens in the buffer
    enum class ReplaceRangeMode {
        Fill,
        Replace,
    };

    struct FoldRange {
        ByteRange range;
        long startLine = 0;
        long endLine = 0;
        std::string kind;
        bool collapsed = false;
    };

    struct ChangeRecord {
        std::string strDeleted;
        std::string strInserted;
        GlyphIterator itrStart;
        GlyphIterator itrEnd;

        void Clear() {
            strDeleted.clear();
            itrStart.Invalidate();
            itrEnd.Invalidate();
        }
    };

    using fnKeyNotifier = std::function<bool(uint32_t key, uint32_t modifier)>;
    class ZepBuffer : public ZepComponent {
    public:
        ZepBuffer(ZepEditor& editor, const std::string& strName);
        ZepBuffer(ZepEditor& editor, const fs::path& path);
        virtual ~ZepBuffer();

        void Clear();
        void SetText(const std::string& strText, bool initFromFile = false);
        void Load(const fs::path& path);
        bool Save(int64_t& size);

        fs::path GetFilePath() const;
        std::string GetFileExtension() const;
        void SetFilePath(const fs::path& path);

        GlyphIterator GetLinePos(GlyphIterator bufferLocation, LineLocation lineLocation) const;
        bool GetLineOffsets(const long line, ByteRange& range) const;
        GlyphIterator ClampToVisibleLine(GlyphIterator in) const;
        long GetBufferColumn(GlyphIterator location) const;
        using fnMatch = std::function<bool(const char)>;

        bool Move(GlyphIterator& loc, Direction dir) const;
        void MotionBegin(GlyphIterator& start) const;
        bool Skip(fnMatch IsToken, GlyphIterator& start, Direction dir) const;
        bool SkipOne(fnMatch IsToken, GlyphIterator& start, Direction dir) const;
        bool SkipNot(fnMatch IsToken, GlyphIterator& start, Direction dir) const;

        GlyphIterator Find(GlyphIterator start, const uint8_t* pBegin, const uint8_t* pEnd) const;
        GlyphIterator FindFirstCharOf(GlyphIterator start, const std::string& chars, int32_t& foundIndex, Direction dir) const;
        GlyphIterator FindOnLineMotion(GlyphIterator start, const uint8_t* pCh, Direction dir) const;
        GlyphIterator FindMatchingParen(GlyphIterator bufferCursor);
        std::pair<GlyphIterator, GlyphIterator> FindMatchingPair(GlyphIterator start, const uint8_t ch) const;
        GlyphIterator WordMotion(GlyphIterator start, uint32_t searchType, Direction dir) const;
        GlyphIterator EndWordMotion(GlyphIterator start, uint32_t searchType, Direction dir) const;
        GlyphIterator ChangeWordMotion(GlyphIterator start, uint32_t searchType, Direction dir) const;
        GlyphRange AWordMotion(GlyphIterator start, uint32_t searchType) const;
        GlyphRange InnerWordMotion(GlyphIterator start, uint32_t searchType) const;
        GlyphRange StandardCtrlMotion(GlyphIterator cursor, Direction searchDir) const;

        // Things that change
        bool Delete(const GlyphIterator& startOffset, const GlyphIterator& endOffset, ChangeRecord& changeRecord);
        bool Insert(const GlyphIterator& startOffset, const std::string& str, ChangeRecord& changeRecord);
        bool Replace(const GlyphIterator& startOffset, const GlyphIterator& endOffset, /*note; not ref*/ std::string str, ReplaceRangeMode mode, ChangeRecord& changeRecord);

        long GetLineCount() const {
            return long(m_lineEnds.size());
        }
        long GetBufferLine(GlyphIterator offset) const;

        GlyphIterator End() const;
        GlyphIterator Begin() const;

        const GapBuffer<uint8_t>& GetWorkingBuffer() const {
            return m_workingBuffer;
        }

        GapBuffer<uint8_t>& GetMutableWorkingBuffer() {
            return m_workingBuffer;
        }

        const std::vector<ByteIndex> GetLineEnds() const {
            return m_lineEnds;
        }

        void SetToneColor(const NVec4f& toneColor) {
            m_toneColor = toneColor;
        }

        const NVec4f& GetToneColor() const {
            return m_toneColor;
        }

        void SetSyntax(std::shared_ptr<ZepSyntax> syntax) {
            m_spSyntax = syntax;
        }

        void SetSyntaxProvider(SyntaxProvider provider) {
            if (provider.syntaxID != m_syntaxProvider.syntaxID) {
                if (provider.factory) {
                    m_spSyntax = provider.factory(this);
                } else {
                    m_spSyntax.reset();
                }

                m_syntaxProvider = provider;
            }
        }

        ZepSyntax* GetSyntax() const {
            return m_spSyntax.get();
        }

        const std::string& GetName() const;

        std::string GetDisplayName() const;
        virtual void Notify(std::shared_ptr<ZepMessage> message) override;

        ZepTheme& GetTheme() const;
        void SetTheme(std::shared_ptr<ZepTheme> spTheme);

        void SetSelection(const GlyphRange& sel);
        GlyphRange GetInclusiveSelection() const;
        bool HasSelection() const;
        void ClearSelection();

        void AddRangeMarker(std::shared_ptr<RangeMarker> spMarker);
        void ClearRangeMarker(std::shared_ptr<RangeMarker> spMarker);
        void ClearRangeMarkers(const std::set<std::shared_ptr<RangeMarker>>& markers);
        void ClearRangeMarkers(uint32_t types);
        tRangeMarkers GetRangeMarkers(uint32_t types) const;
        tRangeMarkers GetRangeMarkersOnLine(uint32_t types, long line) const;
        void HideMarkers(uint32_t markerType);
        void ShowMarkers(uint32_t markerType, uint32_t displayType);

        void SetFoldRanges(std::vector<FoldRange> ranges);
        void ClearFoldRanges();
        bool ToggleFoldAtByte(ByteIndex byteIndex);
        bool SetFoldCollapsedAtByte(ByteIndex byteIndex, bool collapsed);
        bool SetAllFoldsCollapsed(bool collapsed);
        const std::vector<FoldRange>& GetFoldRanges() const;
        const FoldRange* GetFoldStartingOnLine(long line) const;
        const FoldRange* GetCollapsedFoldContainingLine(long line) const;
        const FoldRange* GetCollapsedFoldContainingByte(ByteIndex byteIndex) const;
        bool IsLineHiddenByFold(long line) const;
        bool IsByteHiddenByFold(ByteIndex byteIndex) const;
        ByteIndex GetVisibleByteForFoldedByte(ByteIndex byteIndex) const;

        void ForEachMarker(uint32_t types, Direction dir, const GlyphIterator& begin, const GlyphIterator& end, std::function<bool(const std::shared_ptr<RangeMarker>&)> fnCB) const;
        std::shared_ptr<RangeMarker> FindNextMarker(GlyphIterator start, Direction dir, uint32_t markerType);

        void SetBufferType(BufferType type);
        BufferType GetBufferType() const;

        void SetLastEditLocation(GlyphIterator loc);
        GlyphIterator GetLastEditLocation();

        ZepMode* GetMode() const;
        void SetMode(std::shared_ptr<ZepMode> spMode);

        uint64_t GetLastUpdateTime() const {
            return m_lastUpdateTime;
        }

        uint64_t GetUpdateCount() const {
            return m_updateCount;
        }

        bool IsHidden() const;

        bool HasFileFlags(uint32_t flags) const;
        void SetFileFlags(uint32_t flags, bool set = true);
        void ClearFileFlags(uint32_t flags);
        void ToggleFileFlag(uint32_t flags);

        GlyphRange GetExpression(ExpressionType type, const GlyphIterator location, const std::vector<char>& beginExpression, const std::vector<char>& endExpression) const;
        std::string GetBufferText(const GlyphIterator& start, const GlyphIterator& end) const;

        void SetPostKeyNotifier(fnKeyNotifier notifier);
        fnKeyNotifier GetPostKeyNotifier() const;

        void EndFlash() const;
        void BeginFlash(float seconds, FlashType flashType, const GlyphRange& range);

        uint64_t ToHandle() const;
        static ZepBuffer* FromHandle(ZepEditor& editor, uint64_t handle);

        Zep::signal<void(ZepBuffer& buffer, const GlyphIterator&, const std::string&)> sigPreInsert;
        Zep::signal<void(ZepBuffer& buffer, const GlyphIterator&, const GlyphIterator&)> sigPreDelete;

        std::stack<std::shared_ptr<ZepCommand>>& GetUndoStack();
        std::stack<std::shared_ptr<ZepCommand>>& GetRedoStack();

    private:
        void MarkUpdate();

    private:
        // Buffer & record of the line end locations
        GapBuffer<uint8_t> m_workingBuffer;

        std::vector<ByteIndex> m_lineEnds;

        // File and modification info
        fs::path m_filePath;
        mutable std::string m_strName;
        uint32_t m_fileFlags = 0;
        BufferType m_bufferType = BufferType::Normal;
        GlyphIterator m_lastEditLocation; // = 0;
        uint64_t m_updateCount = 0;
        uint64_t m_lastUpdateTime = 0;

        // Syntax and theme
        std::shared_ptr<ZepSyntax> m_spSyntax;
        std::shared_ptr<ZepTheme> m_spOverrideTheme;
        SyntaxProvider m_syntaxProvider;
        NVec4f m_toneColor = NVec4f(0.0f);

        // Selections
        GlyphRange m_selection;
        tRangeMarkers m_rangeMarkers;
        std::vector<FoldRange> m_foldRanges;
        long m_collapsedFoldCount = 0;

        // Modes
        std::shared_ptr<ZepMode> m_spMode;
        fnKeyNotifier m_postKeyNotifier;

        std::stack<std::shared_ptr<ZepCommand>> m_undoStack;
        std::stack<std::shared_ptr<ZepCommand>> m_redoStack;
    };

    // Notification payload
    enum class BufferMessageType {
        // Inform clients that we are about to mess with the buffer
        PreBufferChange = 0,
        TextChanged,
        TextDeleted,
        TextAdded,
        Loaded,
        MarkersChanged,
        FoldsChanged
    };

    struct BufferMessage : public ZepMessage {
        BufferMessage(ZepBuffer* pBuff, BufferMessageType messageType, const GlyphIterator& startLoc, const GlyphIterator& endLoc)
            : ZepMessage(Msg::Buffer),
              pBuffer(pBuff),
              type(messageType),
              startLocation(startLoc),
              endLocation(endLoc) {
        }

        ZepBuffer* pBuffer;
        BufferMessageType type;
        GlyphIterator startLocation;
        GlyphIterator endLocation;
    };

} // namespace Zep
