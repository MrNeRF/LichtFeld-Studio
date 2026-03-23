#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <regex>

#include "zep/buffer.h"
#include "zep/editor.h"
#include "zep/filesystem.h"

#include "zep/mcommon/string/stringutils.h"

#include "zep/mcommon/logger.h"

namespace Zep {

    namespace {

        // A VIM-like definition of a word.  Actually, in Vim this can be changed, but this editor
        // assumes a word is alphanumeric or underscore for consistency
        inline bool IsWordChar(const char c) {
            auto ch = ToASCII(c);
            return std::isalnum(ch) || ch == '_';
        }
        inline bool IsWordOrSepChar(const char c) {
            auto ch = ToASCII(c);
            return std::isalnum(ch) || ch == '_' || ch == ' ' || ch == '\n' || ch == 0;
        }
        inline bool IsWORDChar(const char c) {
            auto ch = ToASCII(c);
            return std::isgraph(ch);
        }
        inline bool IsWORDOrSepChar(const char c) {
            auto ch = ToASCII(c);
            return std::isgraph(ch) || ch == ' ' || ch == '\n' || ch == 0;
        }
        inline bool IsSpace(const char c) {
            auto ch = ToASCII(c);
            return ch == ' ';
        }
        inline bool IsSpaceOrNewline(const char c) {
            auto ch = ToASCII(c);
            return ch == ' ' || ch == '\n';
        }
        inline bool IsSpaceOrTerminal(const char c) {
            auto ch = ToASCII(c);
            return ch == ' ' || ch == 0 || ch == '\n';
        }

        using fnMatch = std::function<bool>(const char);

    } // namespace
    ZepBuffer::ZepBuffer(ZepEditor& editor, const std::string& strName)
        : ZepComponent(editor), m_strName(strName) {
        Clear();
    }

    ZepBuffer::ZepBuffer(ZepEditor& editor, const fs::path& path)
        : ZepComponent(editor) {
        Load(path);
    }

    ZepBuffer::~ZepBuffer() {
    }

    void ZepBuffer::Notify(std::shared_ptr<ZepMessage> message) {
    }

    // Vertical column
    long ZepBuffer::GetBufferColumn(GlyphIterator location) const {
        auto lineStart = GetLinePos(location, LineLocation::LineBegin);
        return CodePointDistance(lineStart, location);
    }

    // Find the location inside the list of line ends
    long ZepBuffer::GetBufferLine(GlyphIterator location) const {
        auto itrLine = std::lower_bound(m_lineEnds.begin(), m_lineEnds.end(), location.Index());
        if (itrLine != m_lineEnds.end() && location.Index() >= *itrLine) {
            itrLine++;
        }
        long line = long(itrLine - m_lineEnds.begin());
        line = std::min(std::max(0l, line), long(m_lineEnds.size() - 1));
        return line;
    }

    // Prepare for a motion
    void ZepBuffer::MotionBegin(GlyphIterator& start) const {
        GlyphIterator newStart = start;

        // Clamp to sensible, begin
        newStart.Clamp();

        bool change = newStart != start;
        if (change) {
            start = newStart;
        }
    }

    bool ZepBuffer::Move(GlyphIterator& loc, Direction dir) const {
        if (dir == Direction::Backward && loc.Index() == 0) {
            return false;
        } else if (dir == Direction::Forward && loc.Index() == End().Index()) {
            return false;
        }

        if (dir == Direction::Backward) {
            loc--;
        } else {
            loc++;
        }
        return true;
    }

    bool ZepBuffer::Skip(fnMatch IsToken, GlyphIterator& start, Direction dir) const {
        if (!start.Valid()) {
            return false;
        }

        bool found = false;
        auto itrEnd = End();
        auto itrBegin = Begin();

        while (IsToken(start.Char())) {
            found = true;
            if (!Move(start, dir))
                break;
        }
        return found;
    }

    bool ZepBuffer::SkipOne(fnMatch IsToken, GlyphIterator& start, Direction dir) const {
        if (!start.Valid()) {
            return false;
        }

        bool found = false;
        if (IsToken(start.Char())) {
            found = true;
            Move(start, dir);
        }
        return found;
    }

    bool ZepBuffer::SkipNot(fnMatch IsToken, GlyphIterator& start, Direction dir) const {
        if (!start.Valid())
            return false;

        bool found = false;
        while (!IsToken(start.Char())) {
            found = true;
            if (!Move(start, dir))
                break;
        }
        return found;
    }

    // This is the vim-like 'caw' rule; The motions and behaviour are based on how vim behaves.
    // This is still quite complex behavior for this particular motion.  I'm open to better ways to express it!
    GlyphRange ZepBuffer::AWordMotion(GlyphIterator start, uint32_t searchType) const {
        auto IsWord = searchType == SearchType::Word ? IsWordChar : IsWORDChar;

        GlyphRange r;
        r.first = start;

        MotionBegin(start);

        // Already on a word; find the limits, and include the space
        if (Skip(IsWord, start, Direction::Backward)) {
            SkipNot(IsWord, start, Direction::Forward);
            r.first = start;
            Skip(IsWord, start, Direction::Forward);
            Skip(IsSpace, start, Direction::Forward);
            r.second = start;
        }
        // ... or skip space
        else if (Skip(IsSpace, start, Direction::Forward)) {
            Skip(IsWord, start, Direction::Forward);
            r.second = start;
        }
        // On a non-word, find the beginning, remove including following spaces
        else if (SkipNot(IsWord, start, Direction::Backward)) {
            Skip(IsSpace, start, Direction::Forward);
            Skip(IsWord, start, Direction::Forward);
            r.first = start;
            SkipNot(IsWord, start, Direction::Forward);
            Skip(IsSpace, start, Direction::Forward);
            r.second = start;
        }

        return r;
    }

    // Implements the ctrl + motion of a standard editor.
    // This is a little convoluted; perhaps the logic can be simpler!
    // Playing around with CTRL+ arrows and shift in an app like notepad will teach you that the rules for how far to jump
    // depend on what you are over, and which direction you are going.....
    // The unit tests are designed to enforce the behavior here
    GlyphRange ZepBuffer::StandardCtrlMotion(GlyphIterator cursor, Direction searchDir) const {
        MotionBegin(cursor);

        auto lineEnd = GetLinePos(cursor, LineLocation::LineCRBegin);
        auto current = std::min(lineEnd, cursor.Clamp());

        GlyphRange r;
        r.first = current;
        r.second = current;

        if (searchDir == Direction::Forward) {
            if (Skip(IsWORDChar, current, searchDir)) {
                // Skip space
                Skip(IsSpace, current, searchDir);
            } else {
                SkipNot(IsWORDChar, current, searchDir);
            }
        } else {
            // Always skip back a char (iterator will clamp)
            current--;

            // Stop on the newline, or continue
            if (current.Char() != '\n') {
                SkipNot(IsWORDChar, current, searchDir);
                Skip(IsWORDChar, current, searchDir);
                SkipNot(IsWORDChar, current, Direction::Forward);
            }
        }
        r.second = current.Clamp();

        return r;
    }

    // Note:
    // In general I'm unhappy with these word motion functions.  They are _hard_ ; especially if you
    // want them to conform to the quirks of Vim.  I haven't found a cleaner way than this to make them
    // work.
    GlyphRange ZepBuffer::InnerWordMotion(GlyphIterator start, uint32_t searchType) const {
        auto IsWordOrSpace = searchType == SearchType::Word ? IsWordOrSepChar : IsWORDOrSepChar;
        auto IsWord = searchType == SearchType::Word ? IsWordChar : IsWORDChar;
        MotionBegin(start);

        GlyphRange r;

        // Special case; change inner word on a newline, stay put, don't delete anything
        if (start.Char() == '\n') {
            r.first = r.second = start;
        } else if (SkipNot(IsWordOrSpace, start, Direction::Forward)) {
            r.second = start;
            start--;
            SkipNot(IsWordOrSpace, start, Direction::Backward);
            Skip(IsWordOrSpace, start, Direction::Forward);
            r.first = start;
        } else if (Skip(IsSpace, start, Direction::Forward)) {
            r.second = start;
            start--;
            Skip(IsSpace, start, Direction::Backward);
            SkipNot(IsSpace, start, Direction::Forward);
            r.first = start;
        } else {
            Skip(IsWord, start, Direction::Forward);
            r.second = start;
            start--;
            Skip(IsWord, start, Direction::Backward);
            SkipNot(IsWord, start, Direction::Forward);
            r.first = start;
        }
        return r;
    }

    GlyphIterator ZepBuffer::Find(GlyphIterator start, const uint8_t* pBeginString, const uint8_t* pEndString) const {
        // Should be a valid start
        assert(start.Valid());
        assert(pBeginString != nullptr);

        if (!start.Valid() && pBeginString) {
            return start;
        }

        // Find the end of the test string
        if (pEndString == nullptr) {
            pEndString = pBeginString;
            while (*pEndString != 0) {
                pEndString++;
            }
        }

        auto itrBuffer = start;
        auto itrEnd = End();
        while (itrBuffer != itrEnd) {
            auto itrNext = itrBuffer;

            // Loop the string and match it to the buffer
            auto pCurrent = pBeginString;
            while (pCurrent != pEndString && itrNext != itrEnd) {
                if (*pCurrent != itrNext.Char()) {
                    break;
                }
                pCurrent++;
                itrNext++;
            };

            // We successfully got to the end
            if (pCurrent == pEndString) {
                return itrBuffer;
            }

            itrBuffer++;
        };

        return GlyphIterator();
    }

    GlyphIterator ZepBuffer::FindOnLineMotion(GlyphIterator start, const uint8_t* pCh, Direction dir) const {
        auto entry = start;
        auto IsMatch = [pCh](const char ch) {
            if (*pCh == ch)
                return true;
            return false;
        };
        auto NotMatchNotEnd = [pCh](const char ch) {
            if (*pCh != ch && ch != '\n')
                return true;
            return false;
        };

        if (dir == Direction::Forward) {
            // Ignore char under cursor, as behavior
            SkipOne(IsMatch, start, dir);

            // Find to the end of the line
            Skip(NotMatchNotEnd, start, dir);
        } else {
            SkipOne(IsMatch, start, dir);
            Skip(NotMatchNotEnd, start, dir);
        }

        if (start.Valid() && *pCh == start.Char()) {
            return start;
        }
        return entry;
    }

    std::pair<GlyphIterator, GlyphIterator> ZepBuffer::FindMatchingPair(GlyphIterator start, const uint8_t ch) const {
        std::string delims;
        switch (ch) {
        case '(':
        case ')':
            delims = "()";
            break;
        case '[':
        case ']':
            delims = "[]";
            break;
        case '{':
        case '}':
            delims = "{}";
            break;
        default:
            // Matching same char at both ends
            delims = std::string(2, ch);
            break;
        }

        std::pair<GlyphIterator, GlyphIterator> ret;
        Direction dir = Direction::Backward;

        auto search = [&](GlyphIterator loc, Direction dir) {
            int openCount = 1;
            for (;;) {
                // Find the previous open for the current delim type
                int newIndex;
                loc = FindFirstCharOf(loc, delims, newIndex, dir);

                // Fell off beginning, no find
                if (newIndex < 0) {
                    return GlyphIterator();
                }

                if (dir == Direction::Forward) {
                    newIndex = 1 - newIndex;
                }

                // Match immediately for ""
                if (delims[0] == delims[1]) {
                    newIndex = 0;
                }

                // Found another opener
                if (newIndex == 0) {
                    openCount--;
                    if (openCount == 0) {
                        // Found opening
                        return loc;
                    }
                }
                // Found a closer
                else if (newIndex == 1) {
                    openCount++;
                }

                if (dir == Direction::Forward) {
                    if (loc == End()) {
                        return GlyphIterator();
                    }
                    loc++;
                } else {
                    if (loc == Begin()) {
                        return GlyphIterator();
                    }
                    loc--;
                }
            }
        };

        // If on the end bracket, start before it.
        if (start.Char() == delims[1] && delims[0] != delims[1]) {
            start--;
        }

        // Search for the begin
        ret.first = search(start, Direction::Backward);
        if (ret.first.Valid() && ret.first != End()) {
            // Search for the end
            ret.second = search(ret.first + 1, Direction::Forward);
        }
        return ret;
    }

    // Only works on searches of ascii characters (but navigates unicode buffer); useful for some vim operations
    // Returns the index of the first found char and its location
    GlyphIterator ZepBuffer::FindFirstCharOf(GlyphIterator start, const std::string& chars, int32_t& found_index, Direction dir) const {
        GlyphIterator itr = start;
        if (!itr.Valid()) {
            found_index = -1;
            return itr;
        }

        for (;;) {
            for (int i = 0; i < chars.length(); i++) {
                if (itr.Char() == chars[i]) {
                    found_index = i;
                    return itr;
                }
            }

            if (dir == Direction::Forward) {
                if (itr == End()) {
                    break;
                }
                itr++;
            } else if (dir == Direction::Backward) {
                if (itr == Begin()) {
                    break;
                }
                itr--;
            }
        }
        found_index = -1;
        return itr;
    }

    GlyphIterator ZepBuffer::WordMotion(GlyphIterator start, uint32_t searchType, Direction dir) const {
        auto IsWord = searchType == SearchType::Word ? IsWordChar : IsWORDChar;

        MotionBegin(start);

        if (dir == Direction::Forward) {
            if (Skip(IsWord, start, dir)) {
                // Skipped a word, skip spaces then done
                Skip(IsSpaceOrTerminal, start, dir);
            } else {
                SkipNot(IsWord, start, dir);
            }
        } else // Backward
        {
            auto startSearch = start;

            // Jump back to the beginning of a word if on it
            if (Skip(IsWord, start, dir)) {
                // If we weren't already on the first char of the word, then we have gone back a word!
                if (startSearch != (start + 1)) {
                    SkipNot(IsWord, start, Direction::Forward);
                    return start;
                }
            } else {
                SkipNot(IsWord, start, dir);
            }

            // Skip any spaces
            Skip(IsSpace, start, dir);

            // Go back to the beginning of the word
            if (Skip(IsWord, start, dir)) {
                SkipNot(IsWord, start, Direction::Forward);
            }
        }
        return start;
    }

    GlyphIterator ZepBuffer::EndWordMotion(GlyphIterator start, uint32_t searchType, Direction dir) const {
        auto IsWord = searchType == SearchType::Word ? IsWordChar : IsWORDChar;

        MotionBegin(start);

        if (dir == Direction::Forward) {
            auto startSearch = start;

            // Skip to the end
            if (Skip(IsWord, start, dir)) {
                // We moved a bit, so we found the end of the current word
                if (startSearch != start - 1) {
                    SkipNot(IsWord, start, Direction::Backward);
                    return start;
                }
            } else {
                SkipNot(IsWord, start, dir);
            }

            // Skip any spaces
            Skip(IsSpaceOrNewline, start, dir);

            // Go back to the beginning of the word
            if (Skip(IsWord, start, dir)) {
                SkipNot(IsWord, start, Direction::Backward);
            }
        } else // Backward
        {
            // Note this is the same as the Next word code, in 'forward' mode
            if (Skip(IsWord, start, dir)) {
                // Skipped a word, skip spaces then done
                Skip(IsSpace, start, dir);
            } else {
                SkipNot(IsWord, start, dir);
            }
        }
        return start;
    }

    GlyphIterator ZepBuffer::ChangeWordMotion(GlyphIterator start, uint32_t searchType, Direction dir) const {
        // Change word is different to work skipping; it will change a string of spaces, for example.
        // Essentially it changes 'what you are over', based on the word rule
        auto IsWord = searchType == SearchType::Word ? IsWordChar : IsWORDChar;
        MotionBegin(start);
        if (Skip(IsWord, start, dir)) {
            return start;
        }
        SkipNot(IsWord, start, dir);
        return start;
    }

    GlyphIterator ZepBuffer::ClampToVisibleLine(GlyphIterator in) const {
        in = in.Clamp();
        auto loc = GetLinePos(in, LineLocation::LineLastNonCR);
        in = std::min(loc, in);
        return in;
    }

    // Method for querying the beginning and end of a line
    bool ZepBuffer::GetLineOffsets(const long line, ByteRange& range) const {
        // Not valid
        if ((long)m_lineEnds.size() <= line) {
            range.first = 0;
            range.second = 0;
            return false;
        }

        // Find the line bounds - we know the end, find the start from the previous
        range.second = m_lineEnds[line];
        range.first = line == 0 ? 0 : m_lineEnds[line - 1];
        return true;
    }

    std::string ZepBuffer::GetFileExtension() const {
        std::string ext;
        if (GetFilePath().has_filename() && GetFilePath().filename().has_extension()) {
            ext = string_tolower(GetFilePath().filename().extension().string());
        } else {
            auto str = GetName();
            size_t dot_pos = str.find_last_of(".");
            if (dot_pos != std::string::npos) {
                ext = string_tolower(str.substr(dot_pos, str.length() - dot_pos));
            }
        }
        return ext;
    }

    // Basic load support; read a file if it's present, but keep
    // the file path in case you want to write later
    void ZepBuffer::Load(const fs::path& path) {
        // Set the name from the path
        if (path.has_filename()) {
            m_strName = path.filename().string();
        } else {
            m_strName = m_filePath.string();
        }

        // Must set the syntax before the first buffer change messages
        // TODO: I believe that some of this buffer config should move to Editor.cpp
        GetEditor().SetBufferSyntax(*this);

        if (GetEditor().GetFileSystem().Exists(path)) {
            m_filePath = GetEditor().GetFileSystem().Canonical(path);
            auto read = GetEditor().GetFileSystem().Read(path);

            // Always set text, to ensure we prepare the buffer with 0 terminator,
            // even if string is empty
            SetText(read, true);
        } else {
            // Can't canonicalize a non-existent path.
            // But we may have a path we haven't save to yet!
            Clear();
            m_filePath = path;
        }
    }

    bool ZepBuffer::Save(int64_t& size) {
        if (ZTestFlags(m_fileFlags, FileFlags::Locked)) {
            return false;
        }

        if (ZTestFlags(m_fileFlags, FileFlags::ReadOnly)) {
            return false;
        }

        auto str = GetWorkingBuffer().string();

        // Put back /r/n if necessary while writing the file
        // At the moment, Zep removes /r/n and just uses /n while modifying text.
        // It replaces the /r on files that had it afterwards
        // Alternatively we could manage them 'in place', but that would make parsing more complex.
        // And then what do you do if there are 2 different styles in the file.
        if (m_fileFlags & FileFlags::StrippedCR) {
            // TODO: faster way to replace newlines
            string_replace_in_place(str, "\n", "\r\n");
        }

        // Remove the appended 0 if necessary
        size = (int64_t)str.size();
        if (m_fileFlags & FileFlags::TerminatedWithZero) {
            size--;
        }

        // Ensure size is at least 0
        size = std::max(size, int64_t(0));

        if (GetEditor().GetFileSystem().Write(m_filePath, &str[0], (size_t)size)) {
            m_fileFlags = ZClearFlags(m_fileFlags, FileFlags::Dirty);

            if (GetEditor().GetFileSystem().Exists(m_filePath)) {
                // We wrote succesfully, so make sure our path is canonical
                m_filePath = GetEditor().GetFileSystem().Canonical(m_filePath);
            }
            return true;
        }
        return false;
    }

    std::string ZepBuffer::GetDisplayName() const {
        if (m_filePath.empty()) {
            return m_strName;
        }
        return m_filePath.string();
    }

    fs::path ZepBuffer::GetFilePath() const {
        return m_filePath;
    }

    void ZepBuffer::SetFilePath(const fs::path& path) {
        auto testPath = path;
        if (GetEditor().GetFileSystem().Exists(testPath)) {
            testPath = GetEditor().GetFileSystem().Canonical(testPath);
        } else {
            if (!path.is_absolute()) {
                auto full = GetEditor().GetFileSystem().GetWorkingDirectory() / path;
                if (GetEditor().GetFileSystem().Exists(full)) {
                    testPath = GetEditor().GetFileSystem().Canonical(full);
                } else {
                    testPath = full;
                }
            }
        }

        if (!GetEditor().GetFileSystem().Equivalent(testPath, m_filePath)) {
            m_filePath = testPath;
        }
        GetEditor().SetBufferSyntax(*this);
    }

    // Remember that we updated the buffer and dirty the state
    // Clients can use these values to figure out update times and dirty state
    void ZepBuffer::MarkUpdate() {
        m_updateCount++;
        m_lastUpdateTime = timer_get_time_now();

        m_fileFlags = ZSetFlags(m_fileFlags, FileFlags::Dirty);
    }

    // Clear this buffer.  If it was previously not clear, it has been updated.
    // Otherwise it is just reset to default state.  A new buffer is always initially cleared.
    void ZepBuffer::Clear() {
        // A buffer that is empty is brand new; just make it 0 chars and return
        if (m_workingBuffer.size() <= 1) {
            m_workingBuffer.clear();
            m_workingBuffer.push_back(0);
            m_lineEnds.clear();
            m_fileFlags = ZSetFlags(m_fileFlags, FileFlags::TerminatedWithZero);
            m_lineEnds.push_back(End().Index() + 1);
            return;
        }

        // Inform clients we are about to change the buffer
        GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::PreBufferChange, GlyphIterator(this), End()));

        m_workingBuffer.clear();
        m_workingBuffer.push_back(0);
        m_lineEnds.clear();
        m_fileFlags = ZSetFlags(m_fileFlags, FileFlags::TerminatedWithZero);
        m_lineEnds.push_back(End().Index() + 1);

        {
            MarkUpdate();
            GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::TextDeleted, GlyphIterator(this), End()));
        }
    }

    // Replace the buffer with the text
    void ZepBuffer::SetText(const std::string& text, bool initFromFile) {
        // First, clear it
        Clear();

        bool lastWasSpace = false;
        if (!text.empty()) {
            // Since incremental insertion of a big file into a gap buffer gives us worst case performance,
            // We build the buffer in a separate array and assign it.  Much faster.
            std::vector<uint8_t> input;

            m_lineEnds.clear();

            // Update the gap buffer with the text
            // We remove \r, we only care about \n
            for (auto& ch : text) {
                if (ch == '\r') {
                    m_fileFlags |= FileFlags::StrippedCR;
                } else {
                    input.push_back(ch);
                    if (ch == '\n') {
                        m_lineEnds.push_back(ByteIndex(input.size()));
                        lastWasSpace = false;
                    } else if (ch == '\t') {
                        m_fileFlags |= FileFlags::HasTabs;
                        lastWasSpace = false;
                    } else if (ch == ' ') {
                        if (lastWasSpace) {
                            m_fileFlags |= FileFlags::HasSpaceTabs;
                        }
                        lastWasSpace = true;
                    } else {
                        lastWasSpace = false;
                    }
                }
            }
            m_workingBuffer.assign(input.begin(), input.end());
        }

        // If file is only tabs, then force tab mode
        if (HasFileFlags(FileFlags::HasTabs) && !HasFileFlags(FileFlags::HasSpaceTabs)) {
            m_fileFlags = ZSetFlags(m_fileFlags, FileFlags::InsertTabs);
        }

        if (m_workingBuffer[m_workingBuffer.size() - 1] != 0) {
            m_fileFlags |= FileFlags::TerminatedWithZero;
            m_workingBuffer.push_back(0);
        }

        // TODO: Why is a line end needed always?
        // TODO: Line ends 1 beyond, or just for end?  Can't remember this detail:
        // understand it, then write a unit test to ensure it.
        m_lineEnds.push_back(End().Index() + 1);

        MarkUpdate();

        // When loading a file, send the Loaded message to distinguish it from adding to a buffer, and remember that the buffer is not dirty in this case
        if (initFromFile) {
            GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::Loaded, Begin(), End()));

            // Doc is not dirty
            m_fileFlags = ZClearFlags(m_fileFlags, FileFlags::Dirty);
        } else {
            GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::TextAdded, Begin(), End()));
        }
    }

    // TODO: This can be cleaner
    // The function needs to find the point on the line which bufferLocation is on.
    // It needs to account for empty lines or the last line, zero terminated.
    // It shouldn't walk away to another line!
    GlyphIterator ZepBuffer::GetLinePos(GlyphIterator bufferLocation, LineLocation lineLocation) const {
        if (lineLocation == LineLocation::None) {
            assert(!"Invalid");
            return GlyphIterator();
        }

        bufferLocation.Clamp();
        if (m_workingBuffer.empty()) {
            return bufferLocation;
        }

        GlyphIterator itr = bufferLocation;
        GlyphIterator itrBegin = Begin();
        GlyphIterator itrEnd = End();

        GlyphIterator itrLineStart(itr);

        // If we are on the CR, move back 1, unless the \n is all that is on the line
        if (itrLineStart != itrBegin) {
            if (itrLineStart.Char() == '\n') {
                itrLineStart.Move(-1);
            }

            // Find the end of the previous line
            while (itrLineStart > itrBegin && itrLineStart.Char() != '\n') {
                itrLineStart.Move(-1);
            }

            if (itrLineStart.Char() == '\n')
                itrLineStart.Move(1);

            itr = itrLineStart;
        }

        switch (lineLocation) {
        default:
        // We are on the first bit of the line anyway
        case LineLocation::LineBegin: {
            return itr.Clamped();
        } break;

        // The point just after the line end
        case LineLocation::BeyondLineEnd: {
            while (itr < itrEnd && itr.Char() != '\n' && itr.Char() != 0) {
                itr.Move(1);
            }
            itr.Move(1);
            return itr.Clamped();
        } break;

        case LineLocation::LineCRBegin: {
            while (itr < itrEnd && itr.Char() != '\n' && itr.Char() != 0) {
                itr.Move(1);
            }
            return itr;
        } break;

        case LineLocation::LineFirstGraphChar: {
            while (itr < itrEnd && !std::isgraph(ToASCII(itr.Char())) && itr.Char() != '\n') {
                itr.Move(1);
            }
            return itr.Clamped();
        } break;

        case LineLocation::LineLastNonCR: {
            auto itrFirst = itr;

            while (itr < itrEnd && itr.Char() != '\n' && itr.Char() != 0) {
                itr.Move(1);
            }

            if (itrFirst != itr) {
                itr.Move(-1);
            }

            return itr.Clamped();
        } break;

        case LineLocation::LineLastGraphChar: {
            while (itr < itrEnd && itr.Char() != '\n' && itr.Char() != 0) {
                itr.Move(1);
            }

            while (itr > itrBegin && itr < itrEnd && !std::isgraph(ToASCII(itr.Char()))) {
                itr.Move(-1);
            }

            if (itr < itrLineStart) {
                itr = itrLineStart;
            }

            return itr.Clamped();
        } break;
        }
    }

    std::string ZepBuffer::GetBufferText(const GlyphIterator& start, const GlyphIterator& end) const {
        return std::string(m_workingBuffer.begin() + start.Index(), m_workingBuffer.begin() + end.Index());
    }

    bool ZepBuffer::Insert(const GlyphIterator& startIndex, const std::string& str, ChangeRecord& changeRecord) {
        if (!startIndex.Valid()) {
            return false;
        }

        GlyphIterator endIndex(this, startIndex.Index() + long(str.length()));

        sigPreInsert(*this, startIndex, str);

        // We are about to modify this range
        // TODO: Is this correct, and/or useful in any way??
        // We aren't changing this range at all; we are shifting those characters forward and replacing the area
        GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::PreBufferChange, startIndex, endIndex));

        // abcdef\r\nabc<insert>dfdf\r\n
        auto itrLine = std::lower_bound(m_lineEnds.begin(), m_lineEnds.end(), startIndex.Index());
        ;
        if (itrLine != m_lineEnds.end() && *itrLine <= startIndex.Index()) {
            itrLine++;
        }

        auto itrEnd = str.end();
        auto itrBegin = str.begin();
        auto itr = str.begin();

        // Make a list of lines to 'insert'
        std::vector<long> lines;
        std::string lineEndSymbols("\n");
        while (itr != itrEnd) {
            // Get to first point after "\n"
            // That's the point just after the end of the current line
            itr = std::find_first_of(itr, itrEnd, lineEndSymbols.begin(), lineEndSymbols.end());
            if (itr != itrEnd) {
                if (itr != itrEnd && *itr == '\n') {
                    itr++;
                }
                lines.push_back(long(itr - itrBegin) + startIndex.Index());
            }
        }

        // Increment the rest of the line ends
        // We make all the remaning line ends bigger by the fixed_size of the insertion
        auto itrAdd = itrLine;
        while (itrAdd != m_lineEnds.end()) {
            *itrAdd += long(str.length());
            itrAdd++;
        }

        if (!lines.empty()) {
            // Update the atomic line counter so clients can see where we are up to.
            m_lineEnds.insert(itrLine, lines.begin(), lines.end());
        }

        changeRecord.strInserted = str;
        m_workingBuffer.insert(m_workingBuffer.begin() + startIndex.Index(), str.begin(), str.end());

        MarkUpdate();

        // This is the range we added (not valid any more in the buffer)
        GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::TextAdded, startIndex, endIndex));

        return true;
    }

    bool ZepBuffer::Replace(const GlyphIterator& startIndex, const GlyphIterator& endIndex, std::string str, ReplaceRangeMode mode, ChangeRecord& changeRecord) {
        if (!startIndex.Valid() || !endIndex.Valid()) {
            return false;
        }

        if (mode == ReplaceRangeMode::Replace) {
            // A replace is really 2 steps; remove the current, insert the new
            Delete(startIndex, endIndex, changeRecord);

            ChangeRecord tempRecord;
            Insert(startIndex, str, tempRecord);
            return true;
        }

        // This is what we effectively delete when we do the replace
        changeRecord.strDeleted = GetBufferText(startIndex, endIndex);

        // We are about to modify this range
        GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::PreBufferChange, startIndex, endIndex));

        // Perform a fill
        for (auto loc = startIndex; loc < endIndex; loc++) {
            // Note we don't support utf8 yet
            // TODO: (0) Broken now we support utf8
            m_workingBuffer[loc.Index()] = str[0];
        }

        MarkUpdate();

        // This is the range we added (not valid any more in the buffer)
        GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::TextChanged, startIndex, endIndex));

        return true;
    }
    // A fundamental operation - delete a range of characters
    // Need to update:
    // - m_lineEnds
    // - m_processedLine
    // - m_pBuffer (i.e remove chars)
    // We also need to inform clients before we change the buffer, and after we delete text with the range we removed.
    // This helps them to fix up their data structures without rebuilding.
    // Assumption: The buffer always is at least a single line/character of '0', representing file end.
    // This makes a few things fall out more easily
    bool ZepBuffer::Delete(const GlyphIterator& startIndex, const GlyphIterator& endIndex, ChangeRecord& changeRecord) {
        assert(startIndex.Valid());
        assert(endIndex.Valid());

        // We are about to modify this range
        GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::PreBufferChange, startIndex, endIndex));

        changeRecord.strDeleted = GetBufferText(startIndex, endIndex);

        sigPreDelete(*this, startIndex, endIndex);

        auto itrLine = std::lower_bound(m_lineEnds.begin(), m_lineEnds.end(), startIndex.Index());
        if (itrLine == m_lineEnds.end()) {
            return false;
        }

        auto itrLastLine = std::upper_bound(itrLine, m_lineEnds.end(), endIndex.Index());
        auto offsetDiff = endIndex.Index() - startIndex.Index();

        if (*itrLine <= startIndex.Index()) {
            itrLine++;
        }

        // Adjust all line offsets beyond us
        for (auto itr = itrLastLine; itr != m_lineEnds.end(); itr++) {
            *itr -= offsetDiff;
        }

        if (itrLine != itrLastLine) {
            m_lineEnds.erase(itrLine, itrLastLine);
        }

        m_workingBuffer.erase(m_workingBuffer.begin() + startIndex.Index(), m_workingBuffer.begin() + endIndex.Index());
        assert(m_workingBuffer.size() > 0 && m_workingBuffer[m_workingBuffer.size() - 1] == 0);

        MarkUpdate();

        // This is the range we deleted (not valid any more in the buffer)
        GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::TextDeleted, startIndex, endIndex));

        return true;
    }

    ZepTheme& ZepBuffer::GetTheme() const {
        if (m_spOverrideTheme) {
            return *m_spOverrideTheme;
        }
        return GetEditor().GetTheme();
    }

    void ZepBuffer::SetTheme(std::shared_ptr<ZepTheme> spTheme) {
        m_spOverrideTheme = spTheme;
    }

    bool ZepBuffer::HasSelection() const {
        return m_selection.first.Valid() && m_selection.second.Valid();
    }

    void ZepBuffer::ClearSelection() {
        m_selection.first = GlyphIterator();
        m_selection.second = GlyphIterator();
    }

    GlyphRange ZepBuffer::GetInclusiveSelection() const {
        return m_selection;
    }

    void ZepBuffer::SetSelection(const GlyphRange& selection) {
        m_selection = selection;
        if (m_selection.first.Valid() && m_selection.second.Valid() &&
            (m_selection.first > m_selection.second)) {
            std::swap(m_selection.first, m_selection.second);
        }
    }

    void ZepBuffer::AddRangeMarker(std::shared_ptr<RangeMarker> spMarker) {
        m_rangeMarkers[spMarker->GetRange().first].insert(spMarker);

        // TODO: Why is this necessary; marks the whole buffer
        GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::MarkersChanged, Begin(), End()));
    }

    void ZepBuffer::ClearRangeMarker(std::shared_ptr<RangeMarker> spMarker) {
        auto itr = m_rangeMarkers.find(spMarker->GetRange().first);
        if (itr != m_rangeMarkers.end()) {
            itr->second.erase(spMarker);
            if (itr->second.empty()) {
                m_rangeMarkers.erase(spMarker->GetRange().first);
            }
        }

        // TODO: Why is this necessary; marks the whole buffer
        GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::MarkersChanged, Begin(), End()));
    }

    void ZepBuffer::ClearRangeMarkers(const std::set<std::shared_ptr<RangeMarker>>& markers) {
        for (auto& marker : markers) {
            ClearRangeMarker(marker);
        }
        GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::MarkersChanged, Begin(), End()));
    }

    void ZepBuffer::ClearRangeMarkers(uint32_t markerType) {
        std::set<std::shared_ptr<RangeMarker>> markers;
        ForEachMarker(markerType, Direction::Forward, Begin(), End(), [&](const std::shared_ptr<RangeMarker>& pMarker) {
            // Timed ones will expire on their own; TODO: Add a cleaner mechanism for selection here
            if (!(pMarker->displayType & RangeMarkerDisplayType::Timed)) {
                markers.insert(pMarker);
            }
            return true;
        });

        for (auto& victim : markers) {
            ClearRangeMarker(victim);
        }

        GetEditor().Broadcast(std::make_shared<BufferMessage>(this, BufferMessageType::MarkersChanged, Begin(), End()));
    }

    bool OverlapInclusive(ByteRange r1, ByteRange r2) {
        // -----aaaaa----
        // ---bbbbbbbbb-------
        if (r1.first <= r2.second && r2.first <= r1.second) {
            return true;
        }
        return false;
    }

    void ZepBuffer::ForEachMarker(uint32_t markerType, Direction dir, const GlyphIterator& begin, const GlyphIterator& end, std::function<bool(const std::shared_ptr<RangeMarker>&)> fnCB) const {
        ByteRange inclusive = ByteRange(begin.Index(), end.Peek(-1).Index());
        if (dir == Direction::Forward) {
            for (auto itr = m_rangeMarkers.begin(); itr != m_rangeMarkers.end(); itr++) {
                for (int pass = 0; pass < 2; pass++) {
                    for (auto& markerItem : itr->second) {
                        if ((markerItem->markerType & markerType) == 0) {
                            continue;
                        }

                        // Enumerate timed markers after all others, because these are effects that should happen last
                        if (pass == 0) {
                            if (markerItem->displayType & RangeMarkerDisplayType::Timed)
                                continue;
                        } else {
                            if (!(markerItem->displayType & RangeMarkerDisplayType::Timed))
                                continue;
                        }

                        ByteRange markerInclusive = ByteRange(markerItem->GetRange().first, std::max(0l, markerItem->GetRange().second - 1));

                        if (!OverlapInclusive(inclusive, markerInclusive)) {
                            continue;
                        }

                        if (!fnCB(markerItem)) {
                            return;
                        }
                    }
                }
            }
        } else {
            auto itrREnd = std::make_reverse_iterator(m_rangeMarkers.begin());
            auto itrRStart = std::make_reverse_iterator(m_rangeMarkers.end());

            for (auto itr = itrRStart; itr != itrREnd; itr++) {
                for (auto& markerItem : itr->second) {
                    if ((markerItem->markerType & markerType) == 0) {
                        continue;
                    }
                    if (!fnCB(markerItem)) {
                        return;
                    }
                }
            }
        }
    }

    void ZepBuffer::HideMarkers(uint32_t markerType) {
        ForEachMarker(markerType, Direction::Forward, Begin(), End(), [&](const std::shared_ptr<RangeMarker>& spMarker) {
            if ((spMarker->markerType & markerType) != 0) {
                spMarker->displayType = RangeMarkerDisplayType::Hidden;
            }
            return true;
        });
    }

    void ZepBuffer::ShowMarkers(uint32_t markerType, uint32_t displayType) {
        ForEachMarker(markerType, Direction::Forward, Begin(), End(), [&](const std::shared_ptr<RangeMarker>& spMarker) {
            if ((spMarker->markerType & markerType) != 0) {
                spMarker->displayType = displayType;
            }
            return true;
        });
    }

    tRangeMarkers ZepBuffer::GetRangeMarkers(uint32_t markerType) const {
        tRangeMarkers markers;
        ForEachMarker(markerType, Direction::Forward, Begin(), End(), [&](const std::shared_ptr<RangeMarker>& spMarker) {
            if ((spMarker->markerType & markerType) != 0) {
                markers[spMarker->GetRange().first].insert(spMarker);
            }
            return true;
        });
        return markers;
    }

    std::shared_ptr<RangeMarker> ZepBuffer::FindNextMarker(GlyphIterator start, Direction dir, uint32_t markerType) {
        start.Clamp();

        std::shared_ptr<RangeMarker> spFound;
        auto search = [&]() {
            ForEachMarker(markerType, dir, Begin(), End(), [&](const std::shared_ptr<RangeMarker>& marker) {
                if (dir == Direction::Forward) {
                    if (marker->GetRange().first <= start.Index()) {
                        return true;
                    }
                } else {
                    if (marker->GetRange().first >= start.Index()) {
                        return true;
                    }
                }

                spFound = marker;
                return false;
            });
        };

        search();
        if (spFound == nullptr) {
            // Wrap
            start = (dir == Direction::Forward ? Begin() : End());
            search();
        }
        return spFound;
    }

    void ZepBuffer::SetBufferType(BufferType type) {
        m_bufferType = type;
    }

    BufferType ZepBuffer::GetBufferType() const {
        return m_bufferType;
    }

    void ZepBuffer::SetLastEditLocation(GlyphIterator loc) {
        m_lastEditLocation = loc;
    }

    GlyphIterator ZepBuffer::GetLastEditLocation() {
        if (!m_lastEditLocation.Valid()) {
            m_lastEditLocation = GlyphIterator(this, 0);
        }
        return m_lastEditLocation;
    }

    ZepMode* ZepBuffer::GetMode() const {
        if (m_spMode) {
            return m_spMode.get();
        }
        return GetEditor().GetGlobalMode();
    }

    void ZepBuffer::SetMode(std::shared_ptr<ZepMode> spMode) {
        m_spMode = spMode;
    }

    tRangeMarkers ZepBuffer::GetRangeMarkersOnLine(uint32_t markerTypes, long line) const {
        ByteRange range;
        GetLineOffsets(line, range);

        tRangeMarkers rangeMarkers;
        ForEachMarker(markerTypes,
                      Zep::Direction::Forward,
                      GlyphIterator(this, range.first), GlyphIterator(this, range.second),
                      [&](const std::shared_ptr<RangeMarker>& marker) {
                          rangeMarkers[marker->GetRange().first].insert(marker);
                          return true;
                      });
        return rangeMarkers;
    }

    bool ZepBuffer::IsHidden() const {
        auto windows = GetEditor().FindBufferWindows(this);
        return windows.empty();
    }

    void ZepBuffer::SetFileFlags(uint32_t flags, bool set) {
        m_fileFlags = ZSetFlags(m_fileFlags, flags, set);
        GetEditor().UpdateTabs();
    }

    void ZepBuffer::ClearFileFlags(uint32_t flags) {
        m_fileFlags = ZClearFlags(m_fileFlags, flags);
        GetEditor().UpdateTabs();
    }

    bool ZepBuffer::HasFileFlags(uint32_t flags) const {
        return ZTestFlags(m_fileFlags, flags);
    }

    void ZepBuffer::ToggleFileFlag(uint32_t flags) {
        m_fileFlags = ZSetFlags(m_fileFlags, flags, !ZTestFlags(m_fileFlags, flags));
    }

    GlyphRange ZepBuffer::GetExpression(ExpressionType type, const GlyphIterator location, const std::vector<char>& beginExpression, const std::vector<char>& endExpression) const {
        GlyphIterator itr = Begin();
        GlyphIterator itrEnd = End();

        int maxDepth = -1;
        struct Expression {
            int depth = 0;
            GlyphRange range;
            std::vector<std::shared_ptr<Expression>> children;
            Expression* pParent = nullptr;
        };

        std::vector<std::shared_ptr<Expression>> topLevel;

        Expression* pCurrent = nullptr;
        Expression* pInner = nullptr;

        while (itr != itrEnd) {
            auto ch = itr.Char();

            for (auto& exp : beginExpression) {
                if (exp == ch) {
                    auto spChild = std::make_shared<Expression>();
                    if (pCurrent) {
                        pCurrent->children.push_back(spChild);
                        spChild->pParent = pCurrent;
                        spChild->depth = pCurrent->depth + 1;
                    } else {
                        topLevel.push_back(spChild);
                    }
                    pCurrent = spChild.get();
                    pCurrent->range.first = itr;
                }
            }

            for (auto& exp : endExpression) {
                if (exp == ch) {
                    if (pCurrent) {
                        pCurrent->range.second = itr.Peek(1);

                        // Check the sub exp
                        if ((pCurrent->range.first <= location) && (pCurrent->range.second > location)) {
                            if (pCurrent->depth > maxDepth) {
                                maxDepth = pCurrent->depth;
                                pInner = pCurrent;
                            }
                        }
                        pCurrent = pCurrent->pParent;
                    }
                }
            }

            itr.Move(1);
        }

        if (type == ExpressionType::Inner) {
            if (pInner) {
                return pInner->range;
            }
            return GlyphRange(Begin(), Begin());
        }

        Expression* pBest = nullptr;
        int dist = std::numeric_limits<int>::max();

        for (auto& outer : topLevel) {
            if (location >= outer->range.first && location < outer->range.second) {
                return outer->range;
            } else {
                auto leftDist = std::abs(outer->range.first.Index() - location.Index());
                auto rightDist = std::abs(location.Index() - outer->range.second.Index());
                if (leftDist < dist) {
                    pBest = outer.get();
                    dist = leftDist;
                }
                if (rightDist < dist) {
                    pBest = outer.get();
                    dist = rightDist;
                }
            }
        }

        if (pBest) {
            return pBest->range;
        }

        return GlyphRange(Begin(), Begin());
    }

    GlyphIterator ZepBuffer::End() const {
        return GlyphIterator(this, std::max(0l, long(m_workingBuffer.size() - 1)));
    }

    GlyphIterator ZepBuffer::Begin() const {
        return GlyphIterator(this);
    }

    void ZepBuffer::SetPostKeyNotifier(fnKeyNotifier notifier) {
        m_postKeyNotifier = notifier;
    }

    fnKeyNotifier ZepBuffer::GetPostKeyNotifier() const {
        return m_postKeyNotifier;
    }

    void ZepBuffer::EndFlash() const {
        GetEditor().SetFlags(ZClearFlags(GetEditor().GetFlags(), ZepEditorFlags::FastUpdate));
    }

    void ZepBuffer::BeginFlash(float seconds, FlashType flashType, const GlyphRange& range) {
        if (range.first == range.second) {
            return;
        }

        auto spMarker = std::make_shared<RangeMarker>(*this);
        spMarker->SetRange(ByteRange(range.first.Index(), range.second.Index()));
        spMarker->SetBackgroundColor(ThemeColor::FlashColor);
        spMarker->displayType = RangeMarkerDisplayType::Timed | RangeMarkerDisplayType::Background;
        spMarker->markerType = RangeMarkerType::Mark;
        spMarker->duration = seconds;
        spMarker->flashType = flashType;
        timer_restart(spMarker->timer);

        GetEditor().SetFlags(ZSetFlags(GetEditor().GetFlags(), ZepEditorFlags::FastUpdate));
    }

    // Conversion to handle for clients that don't want to know about ZepBuffer, but pass
    // around references to them
    uint64_t ZepBuffer::ToHandle() const {
        return (uint64_t)this;
    }

    ZepBuffer* ZepBuffer::FromHandle(ZepEditor& editor, uint64_t handle) {
        return editor.GetBufferFromHandle(handle);
    }

    const std::string& ZepBuffer::GetName() const {
        if (!m_filePath.empty() && m_filePath.has_filename()) {
            m_strName = m_filePath.filename().string();
        }
        return m_strName;
    }

    std::stack<std::shared_ptr<ZepCommand>>& ZepBuffer::GetUndoStack() {
        return m_undoStack;
    }

    std::stack<std::shared_ptr<ZepCommand>>& ZepBuffer::GetRedoStack() {
        return m_redoStack;
    }

    GlyphIterator ZepBuffer::FindMatchingParen(GlyphIterator bufferCursor) {
        int32_t findIndex = 0;
        std::string delims = "\n(){}[]";
        Direction dir = Direction::Forward;

        GlyphIterator loc;
        loc = FindFirstCharOf(bufferCursor, delims, findIndex, dir);

        if (findIndex > 0) {
            // Make a new end location
            auto end_loc = loc;

            std::string closing;
            std::string opening = std::string(1, delims[findIndex]);

            // opening bracket
            if (findIndex & 0x1) {
                end_loc++;
                closing = delims[findIndex + 1];
            } else {
                end_loc--;
                closing = delims[findIndex - 1];
                dir = Direction::Backward;
            }
            std::string openClose = opening + closing;

            // Track open/close braket pairs
            int closingCount = 1;

            for (;;) {
                // Find the next open or close of the current delim type
                int newIndex;
                end_loc = FindFirstCharOf(end_loc, openClose, newIndex, dir);

                // Fell off, no find
                if (newIndex < 0) {
                    break;
                }

                // Found another opener/no good
                if (newIndex == 0) {
                    closingCount++;
                }
                // Found a closer
                else if (newIndex == 1) {
                    closingCount--;
                    if (closingCount == 0) {
                        return end_loc;
                    }
                }

                if (dir == Direction::Forward) {
                    if (end_loc == End()) {
                        break;
                    }
                    end_loc++;
                } else {
                    if (end_loc == Begin()) {
                        break;
                    }
                    end_loc--;
                }
            }
        }
        return GlyphIterator();
    }

} // namespace Zep
