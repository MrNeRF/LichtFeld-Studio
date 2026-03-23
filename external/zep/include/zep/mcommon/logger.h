/*
 * https://stackoverflow.com/questions/5028302/small-logger-class
 * File:   ZLog.h
 * Author: Alberto Lepe <dev@alepe.com>
 *
 * Created on December 1, 2015, 6:00 PM
 * Modified by cmaughan
 */

#pragma once
#include <iostream>
#include <sstream>
#include <thread>

#ifdef WIN32
// A reference to the debug API on windows, to help the logger output in VC.  This is better
// than out to the console sometimes, and as long as you are building on Windows, you are referencing the necessary
// kernel32.dll....
extern "C" {
__declspec(dllimport) void __stdcall OutputDebugStringA(_In_opt_ const char* pszChar);
}
#endif

#undef ERROR

namespace Zep {

    enum class ZLT {
        NONE,
        DBG,
        INFO,
        WARNING,
        ERROR,
        ALWAYS
    };

    struct ZLogger {
        bool headers = false;
        ZLT level = ZLT::WARNING;
    };

    extern ZLogger logger;

    class ZLog {
    public:
        ZLog() {
        }
        ZLog(ZLT type) {
            msglevel = type;
            if (logger.headers && msglevel >= logger.level) {
                operator<<("[" + getLabel(type) + "] ");
            }
            out << "(T:" << std::this_thread::get_id() << ") ";
        }
        ~ZLog() {
            if (opened) {
                out << std::endl;
#ifdef WIN32
                OutputDebugStringA(out.str().c_str());
#else
                std::cout << out.str();
#endif
            }
            opened = false;
        }
        template <class T>
        ZLog& operator<<(const T& msg) {
            if (disabled || msglevel < logger.level)
                return *this;
            out << msg;
            opened = true;
            return *this;
        }

        static bool disabled;

    private:
        bool opened = false;
        ZLT msglevel = ZLT::DBG;
        inline std::string getLabel(ZLT type) {
            std::string label;
            switch (type) {
            case ZLT::DBG:
                label = "DEBUG";
                break;
            case ZLT::INFO:
                label = "INFO ";
                break;
            case ZLT::WARNING:
                label = "WARN ";
                break;
            case ZLT::ERROR:
                label = "ERROR";
                break;
            case ZLT::ALWAYS:
                label = "ALWAYS";
                break;
            case ZLT::NONE:
                label = "NONE";
                break;
            }
            return label;
        }
        std::ostringstream out;
    };

#ifndef ZLOG
#ifdef _DEBUG
#define ZLOG(a, b) ZLog(Zep::ZLT::a) << b
#else
#define ZLOG(a, b)
#endif
#endif

} // namespace Zep
