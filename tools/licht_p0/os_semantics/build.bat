@echo off
setlocal
cl /std:c++20 /EHsc /O2 /W3 /Fe:os_semantics.exe os_semantics.cpp
if errorlevel 1 exit /b 1
echo built os_semantics.exe
