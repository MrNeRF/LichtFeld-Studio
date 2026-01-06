
rd /q /s build
rd /q /s dist
cmake -B build -DBUILD_PORTABLE=ON
cmake --build build -j 16 --config Release
cmake --install build --prefix ./dist`