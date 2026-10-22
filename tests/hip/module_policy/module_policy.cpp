/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../../src/core/tensor/backend/hip/module_paths.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace lfs::core::internal;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(int argc,char** argv)try {
    require(argc==2,"Pass a fixture directory");
    const auto root=std::filesystem::absolute(argv[1]);
    const auto compiled=root/"build"/"module";
    const auto executable=root/"package";
    const auto generic=executable/"hip"/"module";
    const auto profile=generic.parent_path()/"profiles"/"gfx1150"/"module";
    std::filesystem::create_directories(profile.parent_path());
    std::filesystem::remove(generic);
    std::filesystem::remove(profile);
    require(hip_packaged_module(compiled,executable)==compiled,"Missing package changed developer path");
    require(hip_packaged_module(compiled,"relative")==compiled,"Relative executable directory was accepted");
    std::ofstream(generic).put('x');
    require(hip_packaged_module(compiled,executable)==generic,"Relocated module not selected");
    require(hip_profile_module(generic,"gfx1150")==generic,"Missing profile did not select generic");
    std::ofstream(profile).put('x');
    require(hip_profile_module(generic,"gfx1150:xnack-")==profile,"ISA feature suffix was not normalized");
    require(hip_profile_module(generic,"gfx1103")==generic,"Another architecture selected this profile");
    for(const auto bad:{"", "gfx", "../gfx1150", "gfx1150/..", "gfx1150,other", "GFX1150"})
        require(hip_profile_module(generic,bad)==generic,"Invalid architecture became a path");
    require(lfs_hip_has_architecture("gfx1103,gfx1150","gfx1150:xnack-"),"Fat target list rejected supported ISA");
    require(!lfs_hip_has_architecture("gfx1103,gfx1150","gfx110"),"Target prefix incorrectly matched");
    require(!lfs_hip_has_architecture("gfx1103,gfx1150","gfx942"),"Wave64 target incorrectly admitted");
    require(hip_executable_directory().is_absolute(),"Cannot locate executable for relocation");
    std::cout<<"PASS HIP module paths, relocation, architecture list and conservative selection\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
