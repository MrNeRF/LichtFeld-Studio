/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ply_io.hpp"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

namespace lfs::lod {

    namespace {
        struct Prop {
            std::string name;
            int bytes = 4; // float assumed; others rejected
        };
    } // namespace

    bool loadPly(const std::string& path, SplatCloud& out, std::string* err) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            if (err)
                *err = "cannot open " + path;
            return false;
        }
        auto fail = [&](const std::string& m) {
            if (err)
                *err = m;
            std::fclose(f);
            return false;
        };

        // -- header --
        std::string header;
        char line[512];
        size_t vertex_count = 0;
        std::vector<Prop> props;
        bool binary_le = false;
        while (std::fgets(line, sizeof line, f)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
                s.pop_back();
            if (s == "end_header")
                break;
            std::istringstream ss(s);
            std::string tok;
            ss >> tok;
            if (tok == "format") {
                std::string fmt;
                ss >> fmt;
                binary_le = (fmt == "binary_little_endian");
            } else if (tok == "element") {
                std::string what;
                ss >> what >> vertex_count;
                if (what != "vertex")
                    return fail("unsupported element " + what);
            } else if (tok == "property") {
                std::string type, name;
                ss >> type >> name;
                if (type != "float" && type != "float32")
                    return fail("non-float property " + name);
                props.push_back({name, 4});
            }
        }
        if (!binary_le)
            return fail("only binary_little_endian PLY supported");
        if (vertex_count == 0 || props.empty())
            return fail("empty PLY");

        int ix = -1, iy = -1, iz = -1, idc[3] = {-1, -1, -1}, isc[3] = {-1, -1, -1},
            irot[4] = {-1, -1, -1, -1}, iop = -1;
        for (int i = 0; i < (int)props.size(); ++i) {
            const std::string& n = props[i].name;
            if (n == "x")
                ix = i;
            else if (n == "y")
                iy = i;
            else if (n == "z")
                iz = i;
            else if (n == "opacity")
                iop = i;
            else if (n.rfind("f_dc_", 0) == 0) {
                int k = std::atoi(n.c_str() + 5);
                if (k >= 0 && k < 3)
                    idc[k] = i;
            } else if (n.rfind("scale_", 0) == 0) {
                int k = std::atoi(n.c_str() + 6);
                if (k >= 0 && k < 3)
                    isc[k] = i;
            } else if (n.rfind("rot_", 0) == 0) {
                int k = std::atoi(n.c_str() + 4);
                if (k >= 0 && k < 4)
                    irot[k] = i;
            }
        }
        if (ix < 0 || iy < 0 || iz < 0 || iop < 0 || isc[0] < 0 || irot[0] < 0 || idc[0] < 0)
            return fail("missing gaussian splat properties");

        const size_t stride = props.size();
        out.resize(vertex_count);
        std::vector<float> row(stride);
        for (size_t v = 0; v < vertex_count; ++v) {
            if (std::fread(row.data(), 4, stride, f) != stride)
                return fail("truncated PLY body");
            out.means[v] = {row[ix], row[iy], row[iz]};
            out.sh0[v] = {row[idc[0]], row[idc[1]], row[idc[2]]};
            out.log_scales[v] = {row[isc[0]], row[isc[1]], row[isc[2]]};
            out.rotations[v] = {row[irot[0]], row[irot[1]], row[irot[2]], row[irot[3]]};
            out.opacity_raw[v] = row[iop];
        }
        std::fclose(f);
        return true;
    }

    bool savePly(const std::string& path, const SplatCloud& cloud, std::string* err) {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) {
            if (err)
                *err = "cannot open " + path + " for write";
            return false;
        }
        std::ostringstream h;
        h << "ply\nformat binary_little_endian 1.0\n"
          << "element vertex " << cloud.size() << "\n";
        for (const char* p : {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"})
            h << "property float " << p << "\n";
        h << "end_header\n";
        const std::string hs = h.str();
        std::fwrite(hs.data(), 1, hs.size(), f);

        for (size_t i = 0; i < cloud.size(); ++i) {
            const float row[14] = {
                cloud.means[i].x, cloud.means[i].y, cloud.means[i].z,
                cloud.sh0[i].x, cloud.sh0[i].y, cloud.sh0[i].z,
                cloud.opacity_raw[i],
                cloud.log_scales[i].x, cloud.log_scales[i].y, cloud.log_scales[i].z,
                cloud.rotations[i].w, cloud.rotations[i].x, cloud.rotations[i].y, cloud.rotations[i].z};
            std::fwrite(row, 4, 14, f);
        }
        std::fclose(f);
        return true;
    }

} // namespace lfs::lod
