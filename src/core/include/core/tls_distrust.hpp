/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <openssl/evp.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace lfs::core {

    // SHA-256 of the SubjectPublicKeyInfo of the eight eMudhra intermediate CAs whose private
    // keys are recoverable from their certificates (disclosed 2026-10-20). Their Sectigo
    // cross-signs carry the same keys. OpenSSL on Windows trusts the roots behind them and
    // never sees Windows' own distrust, so OpenSSL clients must refuse these keys themselves.
    inline constexpr std::array<std::string_view, 8> kDistrustedCaSpkiSha256 = {
        // EM DV TLS CA - G3B
        "ebed6475c2c2d80fcb199f3fe8dd6e6b3ce60b77eb15d4e42c3dec0c6f5b5105",
        // EM OV TLS CA - G3B
        "43bffa19d6ef651451f6841e019931e96f6f2eaeff28ab4b6f1ce4576de2717b",
        // EM EV TLS CA - G3B
        "e0254c7fcdc265e2f0f8a428dabb75ab4ae039b4872d62f5ee57f0877d5a8a41",
        // EM DV TLS CA - G3C
        "49dd353dc2b919a3770645117d0e60c16115e7d1bc259eefe734b00ea434fe5f",
        // EM OV TLS CA - G3C
        "63ecefee431d49523bc1136c4e72d0206a477086caf606cd5cf025686c109cf8",
        // EM EV TLS CA - G3C
        "5c0660a2b164910d8ff589c18e14552e9e0ec237dcefe2cb736d41a4811f81a0",
        // EM OV CS CA - G3B
        "ec4ffa1ad0ac88081314b7aaffef6a91f97cd57291bb5ba686c5415f5eb4f6b6",
        // EM OV CS CA - G3C
        "92c063d67e8f14db71d288e11266ffdaf69cbf115f7bab14f3bc2f68d25e0e30",
    };

    inline std::string spki_sha256_hex(const X509* cert) {
        unsigned char* der = nullptr;
        const int der_len = i2d_X509_PUBKEY(X509_get_X509_PUBKEY(cert), &der);
        if (der_len <= 0)
            return {};

        unsigned char digest[EVP_MAX_MD_SIZE];
        unsigned int digest_len = 0;
        const bool hashed = EVP_Digest(der, static_cast<size_t>(der_len), digest, &digest_len,
                                       EVP_sha256(), nullptr) == 1;
        OPENSSL_free(der);
        if (!hashed)
            return {};

        static constexpr char kHex[] = "0123456789abcdef";
        std::string hex;
        hex.reserve(digest_len * 2);
        for (unsigned int i = 0; i < digest_len; ++i) {
            hex.push_back(kHex[digest[i] >> 4]);
            hex.push_back(kHex[digest[i] & 0x0F]);
        }
        return hex;
    }

    inline bool is_distrusted_ca(const X509* cert) {
        return std::ranges::find(kDistrustedCaSpkiSha256, spki_sha256_hex(cert)) != kDistrustedCaSpkiSha256.end();
    }

    inline bool contains_distrusted_ca(const STACK_OF(X509) * chain) {
        for (int i = 0; i < sk_X509_num(chain); ++i) {
            if (is_distrusted_ca(sk_X509_value(chain, i)))
                return true;
        }
        return false;
    }

    // Checks what the server sent as well as the chain OpenSSL built, which can include
    // intermediates taken from the local store.
    inline bool peer_chain_has_distrusted_ca(const SSL* ssl) {
        return contains_distrusted_ca(SSL_get_peer_cert_chain(ssl)) ||
               contains_distrusted_ca(SSL_get0_verified_chain(ssl));
    }

} // namespace lfs::core
