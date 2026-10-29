/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace lfs::core {

#if defined(__cpp_lib_move_only_function)
    template <class Signature>
    using MoveOnlyFunction = std::move_only_function<Signature>;
#else
    // Apple's libc++ has no std::move_only_function yet. This covers the
    // unqualified R(Args...) form: a nullable, movable owner of one callable.
    template <class Signature>
    class MoveOnlyFunction;

    template <class R, class... Args>
    class MoveOnlyFunction<R(Args...)> {
    public:
        MoveOnlyFunction() noexcept = default;
        MoveOnlyFunction(std::nullptr_t) noexcept {}

        template <class F>
            requires(!std::is_same_v<std::remove_cvref_t<F>, MoveOnlyFunction> &&
                     std::is_invocable_r_v<R, std::decay_t<F>&, Args...>)
        MoveOnlyFunction(F&& callable)
            : callable_(std::make_unique<Model<std::decay_t<F>>>(std::forward<F>(callable))) {}

        MoveOnlyFunction(MoveOnlyFunction&&) noexcept = default;
        MoveOnlyFunction& operator=(MoveOnlyFunction&&) noexcept = default;
        MoveOnlyFunction& operator=(std::nullptr_t) noexcept {
            callable_.reset();
            return *this;
        }

        explicit operator bool() const noexcept { return callable_ != nullptr; }
        R operator()(Args... args) { return callable_->call(std::forward<Args>(args)...); }

    private:
        struct Callable {
            virtual ~Callable() = default;
            virtual R call(Args&&... args) = 0;
        };

        template <class F>
        struct Model final : Callable {
            template <class G>
            explicit Model(G&& callable) : callable(std::forward<G>(callable)) {}
            R call(Args&&... args) override { return std::invoke(callable, std::forward<Args>(args)...); }
            F callable;
        };

        std::unique_ptr<Callable> callable_;
    };
#endif

} // namespace lfs::core
