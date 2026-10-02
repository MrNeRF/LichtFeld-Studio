/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ObserverPtr.h>
#include <cstddef>
#include <utility>

namespace lfs::vis::gui {

    // Copyable observer of an Rml::Element. With a shared RmlUi on Windows,
    // Rml::ObserverPtr<Element> is a dllimport template that exports only the
    // members RmlUi instantiates itself (move, nullptr construction, get), so
    // its default and copy operations fail to link. Copies re-observe the
    // element through Element::GetObserverPtr instead.
    class ElementObserver {
    public:
        ElementObserver() noexcept : ptr_(nullptr) {}
        ElementObserver(std::nullptr_t) noexcept : ptr_(nullptr) {}
        ElementObserver(Rml::ObserverPtr<Rml::Element>&& ptr) noexcept : ptr_(std::move(ptr)) {}
        ElementObserver(const ElementObserver& other) : ptr_(observe(other.get())) {}
        ElementObserver(ElementObserver&& other) noexcept : ptr_(std::move(other.ptr_)) {}

        ElementObserver& operator=(const ElementObserver& other) {
            if (this != &other)
                ptr_ = observe(other.get());
            return *this;
        }
        ElementObserver& operator=(ElementObserver&& other) noexcept {
            ptr_ = std::move(other.ptr_);
            return *this;
        }
        ElementObserver& operator=(Rml::ObserverPtr<Rml::Element>&& ptr) noexcept {
            ptr_ = std::move(ptr);
            return *this;
        }
        ElementObserver& operator=(std::nullptr_t) noexcept {
            ptr_ = Rml::ObserverPtr<Rml::Element>(nullptr);
            return *this;
        }

        [[nodiscard]] Rml::Element* get() const noexcept { return ptr_.get(); }
        Rml::Element* operator->() const noexcept { return get(); }
        explicit operator bool() const noexcept { return get() != nullptr; }

    private:
        static Rml::ObserverPtr<Rml::Element> observe(Rml::Element* const element) {
            return element ? element->GetObserverPtr() : Rml::ObserverPtr<Rml::Element>(nullptr);
        }

        Rml::ObserverPtr<Rml::Element> ptr_;
    };

} // namespace lfs::vis::gui
