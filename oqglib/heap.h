#pragma once
#include <cstddef>
#include <limits>
#include <utility>
#include <type_traits>

namespace gg {

template <
    size_t   K,
    typename Item,              
    bool     KeepSmallest = true 
>
struct FixedKHeap {
    static_assert(K > 0, "K must be > 0");

    Item   heap[K];
    size_t sz = 0;

    inline bool   empty() const noexcept { return sz == 0; }
    inline size_t size()  const noexcept { return sz; }
    inline bool   full()  const noexcept { return sz == K; }

    inline auto bound() const noexcept {
        if (sz < K) {
            if constexpr (KeepSmallest) 
                return std::numeric_limits<decltype(heap[0].dist)>::infinity();
            else
                return std::numeric_limits<decltype(heap[0].dist)>::lowest();
        }
        return heap[0].dist;
    }

    inline const Item& top() const noexcept {
        return heap[0];
    }

    inline bool try_push(const Item& x) noexcept {
        if (sz < K) {
            size_t i = sz++;
            heap[i] = x;
            sift_up(i);
            return true;
        }
        if constexpr (KeepSmallest) {
            if (!(x < heap[0])) return false; 
        } else {
            if (!(x > heap[0])) return false;
        }
        heap[0] = x;
        sift_down(0);
        return true;
    }

    inline Item pop() noexcept {
        Item top = heap[0];
        heap[0] = heap[--sz];
        if (sz > 0) sift_down(0);
        return top;
    }

    inline void sort_ascending() noexcept {
        for (size_t n = sz; n > 1; --n) {
            std::swap(heap[0], heap[n-1]);
            sift_down_range(0, n-1);
        }
    }

private:
    inline void sift_up(size_t i) noexcept {
        Item x = heap[i];
        while (i) {
            size_t p = (i - 1) >> 1;
            if constexpr (KeepSmallest) {
                if (!(heap[p] < x)) break;  // 父 >= 子，停止
            } else {
                if (!(heap[p] > x)) break;  // 父 <= 子，停止
            }
            heap[i] = heap[p];
            i = p;
        }
        heap[i] = x;
    }

    inline void sift_down(size_t i) noexcept { sift_down_range(i, sz); }

    inline void sift_down_range(size_t i, size_t n) noexcept {
        Item x = heap[i];
        while (true) {
            size_t l = i*2 + 1;
            if (l >= n) break;
            size_t r = l + 1;
            size_t c;
            if constexpr (KeepSmallest) {
                // max-heap: 选更大孩子
                c = (r < n && heap[l] < heap[r]) ? r : l;
                if (!(x < heap[c])) break;
            } else {
                // min-heap: 选更小孩子
                c = (r < n && heap[l] > heap[r]) ? r : l;
                if (!(x > heap[c])) break;
            }
            heap[i] = heap[c];
            i = c;
        }
        heap[i] = x;
    }
};


template <typename Item>
struct MinHeapFast {
    std::vector<Item> h;

    inline bool   empty() const noexcept { return h.empty(); }
    inline size_t size()  const noexcept { return h.size(); }
    inline const Item& top() const noexcept { return h[0]; }

    inline void push(const Item& x) noexcept {
        h.push_back(x);
        sift_up(h.size() - 1);
    }

    inline Item pop() noexcept {
        Item t = h[0];
        h[0] = h.back();
        h.pop_back();
        if (!h.empty()) sift_down(0);
        return t;
    }

    inline auto next_id() const { return h[0].vecID; }
    inline auto next_next_id() const {
        if (h.size() == 2) return h[1].vecID;
        return (h[1] < h[2]) ? h[1].vecID : h[2].vecID; 
    }

private:
    inline void sift_up(size_t i) noexcept {
        Item x = h[i];
        while (i) {
            size_t p = (i - 1) >> 1;
            if (!(x < h[p])) break;      // parent <= x  -> OK
            h[i] = h[p];
            i = p;
        }
        h[i] = x;
    }
    inline void sift_down(size_t i) noexcept {
        Item x = h[i];
        const size_t n = h.size();
        while (true) {
            size_t l = i*2 + 1;
            if (l >= n) break;
            size_t r = l + 1;
            size_t c = (r < n && h[r] < h[l]) ? r : l; // pick smaller child
            if (!(h[c] < x)) break;                    // x <= child -> stop
            h[i] = h[c];
            i = c;
        }
        h[i] = x;
    }
};


template <typename Item>
struct SearchBuffer {
    std::vector<Item> h;

    static_assert(std::is_trivially_copyable<Item>::value,
                  "SearchBuffer requires trivially copyable Item for noexcept memmove.");
    
    inline bool   empty() const noexcept { return h.empty(); }
    inline size_t size()  const noexcept { return h.size(); }
    inline const Item& top() const noexcept { return h[0]; }

    inline void push(const Item& x) noexcept {
        const size_t n = h.size();
        h.emplace_back();                // 先扩一位
        size_t lo = 0, hi = n;
        while (lo < hi) {
            size_t mid = (lo + hi) >> 1;
            if (h[mid] < x) lo = mid + 1;
            else            hi = mid;
        }
        if (n > lo) {
            std::memmove(&h[lo + 1], &h[lo], (n - lo) * sizeof(Item));
        }
        h[lo] = x;
    }

    inline Item pop() noexcept {
        Item t = h[0];
        const size_t n = h.size();
        if (n == 1) {
            h.pop_back();
            return t;
        }
        std::memmove(&h[0], &h[1], (n - 1) * sizeof(Item));
        h.pop_back();
        return t;
    }

    inline auto next_id() const { return h[0].vecID; }
};

}