// Exact per-cell credits of a convex polytope on the 128^3 grid (GMP, trusted-kernel helper).
//
// Mirrors cbc_kernel.Kernel._grid_credit: adaptive octree over the polytope's cell range,
// each node carrying the exact clip of the polytope to its block; a block whose 8 corners
// satisfy every facet row lies inside the convex polytope and gets exact cell volumes; a
// single boundary cell gets floor/ceil(vol * 2^160) of its exact clip. All arithmetic is
// exact (mpz/mpq). The Python caller re-checks conservation against the independent
// Python rational volume of the polytope.
//
// Input polytope: faces as cyclic lists of homogeneous integer vertices (x, y, z, w), w > 0,
// exactly the representation of global_inverse/poly.py.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gmpxx.h>

#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;
using Z = mpz_class;
using QQ = mpq_class;
using V = std::array<Z, 4>;
using Row = std::array<Z, 4>;

struct Poly {
    std::vector<V> v;
    std::vector<std::vector<int>> f;
};

static Z to_z(py::handle h) {
    // Python int -> mpz through its hexadecimal representation (exact).
    py::object s = py::module_::import("builtins").attr("format")(h, "x");
    std::string t = s.cast<std::string>();
    Z z;
    if (!t.empty() && t[0] == '-') {
        z.set_str(t.substr(1), 16);
        z = -z;
    } else {
        z.set_str(t, 16);
    }
    return z;
}

static py::object to_py(const Z& z) {
    std::string s = z.get_str(16);
    PyObject* o = PyLong_FromString(s.c_str(), nullptr, 16);
    if (!o) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(o);
}

static Z form(const Row& r, const V& p) { return r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3] * p[3]; }

static void primitive(V& p) {
    Z g;
    mpz_gcd(g.get_mpz_t(), p[0].get_mpz_t(), p[1].get_mpz_t());
    mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), p[2].get_mpz_t());
    mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), p[3].get_mpz_t());
    if (g != 0 && g != 1)
        for (auto& x : p) mpz_divexact(x.get_mpz_t(), x.get_mpz_t(), g.get_mpz_t());
}

static Row primitive_row(Row r) {
    V t = r;
    primitive(t);
    return t;
}

// Exact convex-polygon order of cap points in an injective coordinate projection
// (same construction as poly.cap_hull: lexicographic sort, monotone chain, strict turns).
static std::vector<int> cap_hull(const std::vector<V>& pts, const std::vector<int>& idx, const Row& plane) {
    int drop = 0;
    for (int k = 1; k < 3; ++k)
        if (abs(plane[k]) > abs(plane[drop])) drop = k;
    int I = drop == 0 ? 1 : 0, J = drop == 2 ? 1 : 2;
    std::vector<int> p = idx;
    auto less = [&](int a, int b) {
        // compare (x_I/w, x_J/w) lexicographically, exactly (w > 0)
        Z l = pts[a][I] * pts[b][3], r = pts[b][I] * pts[a][3];
        if (l != r) return l < r;
        return pts[a][J] * pts[b][3] < pts[b][J] * pts[a][3];
    };
    std::sort(p.begin(), p.end(), less);
    auto chain = [&](const std::vector<int>& items) {
        std::vector<int> out;
        for (int c : items) {
            while (out.size() >= 2) {
                const V& a = pts[out[out.size() - 2]];
                const V& b = pts[out[out.size() - 1]];
                const V& cc = pts[c];
                Z dx = b[I] * a[3] - a[I] * b[3], dy = b[J] * a[3] - a[J] * b[3];
                if (dx * (cc[J] * a[3] - a[J] * cc[3]) > dy * (cc[I] * a[3] - a[I] * cc[3])) break;
                out.pop_back();
            }
            out.push_back(c);
        }
        return out;
    };
    std::vector<int> lower = chain(p);
    std::vector<int> rev(p.rbegin(), p.rend());
    std::vector<int> upper = chain(rev);
    std::vector<int> hull(lower.begin(), lower.end() - 1);
    hull.insert(hull.end(), upper.begin(), upper.end() - 1);
    return hull;
}

// Split by row: returns (negative-side closure, positive-side closure); empty Poly = None.
static std::pair<Poly, Poly> split(const Poly& P, const Row& row, bool& neg_empty, bool& pos_empty) {
    std::vector<Z> s(P.v.size());
    bool anyp = false, anyn = false;
    for (size_t i = 0; i < P.v.size(); ++i) {
        s[i] = form(row, P.v[i]);
        if (s[i] > 0) anyp = true;
        if (s[i] < 0) anyn = true;
    }
    neg_empty = pos_empty = false;
    if (!anyp) { pos_empty = true; return {P, Poly{}}; }
    if (!anyn) { neg_empty = true; return {Poly{}, P}; }
    // new vertex table: original vertices + intersection points (cached per edge)
    std::vector<V> pts = P.v;
    std::vector<Z> sg = s;
    std::map<std::pair<int, int>, int> cut;
    std::vector<int> capidx;
    std::vector<char> incap(pts.size(), 0);
    std::vector<std::vector<int>> nf, pf;
    for (const auto& face : P.f) {
        std::vector<int> a, b;
        int last = face.back();
        for (int p : face) {
            const Z &u = s[last], &w = s[p];
            if ((u < 0 && w > 0) || (w < 0 && u > 0)) {
                std::pair<int, int> key = last < p ? std::make_pair(last, p) : std::make_pair(p, last);
                auto it = cut.find(key);
                int h;
                if (it == cut.end()) {
                    V q;
                    Z au = abs(u), aw = abs(w);
                    for (int k = 0; k < 4; ++k) q[k] = aw * P.v[last][k] + au * P.v[p][k];
                    primitive(q);
                    h = (int)pts.size();
                    pts.push_back(q);
                    incap.push_back(0);
                    cut[key] = h;
                } else {
                    h = it->second;
                }
                a.push_back(h);
                b.push_back(h);
                if (!incap[h]) { incap[h] = 1; capidx.push_back(h); }
            }
            if (w <= 0) a.push_back(p);
            if (w >= 0) b.push_back(p);
            if (w == 0 && !incap[p]) { incap[p] = 1; capidx.push_back(p); }
            last = p;
        }
        if (a.size() >= 3) nf.push_back(a);
        if (b.size() >= 3) pf.push_back(b);
    }
    std::vector<int> capf = cap_hull(pts, capidx, row);
    nf.push_back(capf);
    pf.push_back(capf);
    auto build = [&](const std::vector<std::vector<int>>& faces) {
        Poly out;
        std::map<int, int> remap;
        for (const auto& face : faces) {
            std::vector<int> g;
            for (int i : face) {
                auto it = remap.find(i);
                if (it == remap.end()) {
                    int j = (int)out.v.size();
                    out.v.push_back(pts[i]);
                    remap[i] = j;
                    g.push_back(j);
                } else {
                    g.push_back(it->second);
                }
            }
            out.f.push_back(g);
        }
        return out;
    };
    return {build(nf), build(pf)};
}

// Exact volume: sum over fan triangles of |det4(o, a, b, c)| / (6 wo wa wb wc).
static QQ volume(const Poly& P) {
    const V& o = P.v[0];
    QQ total = 0;
    for (const auto& face : P.f) {
        const V& a = P.v[face[0]];
        for (size_t k = 1; k + 1 < face.size(); ++k) {
            const V& b = P.v[face[k]];
            const V& c = P.v[face[k + 1]];
            // differences scaled to common w: (a/wa - o/wo) * wa*wo = a*wo - o*wa
            Z ax = a[0] * o[3] - o[0] * a[3], ay = a[1] * o[3] - o[1] * a[3], az = a[2] * o[3] - o[2] * a[3];
            Z bx = b[0] * o[3] - o[0] * b[3], by = b[1] * o[3] - o[1] * b[3], bz = b[2] * o[3] - o[2] * b[3];
            Z cx = c[0] * o[3] - o[0] * c[3], cy = c[1] * o[3] - o[1] * c[3], cz = c[2] * o[3] - o[2] * c[3];
            Z det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
            Z den = a[3] * b[3] * c[3] * o[3] * o[3] * o[3];
            total += QQ(abs(det), den);
        }
    }
    total /= 6;
    return total;
}

static std::vector<Row> facet_rows(const Poly& P) {
    std::vector<Row> out;
    for (const auto& face : P.f) {
        const V& a = P.v[face[0]];
        auto d = [&](const V& b) {
            return std::array<Z, 3>{b[0] * a[3] - a[0] * b[3], b[1] * a[3] - a[1] * b[3], b[2] * a[3] - a[2] * b[3]};
        };
        auto u = d(P.v[face[1]]);
        std::array<Z, 3> n{0, 0, 0};
        bool ok = false;
        for (size_t k = 2; k < face.size(); ++k) {
            auto w = d(P.v[face[k]]);
            n = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
            if (n[0] != 0 || n[1] != 0 || n[2] != 0) { ok = true; break; }
        }
        if (!ok) continue;
        Row r{n[0] * a[3], n[1] * a[3], n[2] * a[3], -(n[0] * a[0] + n[1] * a[1] + n[2] * a[2])};
        r = primitive_row(r);
        bool flip = false;
        for (const auto& q : P.v)
            if (form(r, q) > 0) { flip = true; break; }
        if (flip) for (auto& x : r) x = -x;
        for (const auto& q : P.v)
            if (form(r, q) > 0) throw std::runtime_error("inconsistent face structure");
        for (int i : face)
            if (form(r, P.v[i]) != 0) throw std::runtime_error("inconsistent face structure");
        out.push_back(r);
    }
    if (out.size() < 4) throw std::runtime_error("polytope with fewer than four facets");
    return out;
}

static Poly from_py(py::handle faces) {
    Poly P;
    std::map<std::string, int> index;  // key: decimal tuple text (exact identity of primitive vertices)
    for (py::handle face : faces) {
        std::vector<int> f;
        for (py::handle vert : face) {
            std::string key = py::str(vert).cast<std::string>();
            auto it = index.find(key);
            if (it == index.end()) {
                V q;
                int k = 0;
                for (py::handle c : vert) q[k++] = to_z(c);
                if (k != 4 || q[3] <= 0) throw std::runtime_error("vertex must be (x, y, z, w) with w > 0");
                int j = (int)P.v.size();
                P.v.push_back(q);
                index[key] = j;
                f.push_back(j);
            } else {
                f.push_back(it->second);
            }
        }
        P.f.push_back(f);
    }
    return P;
}

// Returns (list of (cell, low, high), boundary_cell_count, interior_cell_count).
static py::tuple grid_credit(py::handle faces, int n, py::handle scale_bits_obj) {
    int scale_bits = scale_bits_obj.cast<int>();
    Poly P = from_py(faces);
    std::vector<Row> rows = facet_rows(P);
    // grid bounds (positive-volume range), as poly.grid_bounds
    int lo[3], hi[3];
    for (int k = 0; k < 3; ++k) {
        lo[k] = 1 << 30; hi[k] = -1;
        for (const auto& q : P.v) {
            Z num = q[k] * n;
            Z fl;  mpz_fdiv_q(fl.get_mpz_t(), num.get_mpz_t(), q[3].get_mpz_t());
            Z num2 = q[k] * n - 1;
            Z fl2; mpz_fdiv_q(fl2.get_mpz_t(), num2.get_mpz_t(), q[3].get_mpz_t());
            lo[k] = std::min(lo[k], (int)fl.get_si());
            hi[k] = std::max(hi[k], (int)fl2.get_si());
        }
    }
    Z scale = 1;
    scale <<= scale_bits;
    Z cell_units = scale / (Z(n) * n * n);
    if (cell_units * n * n * n != scale) throw std::runtime_error("cell units not exact");
    std::map<long, std::pair<Z, Z>> credit;
    long boundary = 0, interior = 0;
    struct Node { int b[6]; Poly q; };
    std::vector<Node> stack;
    stack.push_back(Node{{lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]}, P});
    while (!stack.empty()) {
        Node node = std::move(stack.back());
        stack.pop_back();
        int* b = node.b;
        bool inside = true;
        for (int xi = 0; xi < 2 && inside; ++xi)
            for (int yi = 0; yi < 2 && inside; ++yi)
                for (int zi = 0; zi < 2 && inside; ++zi) {
                    V c{Z(xi ? b[1] + 1 : b[0]), Z(yi ? b[3] + 1 : b[2]), Z(zi ? b[5] + 1 : b[4]), Z(n)};
                    for (const auto& r : rows)
                        if (form(r, c) > 0) { inside = false; break; }
                }
        if (inside) {
            for (int x = b[0]; x <= b[1]; ++x)
                for (int y = b[2]; y <= b[3]; ++y)
                    for (int z = b[4]; z <= b[5]; ++z) {
                        long cid = ((long)x * n + y) * n + z;
                        auto& e = credit[cid];
                        e.first += cell_units;
                        e.second += cell_units;
                        ++interior;
                    }
            continue;
        }
        if (b[0] == b[1] && b[2] == b[3] && b[4] == b[5]) {
            QQ m = volume(node.q) * QQ(scale);
            Z low, high;
            mpz_fdiv_q(low.get_mpz_t(), m.get_num_mpz_t(), m.get_den_mpz_t());
            mpz_cdiv_q(high.get_mpz_t(), m.get_num_mpz_t(), m.get_den_mpz_t());
            long cid = ((long)b[0] * n + b[2]) * n + b[4];
            auto& e = credit[cid];
            e.first += low;
            e.second += high;
            ++boundary;
            continue;
        }
        int ext[3] = {b[1] - b[0], b[3] - b[2], b[5] - b[4]};
        int ax = 0;
        for (int k = 1; k < 3; ++k)
            if (ext[k] > ext[ax]) ax = k;
        int mid = (b[2 * ax] + b[2 * ax + 1]) / 2;
        Row plane{0, 0, 0, Z(-(mid + 1))};
        plane[ax] = n;
        bool ne, pe;
        auto halves = split(node.q, plane, ne, pe);
        Node left{{b[0], b[1], b[2], b[3], b[4], b[5]}, Poly{}}, right = left;
        left.b[2 * ax + 1] = mid;
        right.b[2 * ax] = mid + 1;
        if (!ne) { left.q = std::move(halves.first); stack.push_back(std::move(left)); }
        if (!pe) { right.q = std::move(halves.second); stack.push_back(std::move(right)); }
    }
    py::list out;
    for (auto& kv : credit) out.append(py::make_tuple(kv.first, to_py(kv.second.first), to_py(kv.second.second)));
    return py::make_tuple(out, boundary, interior);
}

static py::tuple poly_volume(py::handle faces) {
    Poly P = from_py(faces);
    QQ v = volume(P);
    return py::make_tuple(to_py(v.get_num()), to_py(v.get_den()));
}

PYBIND11_MODULE(cbc_cells, m) {
    m.doc() = "Exact per-cell credits of convex polytopes (GMP)";
    m.def("grid_credit", &grid_credit, "octree exact per-cell (floor, ceil) credits at 2^-scale_bits");
    m.def("volume", &poly_volume, "exact rational volume (num, den)");
}
