"""Build a source-bound, within-backend tile reuse experiment."""
import argparse
from pathlib import Path

from .build import D_BUILD, OUT, compile_study, replace_once
from .local import transform as local_transform


def transform(files):
    files = local_transform(files)
    name = "regions.hpp"
    text = files[name]
    text = replace_once(text, "class Engine {", """class TileSession;
class Engine {""")
    text = replace_once(text, "  std::shared_ptr<Impl> p;\n", """  std::shared_ptr<Impl> p;
  friend class TileSession;
  Result tile_scoped(P, int, std::array<int, 3>, int, double,
                     const Options &, const Settings &, TileSession *) const;
""")
    text = replace_once(text, "I certified_exp", """class TileSession {
  struct State;
  std::shared_ptr<State> p;
  friend class Engine;

public:
  TileSession(const Engine &, P, int, const Options &, const Settings &, bool);
  Result tile(std::array<int, 3>, int, double);
  std::map<std::string, double> info() const;
  std::map<std::string, std::uint64_t> preparation_work() const;
};
I certified_exp""")
    files[name] = text

    name = "regions.cpp"
    text = "#include <mutex>\n" + files[name]
    old = """Result Engine::tile(P source, int grid, std::array<int, 3> origin, int n,
                    double seconds, const Options &o,
                    const Settings &cfg) const {"""
    text = replace_once(text, old, old + """
  return tile_scoped(source, grid, origin, n, seconds, o, cfg, nullptr);
}
#include "tile_session.hpp"
Result Engine::tile_scoped(P source, int grid, std::array<int, 3> origin, int n,
                           double seconds, const Options &o,
                           const Settings &cfg, TileSession *session) const {""")
    text = replace_once(text, """  if (p->independent && cfg.study.method == 5)
    prepared_offsets = p->b->prepare_offsets(source, cfg.study.filter_bits, f.stats);
  Cover tree;""", """  const bool reuse = session && session->p->reuse_cover;
  if (p->independent && cfg.study.method == 5) {
    prepared_offsets = reuse ? session->p->offsets
                             : p->b->prepare_offsets(source, cfg.study.filter_bits, f.stats);
  }
  Cover cold_cover;
  Cover &tree = reuse ? session->p->cover : cold_cover;
  if (reuse) f.stats.add("tile_reused_cover");""")
    text = replace_once(text, "  if (usecover) {", "  if (usecover && !reuse) {")
    files[name] = text

    name = "bindings.cpp"
    text = files[name]
    text = replace_once(text, '  py::class_<Engine>(m, "Engine", py::module_local())',
        '''  py::class_<TileSession>(m, "TileSession", py::module_local())
      .def("info", &TileSession::info)
      .def("preparation_work", &TileSession::preparation_work)
      .def("tile", [](TileSession &session, std::array<int, 3> origin,
                      int side, double seconds) {
        Result r;
        {
          py::gil_scoped_release release;
          r = session.tile(origin, side, seconds);
        }
        return output(r, side);
      });
  py::class_<Engine>(m, "Engine", py::module_local())''')
    text = replace_once(text, '      .def("ray",',
        '''      .def("session",
           [](const Engine &engine, P source, int grid, const Options &o,
              const Settings &cfg, bool reuse_cover) {
             py::gil_scoped_release release;
             return TileSession(engine, source, grid, o, cfg, reuse_cover);
           })
      .def("ray",''')
    files[name] = text
    return files


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=D_BUILD)
    p.add_argument("--output", type=Path, default=OUT / "tile-native")
    args = p.parse_args()
    compile_study(args.baseline, args.output, "tiles", transform,
                  ["precision_search.hpp", "tile_session.hpp"])
