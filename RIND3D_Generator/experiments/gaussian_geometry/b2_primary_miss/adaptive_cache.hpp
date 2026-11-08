// Shared dispatch state for separate primary or independent tile caches.
class AxialHistory {
  struct Axis {
    unsigned char failures = 0, cooldown = 0;
  };
  using Block = std::array<Axis, 3>;
  static constexpr int block_side = 8, retry_after = 12;
  static constexpr int failure_limit = 2;
  int dim;
  std::vector<Block> blocks;

  Block &at(I3 xyz) {
    return blocks[(xyz[2] / block_side * dim + xyz[1] / block_side) * dim +
                  xyz[0] / block_side];
  }

public:
  AxialHistory(int side, Stats &work)
      : dim(side / block_side), blocks(dim * dim * dim) {
    work.maximum("axial_history_payload_bytes",
                 blocks.capacity() * sizeof(Block));
  }

  bool permit(I3 xyz, int dir) {
    auto &state = at(xyz)[-dir - 1];
    if (!state.cooldown)
      return true;
    --state.cooldown;
    return false;
  }

  void observe(I3 xyz, const AxialResult &r) {
    auto &block = at(xyz);
    for (int k = 0; k < 3; ++k) {
      auto &state = block[k];
      if (r.long_status[k] > 0)
        state = {};
      else if (r.long_status[k] < 0) {
        state.failures = std::min<int>(state.failures + 1, failure_limit);
        if (state.failures == failure_limit)
          state.cooldown = retry_after;
      }
    }
  }
};
