# RIND3D local scene explorer

Start from `Phase3_Single_Source_3D` after installing the public dataset:

```bash
uv sync --extra viewer
./scripts/browser.sh
```

Open **http://127.0.0.1:8765**. Stop the server with Ctrl+C. Use `--port 8766` for another port. `--root /absolute/path/to/RIND3D-v0.1` or `RIND3D_DATA_ROOT` selects an existing installation. The module can also run as `python -m rind_phase3.browser --root ...`; the shell script is the ordinary entry point.

The local viewer does not upload or modify scenes. Its default bind address is loopback only. It supports the existing immutable RIND3D v0.1 convex-union release. NumPy, SciPy and the public reader are sufficient; neither a native geometry build, CUDA, PyTorch nor a physics query is needed. Private records are optional and only used for authorized true-source overlays. Three.js r170 and OrbitControls are bundled with their upstream MIT license, so browsing requires no external CDN. This third-party license does not assign a license to RIND3D itself.

## Controls

The left scene library supports ID/ordinal search, nine geometry families and train/val/test filters. On narrow screens, open it with the menu button. Scene lists are paginated in the interface; all 1,000 scenes remain available. Selecting a scene retrieves its geometry and all 256 window records without scanning every full window.

Drag the main 3D view with the left button to rotate, use the wheel to zoom and the right button to pan. `XY`, `XZ` and `YZ` choose axis views. **全局空间** frames the world; **聚焦窗口** frames the selected observation. **重置** restores an oblique view. **截图** exports the current 3D canvas; the fullscreen button expands it. Automatic rotation is optional and off initially. Idle rendering uses an on-demand loop.

**全局响应体积** displays the actual global response in 3D, initially as a translucent blue shadow volume. Its menu switches to gold lit volume, both classes, or the internal lit/shadow boundary surface. The density slider changes visualization opacity, not physical intensity. Solids are a separate label and are drawn using the obstacle meshes. **体积只显示窗口内** clips this preview to the selected window. Volume and exact slice planes can be shown together or independently.

Choose 64³ for responsiveness, 128³ for the default preview or 256³ for finer display. At preview index `i`, the original integer voxel sampled is `i*step + step//2`, with `step=1024/resolution`, on each axis. Each label is read exactly from packed response and solid bits; the coarser spatial display can miss thin structures. It is not a complete-volume equivalence check, a new response renderer, or an exact reconstructed continuous shadow boundary. The three 1024² slices remain available for original-voxel inspection.

The right panel selects any 64³/128³/256³ window and shows exact full-volume lit FREE, shadowed FREE and solid counts. The displayed percentage divides lit FREE by all FREE voxels. Exact counts are computed by the existing stored count index. Extreme ratios retain extra digits and the minority voxel count. Having both classes in a 3D window does not mean every 2D slice contains both.

Check X/Y/Z to show corresponding planes in 3D; the coordinate controls move them through integer voxel indices 0–1023. All three 2D slices are available even if a plane is hidden in 3D. **切片居中** moves them to the selected window's center. **切片随窗口居中** does this on later window selections. Clicking a 2D image sets the other two coordinates; hovering displays the exact original voxel coordinate/class. The `↗` on a slice opens the full original PNG without display overlays.

**全局 1024² / 当前窗口** switches the 2D crop. The local crop uses the selected window's actual arbitrary integer origin. If the selected plane lies outside that window, the interface says so instead of displaying an unrelated local response. The orange border appears on global slices only when they intersect the window. Checkbox **显示全部窗口边框** optionally adds all 256 bounds. Geometry opacity, plane opacity, edges and window clipping affect visualization only.

Keyboard shortcuts, when not editing a field: left/right change windows, G frames the world, F focuses the window, R resets the camera and ? opens help. The address bar keeps the scene ID, window ID and exact XYZ slice indices, so bookmarking/reopening reproduces the selected observation and planes.

## Coordinates and truth

All coordinates are XYZ; stored arrays are ZYX. 3D geometry uses the unit world. Integer voxel `(i,j,k)` is centered at `((i+.5)/1024,(j+.5)/1024,(k+.5)/1024)`. Slice rows/columns are Y/X for a Z plane, Z/X for a Y plane and Z/Y for an X plane. PNG top rows represent the largest vertical world coordinate. Texture interpolation is nearest-neighbor; no smooth lighting response is invented between stored bits.

Gold means lit FREE, dark blue shadowed FREE, grey solid. The orange cube marks the observation window. Surface shading on obstacle meshes helps explain shape and is **not** the dataset point-light response. Canonical affine/halfspace parameters are decoded into float64 display meshes; WebGL itself draws those meshes in display precision. Neither Qhull mesh construction, edge merging, visual clipping nor GPU surface shading supplies scientific labels. The original lossless geometry, visibility oracle and acceptance contracts are untouched. Unsupported geometry fails explicitly rather than substituting voxel geometry.

The public release hides true-source positions. Without an explicit private configuration, `source_position` remains `null` and no source is guessed. For authorized local inspection with the original campaign still available, run:

```bash
./scripts/browser.sh --private-root /path/to/original-campaign
```

Alternatively set `RIND3D_PRIVATE_ROOT`, or save only the path in the Git-ignored project file `.rind3d-viewer.local.json`: `{"private_root":"/path/to/original-campaign"}`. Then the ordinary `./scripts/browser.sh` reuses that setting. `--private-root ""` explicitly disables it. This option permits only a loopback bind address.

The source is displayed only after checking the original sealed scene identity, private record checksums, exact 24-byte float64 source, geometry/source identity and matching SHA256 hashes of the installed geometry, response, solid and window records. **原始记录已匹配** means these records match; it does not claim a new physical verification was performed by the viewer. Missing or mismatched records leave the source hidden. A gold marker, label and dotted floor projection locate the source; its panel gives unit-world XYZ and **定位源** centers the camera on it. The marker remains visible through the response volume so it can be found reliably.

No candidates, private witnesses, oracle records or teacher traces are served. Source-bearing API responses are not cached by the browser, private paths are not served as assets, and actual source positions must not be committed to code, test fixtures, public datasets or model inputs. The viewer is privileged scene inspection, separate from the local-only student interface. Original campaign files and public scientific files remain read-only.

## Resource use and checks

Planes read only intersecting bricks. A volume preview scans all 4096 packed bricks in each field while allocating at most a 256³ uint8 preview, never a 1024³ bool tensor. The server retains only two compressed volume previews; reader, geometry, PNG, source and window-count caches are bounded too. Shared file readers are locked. The client admits one volume request at a time and replaces pending requests with the latest selection. Scene changes disable stale controls; asynchronous responses are checked against the current scene/selection before display. Old Three.js buffers and 3D textures are disposed. A failed request remains an error, not a physical class.

Tests independently compare all axis directions at voxel/brick/world boundaries, check PNG orientation, all preview sizes and solid separation, compare stored preview labels against reader point queries, exercise read-only HTTP/error routes, reject mismatched private-source records and confirm that native physics/PyTorch imports are unnecessary. Real-data tests require `RIND3D_DATA_ROOT`; setting `RIND3D_PRIVATE_ROOT` additionally checks matched original records from all nine families without serializing coordinates:

```bash
RIND3D_DATA_ROOT="$PWD/data/RIND3D-v0.1" uv run --extra dev --extra viewer pytest -q tests/test_browser.py
```

Tested locally on Linux with Python 3.12 and an actual WebGL browser. This is not a claim of universal browser/platform support. If WebGL 2 is unavailable, exact 2D slices remain usable and the page reports the missing 3D capability.
