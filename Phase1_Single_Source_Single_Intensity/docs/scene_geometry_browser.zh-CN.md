# Phase2 历史结果浏览器：全场场景与原始障碍物

在已有教师/学生结果界面中加载一个测试视窗后，“原始全场场景与障碍物”展示：

- 1024×1024原始加权叠加响应，蓝色障碍、青色当前视窗；生成源由原来的“显示生成源”开关控制。
- 直接按原始连续参数绘制的障碍几何。旋转矩形、椭圆、三角形和多边形均保留实际形状，#编号对应原始部件索引。
- 可展开的障碍物表与完整JSON，包含type_id、group_id、原始float64参数、size或顶点；角度单位为弧度，世界x向右、y向下。
- 当前场景JSON、全部已恢复测试样本对应的场景集合JSON，以及当前全场图PNG下载。

集合导出只覆盖浏览器当前样本，按scene_id去重，不代表整个RIND发布。本地恢复包有100个视窗、88个不同scene_id。复合障碍的多个几何部件可以共享group_id，部件数与逻辑障碍数分别显示。

位置概率图可选叠加障碍物轮廓。完整场景及生成源只供展示/导出，不增加学生输入，也不修改教师或学生概率。

## 运行与导出

从G2-02根目录启动现有恢复包：

```bash
RIND_VISUALIZATION_BUNDLE="$PWD/Phase1_Single_Source_Single_Intensity/phase2_visualization_100_recovered" \
  ./Phase2_Multi_Source/view_layer_results.sh --device cpu
```

打开 `http://127.0.0.1:8768/?index=10` 可直接加载指定样本；index是浏览器样本编号，不是scene_id。

本地API：

| 地址 | 返回 |
|---|---|
| `/api/scene?index=10` | 当前样本的全场响应、障碍栅格、原始场景几何和窗口信息 |
| `/api/geometry?index=10&download=1` | 当前场景原始JSON |
| `/api/geometry?scope=all&download=1` | 当前浏览器全部样本对应的去重场景集合JSON |

原始JSON包含生成源参数，并明确标记 `sources_evaluation_only`。场景响应从已保存的逐源可见性乘真实强度求和，不使用多源OR响应、不重新进行候选位置渲染。栅格按场景缓存，集合几何导出不解码所有全场响应。

本地导出副本位于Phase1的 `outputs/reports/scene_geometry/`，由Git忽略，不与代码一起上传。新增展示已核对样本#0、#10、#34的全场裁剪与原局部响应/障碍掩码一致，并核对加入场景面板前后的教师、学生概率逐点不变。
