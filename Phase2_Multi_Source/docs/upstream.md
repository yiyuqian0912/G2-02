# Yushu 多源基线来源

- Repository: https://github.com/yiyuqian0912/G2-02
- Branch: `yushu/phase1-multi-source`
- Commit: `444e042d8e148eb8ce8704d5b578856d25183475`
- Original directory: `Phase1_Multi_Source/`

导入已实现的数据安装/读取、物理重渲染、联合搜索与教师、数据浏览、诊断、测试及 vendor 快照。只抽取这个目录，没有 merge 整个分支，也没有接入其对单源目录的修改。

命名调整：目录→`Phase2_Multi_Source`，包→`rind_phase2`，dataset 类→`Phase2Dataset`，默认数据路径→`data/raw/phase2-multi-source`，配置→`configs/phase2.json`；原始数据浏览默认端口改为 8768，避免与单源结果浏览器 8767 冲突。保留 `RIND_MULTI_DATA_ROOT` 和 `rind-multi-*` CLI。

保留上游联合教师的物理/采样语义。额外补齐与 Phase I 对应的学生/训练/评价模块骨架、配置检查和候选接口校验。未复制或训练单源模型。

上游 verification.md 描述的是 Yushu 的历史实测，不等于本次对完整发布包重新验证。本次验证范围是仓库内合成小场景测试；未下载约 7 GB 的完整原生数据，也未开展多源学生训练。
