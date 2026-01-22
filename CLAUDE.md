# CLAUDE.md

このファイルは Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイドです。

## プロジェクト概要

IRIS-PSE-Detection (`pip install iris-pse-detection`) は Electronic Arts の [IRIS](https://github.com/electronicarts/IRIS) (C++) を Python に移植したものです。光過敏性てんかんを引き起こす可能性のあるフラッシュパターンを動画から検出します。

- PyPI: https://pypi.org/project/iris-pse-detection/
- GitHub: https://github.com/tokoroten/iris-pse-detection

## コマンド

```bash
# 依存関係のインストール
uv sync

# 動画解析の実行
uv run iris <video_file>

# テストの実行
uv run pytest

# リンターの実行
uv run ruff check src/

# パッケージのビルド
uv build
```

## アーキテクチャ

```
src/iris_pse_detection/
├── __init__.py           # パッケージエクスポート
├── cli.py                # コマンドラインインターフェース
├── configuration.py      # 設定パラメータ
├── video_analyser.py     # メイン解析オーケストレーター
├── flash_detection.py    # フラッシュ検出の統合
├── flash.py              # 輝度・赤色フラッシュ計算
├── transition_tracker.py # 遷移カウントとスライディングウィンドウ
├── frame_data.py         # フレームデータ構造
├── frame_rgb_converter.py # sRGB変換
├── pattern_detection.py  # パターン検出（オプション）
└── result.py             # 解析結果構造
```

## 重要な実装詳細

### FpsFrameManager パターン
C++ の `FpsFrameManager` はスライディングウィンドウを管理し、`framesToRemove` として 0 または 1 を返します。Python でも同じパターンを使用:

```python
if self._current_frames >= self._max_frames:
    self._frames_to_remove = 1
else:
    self._frames_to_remove = 0
    self._current_frames += 1
```

### フレーム0のスキップ
`flash_detection.py` で `frame_pos != 0` のチェックが必要。フレーム0には比較対象の前フレームがないため。

### 浮動小数点精度
C++ と Python で微小な差異が発生する場合があります（境界値での符号反転など）。これは許容範囲内です。

## C++ 版との比較

オリジナル C++ 版: https://github.com/electronicarts/IRIS

## ライセンス

MIT License（オリジナル IRIS は BSD 3-Clause）
