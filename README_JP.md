# IRIS-Py

[IRIS](https://github.com/electronicarts/IRIS) の Python 移植版 - Electronic Arts が開発した光過敏性てんかんリスク検出ライブラリです。

IRIS は、W3C WCAG および ISO 9241-391 のガイドラインに基づき、光過敏性てんかんを持つ人々に発作を引き起こす可能性のあるフラッシュパターンを動画コンテンツから検出・解析します。

## インストール

```bash
pip install iris-py
```

または開発用:

```bash
git clone https://github.com/electronicarts/IRIS
cd iris_py
uv sync
```

## 使い方

### コマンドライン

```bash
# pip インストール後
iris video.mp4

# または uv 経由
uv run iris video.mp4
```

### Python API

```python
from iris_py import VideoAnalyser, Configuration

config = Configuration()
analyser = VideoAnalyser(config)
result = analyser.analyse_video("video.mp4")
print(result.overall_result)
```

## 機能

- **輝度フラッシュ検出** - 明暗の急激な変化を検出
- **赤色飽和フラッシュ検出** - 危険な赤色フラッシュを検出
- **1秒スライディングウィンドウ** - 遷移回数の追跡
- **拡張失敗検出** - 4秒以上の連続違反を検出
- **パターン検出** (オプション) - 縞模様などの危険なパターンを検出

## 技術仕様

- W3C WCAG 2.1 準拠
- ISO 9241-391 準拠
- Python 3.10 以上対応

## 依存関係

- numpy >= 1.22
- opencv-python >= 4.6
- scipy >= 1.8

## 精度に関する注意

C++ と Python/NumPy の浮動小数点演算の精度差により、オリジナルの IRIS と結果が微妙に異なる場合があります。この差異は非常に小さく、検出閾値の境界付近でのみ発生します。

## 謝辞

このプロジェクトは、Electronic Arts Inc. が BSD 3-Clause License で公開した [IRIS](https://github.com/electronicarts/IRIS) の Python 移植版です。

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照してください。
