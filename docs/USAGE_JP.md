# IRIS-PSE-Detection 使用ガイド

## 基本的な使い方

### コマンドライン

```bash
# 基本的な解析
iris video.mp4

# JSON出力付き
iris video.mp4 --json

# パターン検出付き
iris video.mp4 --pattern-detection

# フレームリサイズ（処理高速化）
iris video.mp4 --resize 0.5
```

### Python API

```python
from iris_pse_detection import VideoAnalyser, Configuration, AnalysisResult

config = Configuration()
analyser = VideoAnalyser(config)
result = analyser.analyse_video("video.mp4")

if result.overall_result == AnalysisResult.Pass:
    print("動画は安全です")
elif result.overall_result == AnalysisResult.Fail:
    print("動画は不合格です")
```

---

## 結果の読み方

### 全体結果 (`AnalysisResult`)

| 値 | 意味 |
|---|------|
| `Pass` | 問題なし |
| `PassWithWarning` | 軽微な問題あり（許容範囲内） |
| `Fail` | 光過敏性チェック不合格 |
| `LuminanceFlashFailure` | 輝度フラッシュによる不合格 |
| `LuminanceExtendedFlashFailure` | 輝度フラッシュが4秒以上継続 |
| `RedFlashFailure` | 赤色飽和フラッシュによる不合格 |
| `RedExtendedFlashFailure` | 赤色フラッシュが4秒以上継続 |
| `PatternFailure` | 危険なパターンによる不合格 |

### フレームごとのフラッシュ結果 - `FlashResult`

`luminance_frame_result` と `red_frame_result` で使用されます。

| 値 | コード | 意味 |
|---|--------|------|
| `Pass` | 0 | フレームは安全 |
| `PassWithWarning` | 1 | 軽微な問題あり |
| `ExtendedFail` | 2 | 4秒以上のフラッシュ不合格 |
| `FlashFail` | 3 | このフレームでフラッシュ検出 |

### フレームごとのパターン結果 - `PatternResult`

`pattern_frame_result` で使用されます。

| 値 | コード | 意味 |
|---|--------|------|
| `Pass` | 0 | 危険なパターンなし |
| `Fail` | 1 | 危険なパターン検出 |

---

## 結果オブジェクトの構造

```python
result = analyser.analyse_video("video.mp4")

# 全体結果
result.overall_result        # AnalysisResult 列挙型
result.total_frames          # 解析したフレーム総数
result.video_len             # 動画の長さ（秒）
result.analysis_time         # 処理時間（ミリ秒）

# 輝度フラッシュの発生回数
result.total_luminance_incidents.flash_fail_frames      # フラッシュ不合格フレーム数
result.total_luminance_incidents.extended_fail_frames   # 拡張不合格フレーム数
result.total_luminance_incidents.pass_with_warning_frames  # 警告フレーム数
result.total_luminance_incidents.total_failed_frames    # 不合格フレーム総数

# 赤色フラッシュの発生回数（同じ構造）
result.total_red_incidents.flash_fail_frames
result.total_red_incidents.extended_fail_frames
result.total_red_incidents.pass_with_warning_frames
result.total_red_incidents.total_failed_frames

# パターン検出
result.pattern_fail_frames   # パターン不合格フレーム数
```

---

## 出力ファイル

### CSV出力 (`Results/video_FrameData.csv`)

常に生成されます。フレームごとの解析データを含みます。

| カラム | 説明 |
|--------|------|
| `Frame` | フレーム番号（0始まり） |
| `TimeStamp` | タイムスタンプ（HH:MM:SS.ffffff形式） |
| `AverageLuminance` | フレームの平均輝度（0.0-1.0） |
| `FlashAreaLuminance` | 輝度フラッシュが発生した画面領域の割合 |
| `AverageLuminanceDiff` | 前フレームからの輝度差 |
| `AverageLuminanceDiffAcc` | 累積輝度差 |
| `AverageRed` | 平均赤色飽和値 |
| `FlashAreaRed` | 赤色フラッシュが発生した画面領域の割合 |
| `AverageRedDiff` | 前フレームからの赤色差 |
| `AverageRedDiffAcc` | 累積赤色差 |
| `LuminanceTransitions` | 1秒間の輝度遷移回数 |
| `RedTransitions` | 1秒間の赤色遷移回数 |
| `LuminanceExtendedFailCount` | 連続拡張不合格カウント（輝度） |
| `RedExtendedFailCount` | 連続拡張不合格カウント（赤色） |
| `LuminanceFrameResult` | 輝度結果コード（0-3） |
| `RedFrameResult` | 赤色結果コード（0-3） |
| `PatternArea` | 危険なパターンが検出された画面領域の割合 |
| `PatternDetectedLines` | 検出されたストライプ線の数 |
| `PatternFrameResult` | パターン結果コード（0-1） |

### JSON出力 (`Results/video_Result.json`)

`--json` フラグまたは `output_json=True` で生成されます。

```json
{
  "VideoLen": 10.5,
  "AnalysisTime": 1234,
  "TotalFrames": 315,
  "OverallResult": "Pass",
  "Results": ["Pass", "Pass", "PassWithWarning", ...],
  "TotalLuminanceIncidents": {
    "ExtendedFailFrames": 0,
    "FlashFailFrames": 0,
    "PassWithWarningFrames": 5,
    "TotalFailedFrames": 0
  },
  "TotalRedIncidents": {
    "ExtendedFailFrames": 0,
    "FlashFailFrames": 0,
    "PassWithWarningFrames": 0,
    "TotalFailedFrames": 0
  },
  "PatternFailFrames": 0
}
```

---

## 検出閾値

デフォルト設定値：

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `luminance_flash_threshold` | 0.1 | 10%の輝度変化で検出 |
| `red_flash_threshold` | 20.0 | 赤色飽和差の閾値 |
| `area_proportion` | 0.25 | 画面の25%以上が影響を受ける必要 |
| `max_transitions` | 4 | 1秒あたりの最大許容遷移回数 |
| `extended_fail_seconds` | 4 | 拡張不合格となる秒数 |

### 閾値のカスタマイズ

```python
from iris_pse_detection import VideoAnalyser, Configuration

config = Configuration()
config.luminance_flash_threshold = 0.15  # より寛容に（15%）
config.area_proportion = 0.30            # 画面の30%を要求
config.pattern_detection_enabled = True  # パターン検出を有効化

analyser = VideoAnalyser(config)
result = analyser.analyse_video("video.mp4")
```

---

## 結果の解釈

### 安全な動画
```
Overall Result: Pass
Luminance Flash Failures: 0
Red Flash Failures: 0
```

### 警告付きで合格
```
Overall Result: PassWithWarning
Luminance Warnings: 12
```
一部のフレームに境界線上の問題がありますが、一般的には許容範囲内です。

### 不合格の動画
```
Overall Result: LuminanceFlashFailure
Luminance Flash Failures: 45
Luminance Extended Failures: 12
```
動画に危険なフラッシュシーケンスが含まれています。以下を検討してください：
- 明暗のコントラストを下げる
- 遷移を遅くする
- 再生前に警告を表示する

---

## 参考資料

- [W3C WCAG 2.1 - 3回のせん光、または閾値以下](https://www.w3.org/WAI/WCAG21/Understanding/three-flashes-or-below-threshold.html)
- [ISO 9241-391:2016 - 人間工学](https://www.iso.org/standard/56350.html)
- [オリジナル IRIS プロジェクト](https://github.com/electronicarts/IRIS)
