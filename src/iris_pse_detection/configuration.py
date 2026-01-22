# Copyright (c) 2026 iris_pse_detection contributors
# SPDX-License-Identifier: MIT
#
# Based on IRIS by Electronic Arts Inc.
# https://github.com/electronicarts/IRIS

"""Configuration for IRIS analysis."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np


# Default sRGB lookup table values (256 entries for 0-255 input)
DEFAULT_SRGB_VALUES = [
    0, 0.0003035269835488375, 0.000607053967097675, 0.0009105809506465125,
    0.00121410793419535, 0.0015176349177441874, 0.001821161901293025, 0.0021246888848418626,
    0.0024282158683907, 0.0027317428519395373, 0.003035269835488375, 0.003346535763899161,
    0.003676507324047436, 0.004024717018496307, 0.004391442037410293, 0.004776953480693729,
    0.005181516702338386, 0.005605391624202723, 0.006048833022857054, 0.006512090792594475,
    0.006995410187265387, 0.007499032043226175, 0.008023192985384994, 0.008568125618069307,
    0.009134058702220787, 0.00972121732023785, 0.010329823029626936, 0.010960094006488246,
    0.011612245179743885, 0.012286488356915872, 0.012983032342173012, 0.013702083047289686,
    0.014443843596092545, 0.01520851442291271, 0.01599629336550963, 0.016807375752887384,
    0.017641954488384078, 0.018500220128379697, 0.019382360956935723, 0.0202885630566524,
    0.021219010376003555, 0.022173884793387385, 0.02315336617811041, 0.024157632448504756,
    0.02518685962736163, 0.026241221894849898, 0.027320891639074894, 0.028426039504420793,
    0.0295568344378088, 0.030713443732993635, 0.03189603307301153, 0.033104766570885055,
    0.03433980680868217, 0.03560131487502034, 0.03688945040110004, 0.0382043715953465,
    0.03954623527673284, 0.04091519690685319, 0.042311410620809675, 0.043735029256973465,
    0.04518620438567554, 0.046665086336880095, 0.04817182422688942, 0.04970656598412723,
    0.05126945837404324, 0.052860647023180246, 0.05448027644244237, 0.05612849004960009,
    0.05780543019106723, 0.0595112381629812, 0.06124605423161761, 0.06301001765316767,
    0.06480326669290577, 0.06662593864377289, 0.06847816984440017, 0.07036009569659588,
    0.07227185068231748, 0.07421356838014963, 0.07618538148130785, 0.07818742180518633,
    0.08021982031446832, 0.0822827071298148, 0.08437621154414882, 0.08650046203654976,
    0.08865558628577294, 0.09084171118340768, 0.09305896284668745, 0.0953074666309647,
    0.09758734714186246, 0.09989872824711389, 0.10224173308810132, 0.10461648409110419,
    0.10702310297826761, 0.10946171077829933, 0.1119324278369056, 0.11443537382697373,
    0.11697066775851084, 0.11953842798834562, 0.12213877222960187, 0.12477181756095049,
    0.12743768043564743, 0.1301364766903643, 0.13286832155381798, 0.13563332965520566,
    0.13843161503245183, 0.14126329114027164, 0.14412847085805777, 0.14702726649759498,
    0.14995978981060856, 0.15292615199615017, 0.1559264637078274, 0.1589608350608804,
    0.162029375639111, 0.1651321945016676, 0.16826940018969075, 0.1714411007328226,
    0.17464740365558504, 0.17788841598362912, 0.18116424424986022, 0.184474994500441,
    0.18782077230067787, 0.19120168274079138, 0.1946178304415758, 0.19806931955994886,
    0.20155625379439707, 0.20507873639031693, 0.20863687014525575, 0.21223075741405523,
    0.21586050011389926, 0.2195261997292692, 0.2232279573168085, 0.22696587351009836,
    0.23074004852434915, 0.23455058216100522, 0.238397573812271, 0.24228112246555486,
    0.24620132670783548, 0.25015828472995344, 0.25415209433082675, 0.2581828529215958,
    0.26225065752969623, 0.26635560480286247, 0.2704977910130658, 0.27467731206038465,
    0.2788942634768104, 0.2831487404299921, 0.2874408377269175, 0.29177064981753587,
    0.2961382707983211, 0.3005437944157765, 0.3049873140698863, 0.30946892281750854,
    0.31398871337571754, 0.31854677812509186, 0.32314320911295075, 0.3277780980565422,
    0.33245153634617935, 0.33716361504833037, 0.3419144249086609, 0.3467040563550296,
    0.35153259950043936, 0.3564001441459435, 0.3613067797835095, 0.3662525955988395,
    0.3712376804741491, 0.3762621229909065, 0.38132601143253014, 0.386429433787049,
    0.39157247774972326, 0.39675523072562685, 0.4019777798321958, 0.4072402119017367,
    0.41254261348390375, 0.4178850708481375, 0.4232676699860717, 0.4286904966139066,
    0.43415363617474895, 0.4396571738409188, 0.44520119451622786, 0.45078578283822346,
    0.45641102318040466, 0.4620769996544071, 0.467783796112159, 0.47353149614800955,
    0.4793201831008268, 0.4851499400560704, 0.4910208498478356, 0.4969329950608704,
    0.5028864580325687, 0.5088813208549338, 0.5149176653765214, 0.5209955732043543,
    0.5271151257058131, 0.5332764040105052, 0.5394794890121072, 0.5457244613701866,
    0.5520114015120001, 0.5583403896342679, 0.5647115057049292, 0.5711248294648731,
    0.5775804404296506, 0.5840784178911641, 0.5906188409193369, 0.5972017883637634,
    0.6038273388553378, 0.6104955708078648, 0.6172065624196511, 0.6239603916750761,
    0.6307571363461468, 0.6375968739940326, 0.6444796819705821, 0.6514056374198242,
    0.6583748172794485, 0.665387298282272, 0.6724431569576875, 0.6795424696330938,
    0.6866853124353135, 0.6938717612919899, 0.7011018919329731, 0.7083757798916868,
    0.7156935005064807, 0.7230551289219693, 0.7304607400903537, 0.7379104087727308,
    0.7454042095403874, 0.7529422167760779, 0.7605245046752924, 0.768151147247507,
    0.7758222183174236, 0.7835377915261935, 0.7912979403326302, 0.799102738014409,
    0.8069522576692516, 0.8148465722161012, 0.8227857543962835, 0.8307698767746546,
    0.83879901174074, 0.846873231509858, 0.8549926081242338, 0.8631572134541023,
    0.8713671191987972, 0.8796223968878317, 0.8879231178819663, 0.8962693533742664,
    0.9046611743911496, 0.9130986517934192, 0.9215818562772946, 0.9301108583754237,
    0.938685728457888, 0.9473065367331999, 0.9559733532492861, 0.9646862478944651,
    0.9734452903984125, 0.9822505503331171, 0.9911020971138298, 1
]


@dataclass
class FlashParams:
    """Parameters for flash detection."""
    flash_threshold: float
    area_proportion: float
    dark_threshold: float


@dataclass
class TransitionTrackerParams:
    """Parameters for transition tracking."""
    max_transitions: int = 6
    min_transitions: int = 4
    extended_fail_seconds: int = 4
    extended_fail_window: int = 5
    warning_transitions: int = 4


@dataclass
class PatternDetectionParams:
    """Parameters for pattern detection."""
    min_stripes: int = 6
    dark_luminance_threshold: float = 0.8
    time_threshold: float = 0.5
    area_proportion: float = 0.25


@dataclass
class Configuration:
    """Configuration for IRIS video analysis."""

    # Luminance flash parameters
    luminance_flash_threshold: float = 0.1
    luminance_dark_threshold: float = 0.8

    # Red saturation parameters
    red_flash_threshold: float = 20.0
    red_dark_threshold: float = 321.0

    # Area proportion for flash detection
    area_proportion: float = 0.25

    # Transition tracker parameters
    max_transitions: int = 6
    min_transitions: int = 4
    extended_fail_seconds: int = 4
    extended_fail_window: int = 5
    warning_transitions: int = 4

    # Pattern detection parameters
    pattern_detection_enabled: bool = False
    pattern_min_stripes: int = 6
    pattern_time_threshold: float = 0.5
    pattern_area_proportion: float = 0.25
    pattern_dark_luminance_threshold: float = 0.8

    # Video analyser settings
    frame_resize_enabled: bool = False
    frame_resize_proportion: float = 0.2

    # sRGB lookup table
    srgb_values: np.ndarray = field(default_factory=lambda: np.array(DEFAULT_SRGB_VALUES, dtype=np.float32))

    # Output paths
    results_path: str = "Results/"

    # Analysis mode
    analyse_by_time: bool = False

    @classmethod
    def from_json(cls, path: str) -> "Configuration":
        """Load configuration from appsettings.json file."""
        config_path = Path(path) / "appsettings.json"

        if not config_path.exists():
            # Return default configuration
            return cls()

        with open(config_path, "r") as f:
            # Remove comments from JSON (not standard but used in the original)
            content = f.read()
            # Simple comment removal (single-line comments)
            lines = []
            for line in content.split("\n"):
                comment_idx = line.find("//")
                if comment_idx >= 0:
                    line = line[:comment_idx]
                lines.append(line)
            content = "\n".join(lines)
            data = json.loads(content)

        config = cls()

        # Luminance settings
        if "Luminance" in data:
            config.luminance_flash_threshold = data["Luminance"].get(
                "RelativeLuminanceFlashThreshold", config.luminance_flash_threshold
            )
            config.luminance_dark_threshold = data["Luminance"].get(
                "RelativeDarkLuminanceThreshold", config.luminance_dark_threshold
            )

        # Red saturation settings
        if "RedSaturation" in data:
            config.red_flash_threshold = data["RedSaturation"].get(
                "FlashThreshold", config.red_flash_threshold
            )
            config.red_dark_threshold = data["RedSaturation"].get(
                "RedDarkThreshold", config.red_dark_threshold
            )

        # Flash detection settings
        if "FlashDetection" in data:
            config.area_proportion = data["FlashDetection"].get(
                "AreaProportion", config.area_proportion
            )
            if "sRGBValues" in data["FlashDetection"]:
                config.srgb_values = np.array(
                    data["FlashDetection"]["sRGBValues"], dtype=np.float32
                )

        # Transition tracker settings
        if "TransitionTracker" in data:
            tt = data["TransitionTracker"]
            config.max_transitions = tt.get("MaxTransitions", config.max_transitions)
            config.min_transitions = tt.get("MinTransitions", config.min_transitions)
            config.extended_fail_seconds = tt.get("ExtendedFailSeconds", config.extended_fail_seconds)
            config.extended_fail_window = tt.get("ExtendedFailWindow", config.extended_fail_window)
            config.warning_transitions = tt.get("WarningTransitions", config.warning_transitions)

        # Pattern detection settings
        if "PatternDetection" in data:
            pd = data["PatternDetection"]
            config.pattern_min_stripes = pd.get("MinStripes", config.pattern_min_stripes)
            config.pattern_time_threshold = pd.get("TimeThreshold", config.pattern_time_threshold)
            config.pattern_area_proportion = pd.get("AreaProportion", config.pattern_area_proportion)
            config.pattern_dark_luminance_threshold = pd.get(
                "RelativeDarkLuminanceThreshold", config.pattern_dark_luminance_threshold
            )

        # Video analyser settings
        if "VideoAnalyser" in data:
            va = data["VideoAnalyser"]
            config.pattern_detection_enabled = va.get(
                "PatternDetectionEnabled", config.pattern_detection_enabled
            )
            config.frame_resize_enabled = va.get(
                "FrameResizeEnabled", config.frame_resize_enabled
            )
            config.frame_resize_proportion = va.get(
                "ResizeFrameProportion", config.frame_resize_proportion
            )

        # Results path
        config.results_path = data.get("ResultsPath", config.results_path)

        return config

    def get_luminance_params(self) -> FlashParams:
        """Get luminance flash parameters."""
        return FlashParams(
            flash_threshold=self.luminance_flash_threshold,
            area_proportion=self.area_proportion,
            dark_threshold=self.luminance_dark_threshold,
        )

    def get_red_saturation_params(self) -> FlashParams:
        """Get red saturation flash parameters."""
        return FlashParams(
            flash_threshold=self.red_flash_threshold,
            area_proportion=self.area_proportion,
            dark_threshold=self.red_dark_threshold,
        )

    def get_transition_tracker_params(self) -> TransitionTrackerParams:
        """Get transition tracker parameters."""
        return TransitionTrackerParams(
            max_transitions=self.max_transitions,
            min_transitions=self.min_transitions,
            extended_fail_seconds=self.extended_fail_seconds,
            extended_fail_window=self.extended_fail_window,
            warning_transitions=self.warning_transitions,
        )

    def get_pattern_detection_params(self) -> PatternDetectionParams:
        """Get pattern detection parameters."""
        return PatternDetectionParams(
            min_stripes=self.pattern_min_stripes,
            dark_luminance_threshold=self.pattern_dark_luminance_threshold,
            time_threshold=self.pattern_time_threshold,
            area_proportion=self.pattern_area_proportion,
        )
