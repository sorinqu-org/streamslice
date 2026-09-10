import unittest
from pathlib import Path

import yaml

from streamslice.templates import (
    Template,
    TemplateError,
    builtin_templates_dir,
    default_template_name,
    list_templates,
    load_template,
    validate_template_mapping,
)

ROOT = Path(__file__).resolve().parents[1]


def _base_config(templates_dir: str | None = None) -> dict:
    return {"render": {"templates_dir": templates_dir}}


def _valid_mapping() -> dict:
    return {
        "name": "unit-test-template",
        "description": "test",
        "canvas": {"width": 1080, "height": 1920, "fps": 60},
        "layouts": {
            "split": {
                "bands": [
                    {"source": "webcam", "height": 0.4, "fit": "cover"},
                    {"source": "gameplay", "height": 0.6, "fit": "cover"},
                ]
            }
        },
        "subtitles": {},
        "motion": {},
    }


class BuiltinTemplateTests(unittest.TestCase):
    def test_all_builtin_templates_load_and_validate(self) -> None:
        for template in list_templates(_base_config()):
            errors = [
                msg
                for msg in validate_template_mapping(template.to_dict())
                if not msg.startswith("Warning: ")
            ]
            self.assertEqual(errors, [], f"{template.name}: {errors}")

    def test_builtin_templates_dir_contains_four_yaml_files(self) -> None:
        yaml_files = sorted(builtin_templates_dir().glob("*.yaml"))
        names = {p.stem for p in yaml_files}
        self.assertEqual(
            names, {"classic-split", "face-focus", "gameplay-focus", "reaction-cam"}
        )

    def test_classic_split_matches_hardcoded_geometry(self) -> None:
        template = load_template("classic-split", _base_config())
        split = template.layout("split")
        self.assertEqual(len(split.bands), 2)
        webcam, gameplay = split.bands
        self.assertEqual(webcam.source, "webcam")
        self.assertAlmostEqual(webcam.height, 0.38)
        self.assertEqual(webcam.divider_px, 8)
        self.assertEqual(webcam.divider_color, "#000000")
        self.assertEqual(gameplay.source, "gameplay")
        self.assertAlmostEqual(gameplay.height, 0.62)
        self.assertEqual(gameplay.divider_px, 0)
        self.assertEqual(template.width, 1080)
        self.assertEqual(template.height, 1920)
        self.assertEqual(template.fps, 60)


class ValidatorTests(unittest.TestCase):
    def test_valid_mapping_has_no_errors(self) -> None:
        self.assertEqual(validate_template_mapping(_valid_mapping()), [])

    def test_height_sum_mismatch_is_reported(self) -> None:
        data = _valid_mapping()
        data["layouts"]["split"]["bands"][0]["height"] = 0.5
        errors = validate_template_mapping(data)
        self.assertTrue(any("sum to 1.0" in e for e in errors))
        self.assertTrue(any("1.1" in e for e in errors))

    def test_unknown_source_is_reported(self) -> None:
        data = _valid_mapping()
        data["layouts"]["split"]["bands"][0]["source"] = "drone"
        errors = validate_template_mapping(data)
        self.assertTrue(any("source" in e for e in errors))

    def test_unknown_fit_is_reported(self) -> None:
        data = _valid_mapping()
        data["layouts"]["split"]["bands"][0]["fit"] = "stretch"
        errors = validate_template_mapping(data)
        self.assertTrue(any("fit" in e for e in errors))

    def test_zoom_below_one_is_reported(self) -> None:
        data = _valid_mapping()
        data["layouts"]["split"]["bands"][0]["zoom"] = 0.9
        errors = validate_template_mapping(data)
        self.assertTrue(any("zoom" in e for e in errors))

    def test_bad_divider_color_is_reported(self) -> None:
        data = _valid_mapping()
        data["layouts"]["split"]["bands"][0]["divider_color"] = "black"
        errors = validate_template_mapping(data)
        self.assertTrue(any("divider_color" in e for e in errors))

    def test_missing_layouts_is_reported(self) -> None:
        data = _valid_mapping()
        del data["layouts"]
        errors = validate_template_mapping(data)
        self.assertTrue(any("layouts" in e for e in errors))

    def test_missing_name_is_reported(self) -> None:
        data = _valid_mapping()
        del data["name"]
        errors = validate_template_mapping(data)
        self.assertTrue(any("name" in e for e in errors))

    def test_unknown_keys_are_warnings_not_errors(self) -> None:
        data = _valid_mapping()
        data["future_field"] = "something"
        data["layouts"]["split"]["bands"][0]["future_band_field"] = 1
        messages = validate_template_mapping(data)
        warnings = [m for m in messages if m.startswith("Warning: ")]
        errors = [m for m in messages if not m.startswith("Warning: ")]
        self.assertEqual(errors, [])
        self.assertTrue(any("future_field" in w for w in warnings))
        self.assertTrue(any("future_band_field" in w for w in warnings))

    def test_missing_primary_layout_is_reported(self) -> None:
        data = _valid_mapping()
        data["layouts"] = {
            "custom": {
                "bands": [{"source": "webcam", "height": 1.0}],
            }
        }
        errors = validate_template_mapping(data)
        self.assertTrue(any("split" in e and "webcam_full" in e for e in errors))


class TemplatePriorityTests(unittest.TestCase):
    def test_custom_template_overrides_builtin(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data = _valid_mapping()
            data["name"] = "classic-split"
            data["description"] = "overridden"
            (tmp_path / "classic-split.yaml").write_text(
                yaml.safe_dump(data), encoding="utf-8"
            )
            config = _base_config(str(tmp_path))
            template = load_template("classic-split", config)
            self.assertEqual(template.description, "overridden")
            self.assertEqual(template.source_path, tmp_path / "classic-split.yaml")

    def test_load_unknown_template_raises_with_available_names(self) -> None:
        with self.assertRaises(TemplateError) as ctx:
            load_template("does-not-exist", _base_config())
        message = str(ctx.exception)
        self.assertIn("does-not-exist", message)
        self.assertIn("classic-split", message)

    def test_default_template_name(self) -> None:
        self.assertEqual(default_template_name({}), "classic-split")
        self.assertEqual(
            default_template_name({"render": {"template": "face-focus"}}), "face-focus"
        )


class LayoutFallbackTests(unittest.TestCase):
    def test_layout_falls_back_to_split(self) -> None:
        template = load_template("classic-split", _base_config())
        self.assertIs(template.layout("nonexistent"), template.layout("split"))

    def test_layout_falls_back_to_first_available_when_no_split(self) -> None:
        from types import MappingProxyType

        from streamslice.templates import Band, LayoutSpec

        only_layout = LayoutSpec(
            name="webcam_full",
            bands=(Band(source="webcam", height=1.0),),
        )
        template = Template(
            name="no-split",
            layouts=MappingProxyType({"webcam_full": only_layout}),
        )
        self.assertIs(template.layout("split"), only_layout)


class RoundTripTests(unittest.TestCase):
    def test_to_dict_round_trips_through_validation(self) -> None:
        template = load_template("classic-split", _base_config())
        errors = [
            msg
            for msg in validate_template_mapping(template.to_dict())
            if not msg.startswith("Warning: ")
        ]
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
