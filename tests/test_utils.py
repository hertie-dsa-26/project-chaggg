import pytest
from scripts.utils import normalise_crime_type


class TestNormaliseCrimeType:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Real Chicago crime primary_type values
            ("MOTOR VEHICLE THEFT", "motor_vehicle_theft"),
            ("THEFT", "theft"),
            ("BATTERY", "battery"),
            ("CRIMINAL DAMAGE", "criminal_damage"),
            ("DECEPTIVE PRACTICE", "deceptive_practice"),
            ("OFFENSE INVOLVING CHILDREN", "offense_involving_children"),
            ("CRIM SEXUAL ASSAULT", "crim_sexual_assault"),
            ("NON-CRIMINAL (SUBJECT SPECIFIED)", "non_criminal_subject_specified"),
            ("LIQUOR LAW VIOLATION", "liquor_law_violation"),
            ("CONCEALED CARRY LICENSE VIOLATION", "concealed_carry_license_violation"),
        ],
    )
    def test_real_crime_types(self, raw, expected):
        assert normalise_crime_type(raw) == expected

    def test_collapses_internal_whitespace(self):
        assert normalise_crime_type("CRIM  SEXUAL   ASSAULT") == "crim_sexual_assault"

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalise_crime_type("  THEFT  ") == "theft"

    def test_strips_leading_and_trailing_punctuation(self):
        assert normalise_crime_type("--THEFT--") == "theft"

    def test_already_normalized_passes_through(self):
        assert normalise_crime_type("motor_vehicle_theft") == "motor_vehicle_theft"

    def test_mixed_case(self):
        assert normalise_crime_type("Motor Vehicle Theft") == "motor_vehicle_theft"

    def test_digits_preserved(self):
        assert normalise_crime_type("ASSAULT 2") == "assault_2"

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            normalise_crime_type(None)

    def test_nan_raises_type_error(self):
        import math
        with pytest.raises(TypeError):
            normalise_crime_type(math.nan)

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            normalise_crime_type("")

    def test_only_punctuation_raises_value_error(self):
        with pytest.raises(ValueError):
            normalise_crime_type("!!! ---")

    def test_only_whitespace_raises_value_error(self):
        with pytest.raises(ValueError):
            normalise_crime_type("   ")