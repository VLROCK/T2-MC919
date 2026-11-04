from types import SimpleNamespace
from unittest.mock import patch

from src.pipeline import colmap_option


def test_legacy_colmap_option_name_is_selected():
    help_result = SimpleNamespace(
        stdout='--SiftExtraction.max_image_size arg (=3200)\n', stderr=''
    )
    with patch('src.pipeline.subprocess.run', return_value=help_result):
        selected = colmap_option('colmap', 'feature_extractor', (
            '--FeatureExtraction.max_image_size', '--SiftExtraction.max_image_size'))
    assert selected == '--SiftExtraction.max_image_size'
