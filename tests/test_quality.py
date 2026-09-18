import numpy as np

from erit.quality import image_metrics, is_bad_image


def test_uniform_bright_image_is_kept():
    img = np.full((256, 256), 200, np.uint8)
    bad, tsm, std = is_bad_image(img)
    assert not bad and tsm == 0 and std == 0


def test_noisy_dark_image_is_rejected():
    rng = np.random.default_rng(0)
    img = rng.choice([0, 255], size=(256, 256), p=[0.6, 0.4]).astype(np.uint8)
    assert is_bad_image(img)[0]


def test_metrics_definition():
    img = np.zeros((10, 10), np.uint8)
    img[:5] = 255
    tsm, std = image_metrics(img)
    assert tsm == 50 and std == 127.5
