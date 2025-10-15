def test_imports():
    from imu_exercise_classifier.model import CNN_LSTM
    from imu_exercise_classifier.datasets import IMU_FEATURES
    assert len(IMU_FEATURES) == 13
    m = CNN_LSTM(num_features=13, num_classes=3)
    assert m is not None
