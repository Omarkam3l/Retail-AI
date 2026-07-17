def test_package_imports():
    # Base configuration & logging configurations
    from src import config
    from src import common

    # Video Pipeline interfaces
    from src import detection
    from src import tracking
    from src import association
    from src import behavior
    from src import risk
    from src import alerts
    from src import inference
    from src import storage
    from src import api
    from src import dashboard

    # Verify that imports resolved correctly
    assert config is not None
    assert common is not None
    assert detection is not None
    assert tracking is not None
    assert association is not None
    assert behavior is not None
    assert risk is not None
    assert alerts is not None
    assert inference is not None
    assert storage is not None
    assert api is not None
    assert dashboard is not None
