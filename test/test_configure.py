import importlib
import os
import shutil
import unittest
from pathlib import Path

import gridemissions


class TestEnvironmentVariables(unittest.TestCase):
    """Test the interaction between environment variables and the configuration."""

    test_dir_path = Path.home().joinpath(".config/gridemissions_test")
    # A map of test environment variables to their value.
    test_env_variables = {"GRIDEMISSIONS_CONFIG_DIR_PATH": str(test_dir_path)}
    loaded_configure = gridemissions.configure

    @classmethod
    def reload_configure(cls):
        """Reload the configure file."""
        cls.loaded_configure = importlib.reload(cls.loaded_configure)

    def setUp(self) -> None:
        """Called before each test."""
        super().setUp()
        self.reload_configure()

    def test_default_variable(self):
        """Test the default behaviour of an environment variable."""
        expected_path = Path.home().joinpath(".config/gridemissions")

        # The initialized config directory should be the same as the default directory.
        assert str(self.loaded_configure.CONFIG_DIR_PATH) == str(expected_path)

    def test_declared_variable(self):
        """Test using a declared environment variable, interpreted as a path."""
        # Get the test environment, and set it as default.
        test_var = "GRIDEMISSIONS_CONFIG_DIR_PATH"
        os.environ[test_var] = self.test_env_variables.get(test_var)

        # Reload the configuration to propagate the new environment variable.
        self.reload_configure()

        # The initialized config directory should be the same as the new test directory.
        assert str(self.loaded_configure.CONFIG_DIR_PATH) == str(self.test_dir_path)

    def tearDown(self) -> None:
        """Called after each test."""
        super().tearDown()

        # Remove any set environment variables.
        for env in self.test_env_variables:
            try:
                del os.environ[env]
            except KeyError:
                pass

        # Remove the test directory.
        try:
            shutil.rmtree(self.test_dir_path)
        except FileNotFoundError:
            pass
