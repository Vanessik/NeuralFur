"""Shared utilities for preprocessing scripts."""

import os
import sys
import argparse

# Ensure src/ is importable from any preprocessing script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_script_dir, "../src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from animal_config import scenes, postfixx


def create_parser(description=""):
    """Create an argument parser with the common --animal and --root_path flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--animal", "-a", required=True,
        help="Animal name (e.g., panda, fox, cat, whiteTiger, beagle_dog, synth_tiger)",
    )
    parser.add_argument(
        "--root_path", required=True,
        help="Root path to Artemis data directory",
    )
    return parser


def get_scene_type(animal):
    """Get the motion sequence name for a given animal."""
    if animal not in scenes:
        raise ValueError(f"Unknown animal '{animal}'. Choose from: {list(scenes.keys())}")
    return scenes[animal]


def get_data_path(root_path, animal):
    """Build the standard GH data directory path: <root>/<animal>_<postfix>_GH/<scene>/"""
    scene_type = get_scene_type(animal)
    suf = postfixx[animal]
    return os.path.join(root_path, f"{animal}_{suf}_GH", scene_type)
