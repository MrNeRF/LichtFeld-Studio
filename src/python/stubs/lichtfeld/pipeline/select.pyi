"""Selection operations"""

import lichtfeld.pipeline


def all(**kwargs) -> lichtfeld.pipeline.Stage:
    """Create select-all stage"""

def none(**kwargs) -> lichtfeld.pipeline.Stage:
    """Create deselect-all stage"""

def invert(**kwargs) -> lichtfeld.pipeline.Stage:
    """Create invert-selection stage"""

def grow(**kwargs) -> lichtfeld.pipeline.Stage:
    """Create grow-selection stage"""

def shrink(**kwargs) -> lichtfeld.pipeline.Stage:
    """Create shrink-selection stage"""
