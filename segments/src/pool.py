from dataclasses import dataclass, field
from primitives import *

@dataclass
class SegmentPool:
    camera_paths:  list[list[Segment]] = field(default_factory=list)    
    light_paths:  list[list[Segment]] = field(default_factory=list)    

    def add_camera_path(this, path: list[Segment]):
        if len(path):
            this.camera_paths.append(path)
    def add_light_path(this, path: list[Segment]):
        if len(path):
            this.light_paths.append(path)

    def clear(this):
        this.camera_paths.clear()
        this.light_paths.clear()

    @property
    def num_camera_segments(this):
        return sum(len(p) for p in this.camera_paths)

    @property
    def num_light_segments(this):
        return sum(len(p) for p in this.light_paths)

    
