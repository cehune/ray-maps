from primitives import *

class SegmentPool:
    camera_paths:  list[list[Segment]] = field(default_factory=list)    
    light_paths:  list[list[Segment]] = field(default_factory=list)    

    def add_camera_path(self, path: list[Segment]):
        if len(path):
            self.camera_paths.append(path)
    def add_light_path(self, path: list[Segment]):
        if len(path):
            self.light_paths.append(path)

    def clear(self):
        self.camera_paths.clear()
        self.light_paths.clear()

    @property
    def num_camera_segments(self):
        return sum(len(p) for p in self.camera_paths)

    @property
    def num_light_segments(self):
        return sum(len(p) for p in self.light_paths)

    
