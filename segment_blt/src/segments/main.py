import mitsuba as mi
from segments.primitives import *
from segments.renderer import Renderer

# ---- main ------------------------------------------------------------------

def main():
    mi.set_variant('scalar_rgb')

    scene_file_path = '../samples/cbox/cbox.xml'
    save_file_path = "output6.png"
    renderer = Renderer()
    renderer.save_render_by_scene_path(scene_file_path, save_file_path, width = 200, height = 200, n_iterations=8, mode='vec', beta=0.1, add_light_samples=False)

if __name__ == '__main__':
    main()