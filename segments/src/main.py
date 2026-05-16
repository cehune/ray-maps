import mitsuba as mi
from primitives import *
from segments.src.renderer import Renderer





# ---- main ------------------------------------------------------------------

def main():
    mi.set_variant('scalar_rgb')

    scene_file_path = '../samples/cbox/cbox.xml'
    save_file_path = "output3.png"
    renderer = Renderer()
    renderer.save_render_by_scene_path(scene_file_path, save_file_path, width = 128, height = 128, spp = 64)



if __name__ == '__main__':
    main()