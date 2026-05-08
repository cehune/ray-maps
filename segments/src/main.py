import mitsuba as mi
from primitives import *
from renderer import Renderer





# ---- main ------------------------------------------------------------------

def main():
    mi.set_variant('scalar_rgb')

    scene_file_path = '../samples/cbox/cbox.xml'
    save_file_path = "output.png"
    renderer = Renderer()
    renderer.save_render_by_scene_path(scene_file_path, save_file_path, width = 128, height = 128, spp = 32)



if __name__ == '__main__':
    main()