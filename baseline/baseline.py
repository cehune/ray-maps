import os
import mitsuba as mi
import numpy as np

mi.set_variant("scalar_rgb")

MAX_DEPTH = -1
scene_dict = mi.cornell_box()
width = 50
height = 50
scene_dict["sensor"]["film"]["width"] = width
scene_dict["sensor"]["film"]["height"] = height
scene_dict["integrator"] = {"type": "path", "max_depth": MAX_DEPTH}
scene = mi.load_dict(scene_dict)

output_folder = f"spp_renders-{width}x{height}-d{MAX_DEPTH}"
os.makedirs(output_folder, exist_ok=True)

# --- Reference: load if exists, else render and save ---
ref_exr = os.path.join(output_folder, "reference.exr")
# if not os.path.exists(ref_exr):
#     # reference = np.array(mi.Bitmap(ref_exr))
#     # print("Loaded reference from disk.")
#     pass
# else:
print("running baseline")
reference = np.array(mi.render(scene, spp=16384, seed=0))
mi.util.write_bitmap(ref_exr, reference)
mi.util.write_bitmap(ref_exr.replace(".exr", ".png"), reference ** (1/2.2))
print("Rendered and saved reference.")

def rmse(img, ref):
    return float(np.sqrt(np.mean((np.asarray(img) - np.asarray(ref)) ** 2)))

# --- Sanity check: PT-vs-itself ---
# spps = [1, 2, 4, 8, 16, 32, 64, 128, 256]
pt_errs = []

# for i, spp in enumerate(spp):
#     print("current spp: ")
#     img = mi.render(scene, spp=spp, seed=1000 + i)
#     pt_errs.append(rmse(img, reference))          # linear, pre-tonemap — correct

#     file_path = os.path.join(output_folder, f"{spp}.png")
#     mi.util.write_bitmap(file_path, img ** (1/2.2))  # gamma-corrected for viewing only

# Expected slope of log(RMSE) vs log(spp) is about -1/2 for an unbiased estimator.
# slope, intercept = np.polyfit(np.log(spps), np.log(pt_errs), 1)
# print(f"PT sanity slope = {slope:.3f}  (expect ≈ -0.5)")