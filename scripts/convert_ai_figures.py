import shutil
import os
from PIL import Image

src_dir = r"C:\Users\czy\.cursor\projects\c-Dynamic-RRT\assets"
dst_dir = r"C:\Dynamic-RRT\figures"

os.makedirs(dst_dir, exist_ok=True)

files = ["Fig11_Causal_DAG_Generated.png", "RRT_ML_Paper_Illustration.png"]

for file in files:
    src_path = os.path.join(src_dir, file)
    dst_png = os.path.join(dst_dir, file)
    dst_pdf = os.path.join(dst_dir, file.replace(".png", ".pdf"))
    
    if os.path.exists(src_path):
        # copy PNG
        shutil.copy2(src_path, dst_png)
        # convert to PDF
        image = Image.open(src_path)
        img_converted = image.convert("RGB")
        img_converted.save(dst_pdf, "PDF", resolution=100.0)
        print(f"Successfully processed {file}")
    else:
        print(f"File not found: {src_path}")
