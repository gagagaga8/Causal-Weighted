"""Convert RRT_ML_Paper_Illustration.png and Fig11_Causal_DAG_Generated.png to PDF."""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

for name in ["RRT_ML_Paper_Illustration.png", "Fig11_Causal_DAG_Generated.png"]:
    path = os.path.join(FIG_DIR, name)
    out = path.replace(".png", ".pdf")
    if os.path.exists(path):
        img = mpimg.imread(path)
        dpi = 150
        h, w = img.shape[:2]
        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax.imshow(img)
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(out, bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close()
        print("Saved:", out)
    else:
        print("Not found:", path)
