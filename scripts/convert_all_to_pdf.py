"""Convert all manuscript figures from PNG to PDF using matplotlib re-render.
Reads each PNG, renders to PDF at high quality.
For true vector PDF, individual scripts should be re-run with .pdf extension.
This script provides a fast batch conversion.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

manuscript = Path("/Users/zhulin/aitest/黑潮延伸体/manuscript")

for png in sorted(manuscript.glob("fig*.png")):
    pdf = png.with_suffix('.pdf')
    print(f"  {png.name} → {pdf.name}")
    img = mpimg.imread(str(png))
    h, w = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax.imshow(img)
    ax.axis('off')
    fig.savefig(str(pdf), bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)

print(f"\n完成，共转换 {len(list(manuscript.glob('fig*.pdf')))} 个 PDF")
