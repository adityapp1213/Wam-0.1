from pathlib import Path

from PIL import Image, ImageDraw

root = Path('/home/aditya/robotics/wam-0.1/reports/door_variation_pilot')
files = sorted((root / 'normal').glob('*.png')) + sorted((root / 'random').glob('*.png'))
thumb_w = 384
thumb_h = 422
cols = 2
rows = (len(files) + cols - 1) // cols
sheet = Image.new('RGB', (cols * thumb_w, rows * thumb_h), 'white')
draw = ImageDraw.Draw(sheet)
for index, path in enumerate(files):
    image = Image.open(path).convert('RGB')
    image.thumbnail((thumb_w - 8, thumb_h - 28))
    x = (index % cols) * thumb_w + (thumb_w - image.width) // 2
    y = (index // cols) * thumb_h + 22
    sheet.paste(image, (x, y))
    draw.text(((index % cols) * thumb_w + 6, 5 + (index // cols) * thumb_h), path.stem, fill='black')
out = root / 'door_pilot_contact_sheet.png'
sheet.save(out)
print(out)
