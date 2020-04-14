from src.image import Image

img = Image('images/batman.jpg')
palette = img.generate_color_clusters()
print(palette.values)
palette.display_colors()

