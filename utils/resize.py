from PIL import Image
import os


def crop_and_resize(image_path, output_path, crop_size=(720, 720), resize_size=(128, 128)):
    with Image.open(image_path) as img:
        width, height = img.size
        center_x, center_y = width // 2, height // 2

        crop_x1 = max(center_x - crop_size[0] // 2, 0)
        crop_x2 = min(center_x + crop_size[0] // 2, width)
        crop_y1 = max(center_y - crop_size[1] // 2, 0)
        crop_y2 = min(center_y + crop_size[1] // 2, height)

        cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # ANTIALIAS를 LANCZOS로 교체
        resized_img = cropped_img.resize(resize_size)

        # 이미지가 RGBA 모드인 경우 RGB로 변환
        if resized_img.mode == 'RGBA':
            resized_img = resized_img.convert("RGB")

        resized_img.save(output_path)

image_directory = "Data/Training/org_model" 
output_directory = "Data/Training/resized_org_model" 

# 디렉토리 생성 (존재하지 않을 경우)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(image_directory):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(image_directory, filename)
        output_path = os.path.join(output_directory, filename)

        print(f"Processing: {image_path}") 

        try:
            crop_and_resize(image_path, output_path)
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
