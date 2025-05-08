import os, shutil, cv2

def move_text_files(src_root, dst_root): # moving the .txt to a labels folder so yolov8 can register the labels
    # Walk through the directory structure of the source path
    for folder_path, subfolders, files in os.walk(src_root):
        for file in files:
            # Only move .txt files
            if file.endswith(".txt"):
                # Construct the full source file path
                file.replace("_BB", "")
                src_file_path = os.path.join(folder_path, file)

                # Construct the destination file path using absolute paths
                # Replace 'images' with 'labels' in the destination path
                dst_file_path = src_file_path.replace("images", "labels")
                
                # Create the destination directory if it doesn't exist
                dst_folder_path = os.path.dirname(dst_file_path)
                os.makedirs(dst_folder_path, exist_ok=True)

                # Move the file
                shutil.move(src_file_path, dst_file_path)
                print(f"Moved: {src_file_path} -> {dst_file_path}")


def convert_to_yolo_format_from_bbox(bbox, img_width=None, img_height=None, class_id=None):
    x, y, width, height, _ = bbox
    x, y, width, height = map(float, [x, y, width, height])

    # Calculate center
    x_center = x + (width / 2)
    y_center = y + (height / 2)

    # Normalize coordinates
    x_center_normalized = x_center / img_width
    y_center_normalized = y_center / img_height
    width_normalized = width / img_width
    height_normalized = height / img_height

    # YOLO format: class_id x_center y_center width height
    return f"{class_id} {x_center_normalized:.6f} {y_center_normalized:.6f} {width_normalized:.6f} {height_normalized:.6f}"


def convert_to_yolo(root):
    for folder_path, subfolders, files in os.walk(root):
        class_id = 0 if "live" in folder_path else 1
        
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(folder_path, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # Construct image file path (try both .jpg and .png)
                img_base_path = os.path.splitext(file_path)[0].replace('_BB', '')
                img_file_path = img_base_path + ".jpg"
                if not os.path.exists(img_file_path):
                    img_file_path = img_base_path + ".png"
                if not os.path.exists(img_file_path):
                    continue  # Skip if neither format is found

                # Use cv2 to get image dimensions
                img = cv2.imread(img_file_path)
                if img is None:
                    continue  # Skip if image can't be read
                img_height, img_width = img.shape[:2]  # Get height and width

                # Convert each line in the .txt file to YOLO format
                yolo_lines = []
                for line in lines:
                    bbox = list(map(float, line.strip().split()))  # Convert the bbox values to floats
                    yolo_line = convert_to_yolo_format_from_bbox(bbox, img_width, img_height, class_id)
                    yolo_lines.append(yolo_line)

                # Write the YOLO formatted data to a new file
                with open(file_path, 'w') as f:
                    f.writelines(yolo_lines)

                print(f"Processed: {file_path}")


def crop(bbox, img_height=None, img_width=None):
    x, y, w, h = bbox
    
    x = int(bbox[0] * (img_width / 224))
    y = int(bbox[1] * (img_height / 224))
    w = int(bbox[2] * (img_width / 224))
    h = int(bbox[3] * (img_height / 224))

    # Clamp values to ensure they are within image bounds
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    w = max(1, min(w, img_width - x))  # Ensure width > 0
    h = max(1, min(h, img_height - y))  # Ensure height > 0

    return [x, y, w, h]


def crop_image(root):
    for folder_path, _, files in os.walk(root):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(folder_path, file)

                # Construct image file path (try both .jpg and .png)
                img_base_path = os.path.splitext(file_path)[0].replace('_BB', '')
                img_file_path = img_base_path + ".jpg"
                if not os.path.exists(img_file_path):
                    img_file_path = img_base_path + ".png"
                if not os.path.exists(img_file_path):
                    print(f'Image not found: {img_file_path}')
                    continue  # Skip if neither format is found
                
                # Read image once
                img = cv2.imread(img_file_path)
                if img is None:
                    print(f"Failed to load image: {img_file_path}")
                    continue
                
                img_height, img_width = img.shape[:2]

                # Process bounding boxes
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    bbox = list(map(float, line.strip().split()))[:4]  # Extract bbox coordinates
                    bbox = list(map(int, bbox))
                    x, y, w, h = crop(bbox, img_height, img_width)
                    
                    cropped_face = img[y:y + h, x:x + w]

                    # Save cropped image with a new filename to avoid overwriting
                    cv2.imwrite(img_file_path, cropped_face)
                    print(f"Cropped and saved: {img_file_path}")


def remove_id_folders(root):
    for subset in ["train", "val"]: 
        subset_path = os.path.join(root, subset) # images/train
        
        # Iterate over ID folders (like '21', '22')
        for id_folder in os.listdir(subset_path):
            id_folder_path = os.path.join(subset_path, id_folder) # images/train/ID
            
            # Check if it's a directory (ID folder)
            if os.path.isdir(id_folder_path):
                for class_name in os.listdir(id_folder_path):  # Iterate over 'live', 'spoof'
                    class_folder_path = os.path.join(id_folder_path, class_name) # images/train/ID/class
                    
                    if os.path.isdir(class_folder_path):  # Check if it's a class folder (live/spoof)
                        # Move all files from the class folder into the corresponding class folder
                        for file in os.listdir(class_folder_path):
                            src_file_path = os.path.join(class_folder_path, file) # images/train/ID/class/image
                            dst_file_path = os.path.join(subset_path, class_name, file) # images/train/class/file
                            shutil.move(src_file_path, dst_file_path)
                        
                        # After moving files, remove the class folder
                        os.rmdir(class_folder_path)
                        print(f"Removed class folder: {class_folder_path}")
                
                # After moving all files from the ID folder, remove the ID folder
                os.rmdir(id_folder_path)
                print(f"Removed ID folder: {id_folder_path}")


def remove_id_folders(root):
    for subset in ["train", "val"]: 
        subset_path = os.path.join(root, subset) # images/train
        if not os.path.isdir(subset_path):
            continue
        
        # Iterate over ID folders (like '21', '22')
        for id_folder in os.listdir(subset_path):
            id_folder_path = os.path.join(subset_path, id_folder) # images/train/ID
            
            # Check if it's a directory (ID folder)
            if os.path.isdir(id_folder_path):
                for class_name in os.listdir(id_folder_path):  # Iterate over 'live', 'spoof'
                    class_folder_path = os.path.join(id_folder_path, class_name) # images/train/ID/class
                    
                    if os.path.isdir(class_folder_path):  # Check if it's a class folder (live/spoof)
                        dst_folder_path = os.path.join(subset_path, class_name)
                        os.makedirs(dst_folder_path, exist_ok=True)
                        
                        for file in os.listdir(class_folder_path):
                            src_file_path = os.path.join(class_folder_path, file) # images/train/ID/class/image
                            dst_file_path = os.path.join(dst_folder_path, file) # images/train/class/file
                            shutil.move(src_file_path, dst_file_path)
                        
                        # After moving files, remove the class folder
                        os.rmdir(class_folder_path)
                        print(f"Removed class folder: {class_folder_path}")
                
                # After moving all files from the ID folder, remove the ID folder
                os.rmdir(id_folder_path)
                print(f"Removed ID folder: {id_folder_path}")


if __name__ == "__main__":
    root = 'CelebA_Spoof'
    crop_image(root)
    # convert_to_yolo(root)
    move_text_files(os.path.join(root, 'images'), os.path.join(root, 'labels'))
    remove_id_folders(os.path.join(root, 'images'))