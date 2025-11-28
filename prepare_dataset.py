"""
Script untuk mengkonversi semua gambar ke JPG dan membuat CSV dataset
"""
import os
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Try to import pillow_heif for HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("Warning: pillow-heif not installed. HEIC files will be skipped.")
    print("Install with: pip install pillow-heif")

def convert_to_jpg(dataset_dir):
    """
    Konversi semua file HEIC, WebP, dan PNG ke JPG
    """
    train_dir = os.path.join(dataset_dir, 'Train')
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} tidak ditemukan!")
        return
    
    print("Mengkonversi semua gambar ke format JPG...")
    
    # Iterasi setiap folder nama orang
    person_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
    person_folders = [f for f in person_folders if not f.startswith('.') and not f.startswith('_')]
    
    converted_count = 0
    error_count = 0
    skipped_count = 0
    error_details = []
    
    for person_name in tqdm(person_folders, desc="Processing folders"):
        person_dir = os.path.join(train_dir, person_name)
        
        # Get all image files
        image_files = [f for f in os.listdir(person_dir) if not f.startswith('.')]
        
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            
            # Skip jika sudah JPG/JPEG
            if img_file.lower().endswith(('.jpg', '.jpeg')):
                skipped_count += 1
                continue
            
            # Skip jika bukan file gambar
            if not img_file.lower().endswith(('.heic', '.webp', '.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            
            try:
                # Try to open with PIL first
                try:
                    with Image.open(img_path) as img:
                        # Convert to RGB (penting untuk HEIC, PNG, WebP dengan transparency)
                        if img.mode in ('RGBA', 'LA', 'P'):
                            # Create white background untuk transparency
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Buat nama file baru dengan extension .jpg
                        base_name = os.path.splitext(img_file)[0]
                        new_filename = f"{base_name}.jpg"
                        new_path = os.path.join(person_dir, new_filename)
                        
                        # Simpan as JPG dengan kualitas tinggi
                        img.save(new_path, 'JPEG', quality=95, optimize=True)
                        
                        # Hapus file original jika bukan JPG dan konversi berhasil
                        if img_path != new_path and os.path.exists(new_path):
                            os.remove(img_path)
                            converted_count += 1
                        
                except Exception as pil_error:
                    # Jika PIL gagal, coba dengan cv2 untuk WebP
                    if img_file.lower().endswith('.webp'):
                        import cv2
                        img_cv = cv2.imread(img_path)
                        if img_cv is not None:
                            base_name = os.path.splitext(img_file)[0]
                            new_filename = f"{base_name}.jpg"
                            new_path = os.path.join(person_dir, new_filename)
                            
                            cv2.imwrite(new_path, img_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            
                            if os.path.exists(new_path):
                                os.remove(img_path)
                                converted_count += 1
                            else:
                                raise pil_error
                        else:
                            raise pil_error
                    else:
                        raise pil_error
                    
            except Exception as e:
                error_msg = f"{person_name}/{img_file}: {str(e)}"
                error_details.append(error_msg)
                print(f"\n❌ Error: {error_msg}")
                error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Konversi selesai!")
    print(f"{'='*60}")
    print(f"✓ File dikonversi: {converted_count}")
    print(f"- File sudah JPG (skip): {skipped_count}")
    print(f"✗ Error: {error_count}")
    
    if error_details:
        print(f"\nDetail errors:")
        for err in error_details[:10]:  # Show first 10 errors
            print(f"  - {err}")
        if len(error_details) > 10:
            print(f"  ... dan {len(error_details) - 10} error lainnya")
    
    return converted_count, error_count


def create_csv_dataset(dataset_dir):
    """
    Membuat file CSV yang berisi filename dan label
    """
    train_dir = os.path.join(dataset_dir, 'Train')
    
    print("\nMembuat dataset CSV...")
    
    # List untuk menyimpan data
    data = []
    
    # Get semua folder nama orang (skip hidden folders)
    person_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
    person_folders = [f for f in person_folders if not f.startswith('.') and not f.startswith('_')]
    person_folders.sort()  # Sort untuk konsistensi
    
    # Create label mapping
    label_to_id = {name: idx for idx, name in enumerate(person_folders)}
    
    print(f"Ditemukan {len(person_folders)} orang/kelas")
    
    # Iterasi setiap folder
    for person_name in tqdm(person_folders, desc="Creating CSV"):
        person_dir = os.path.join(train_dir, person_name)
        label_id = label_to_id[person_name]
        
        # Get semua file JPG di folder ini
        image_files = [f for f in os.listdir(person_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg')) and not f.startswith('.')]
        
        # Tambahkan ke data
        for img_file in image_files:
            # Format: person_name/image.jpg untuk path relatif
            relative_path = f"{person_name}/{img_file}"
            data.append({
                'filename': relative_path,
                'label': label_id,
                'person_name': person_name
            })
    
    # Buat DataFrame
    df = pd.DataFrame(data)
    
    # Simpan ke CSV
    csv_path = os.path.join(dataset_dir, 'train.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\nDataset CSV berhasil dibuat: {csv_path}")
    print(f"Total images: {len(df)}")
    print(f"Total classes: {len(person_folders)}")
    print(f"\nDistribusi per kelas:")
    print(df.groupby('person_name').size().describe())
    
    # Simpan juga label mapping
    label_map_path = os.path.join(dataset_dir, 'label_mapping.csv')
    label_df = pd.DataFrame(list(label_to_id.items()), columns=['person_name', 'label_id'])
    label_df.to_csv(label_map_path, index=False)
    print(f"\nLabel mapping disimpan: {label_map_path}")
    
    return df


if __name__ == "__main__":
    # Path ke dataset
    dataset_dir = "/home/manix/Documents/Semester 7/DeepLearn/Tubes/Dataset"
    
    print("=" * 60)
    print("DATASET PREPARATION SCRIPT")
    print("=" * 60)
    
    # Step 1: Konversi semua ke JPG
    convert_to_jpg(dataset_dir)
    
    # Step 2: Buat CSV dataset
    df = create_csv_dataset(dataset_dir)
    
    print("\n" + "=" * 60)
    print("SELESAI!")
    print("=" * 60)
    print("\nFile yang dihasilkan:")
    print(f"1. {os.path.join(dataset_dir, 'train.csv')}")
    print(f"2. {os.path.join(dataset_dir, 'label_mapping.csv')}")
    print("\nAnda bisa langsung menggunakan datareader.py sekarang!")
