"""
Launcher untuk Face Recognition App dengan ResNet Model

Usage (dari folder Tubes):
    python APP/run_with_resnet.py
    python APP/run_with_resnet.py --model-path checkpoints/resnet34_frozen_best.pth

Usage (dari folder APP):
    python run_with_resnet.py --model-path ../checkpoints/resnet34_frozen_best.pth

Deployment:
    1. Model otomatis load dari checkpoints/resnet34_frozen_best.pth
    2. Jalankan aplikasi:
       python APP/run_with_resnet.py
"""

import argparse
from pathlib import Path
import sys
import os

# Import ResNet adapter
from resnet_adapter import ResNetApp


def main():
    parser = argparse.ArgumentParser(
        description="Run Face Recognition App with ResNet Model"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to ResNet model file (.pth)"
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=70,  # 70 classes di ResNet model
        help="Number of face classes in the model"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the server (default: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    
    args = parser.parse_args()
    
    # Auto-detect model path jika tidak di-specify
    if args.model_path is None:
        # Coba cari di beberapa lokasi umum
        possible_paths = [
            Path("checkpoints/resnet34_frozen_best.pth"),  # Dari Tubes/
            Path("../checkpoints/resnet34_frozen_best.pth"),  # Dari APP/
            Path("resnet34_frozen_best.pth"),  # Current dir
            Path("models/resnet34_frozen_best.pth"),  # models folder
        ]
        
        for path in possible_paths:
            if path.exists():
                args.model_path = str(path)
                print(f"✅ Auto-detected model at: {args.model_path}")
                break
        
        if args.model_path is None:
            print("❌ Error: Model tidak ditemukan!")
            print("   Coba specify path dengan --model-path")
            print(f"   Lokasi yang dicek:")
            for p in possible_paths:
                print(f"     - {p.absolute()}")
            sys.exit(1)
    
    # Verify model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Error: Model file tidak ditemukan: {model_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"🚀 Face Recognition App - ResNet Model")
    print(f"{'='*60}")
    print(f"📦 Model: {model_path.name}")
    print(f"📍 Path: {model_path.absolute()}")
    print(f"👥 Classes: {args.num_classes}")
    print(f"🌐 Server: {args.host}:{args.port}")
    print(f"🔗 Share: {'Enabled' if args.share else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # Initialize and run app
    app = ResNetApp(
        model_path=str(model_path.absolute()),
        num_classes=args.num_classes
    )
    
    # Launch Gradio interface
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
