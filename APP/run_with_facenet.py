"""
Launcher untuk Face Recognition App dengan FaceNet Model

Usage (dari folder Tubes):
    python APP/run_with_facenet.py
    python APP/run_with_facenet.py --model-path best_facenet_model.pth
    python APP/run_with_facenet.py --num-classes 70

Usage (dari folder APP):
    python run_with_facenet.py --model-path ../best_facenet_model.pth

Usage (dengan environment variable):
    export FACENET_MODEL_PATH=/path/to/model.pth
    python APP/run_with_facenet.py

Deployment:
    1. Letakkan model di salah satu lokasi:
       - APP/best_facenet_model.pth
       - models/best_facenet_model.pth
       - Atau set FACENET_MODEL_PATH env variable
    
    2. Jalankan aplikasi:
       python APP/run_with_facenet.py
    
    3. Atau gunakan Docker dengan volume mount

Note: 
    - Auto-detect mencari model di 4 lokasi (Tubes/, APP/, current dir, models/)
    - Prioritas: CLI argument > env variable > auto-detect
    - Default num-classes adalah 70
"""

import argparse
from pathlib import Path
import sys
import os

# Import FaceNet app
from facenet_adapter import FaceNetApp


def main():
    parser = argparse.ArgumentParser(
        description="Run Face Recognition App with FaceNet Model"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to FaceNet model weights (.pth file)"
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=70,
        help="Number of persons/classes in the model"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )
    
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server address (use 0.0.0.0 for all interfaces)"
    )
    
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port"
    )
    
    args = parser.parse_args()
    
    # Resolve model path dengan prioritas:
    # 1. Command line argument
    # 2. Environment variable
    # 3. Auto-detect di beberapa lokasi
    
    if args.model_path:
        model_path = Path(args.model_path)
    elif os.getenv("FACENET_MODEL_PATH"):
        model_path = Path(os.getenv("FACENET_MODEL_PATH"))
        print(f"📍 Using model from env variable: {model_path}")
    else:
        # Auto-detect: cek beberapa lokasi umum
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir.parent / "best_facenet_model.pth",  # Tubes/best_facenet_model.pth
            script_dir / "best_facenet_model.pth",         # APP/best_facenet_model.pth
            Path("best_facenet_model.pth"),                # Current directory
            Path("models") / "best_facenet_model.pth",     # models/best_facenet_model.pth
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                print(f"📍 Auto-detected model at: {path.absolute()}")
                break
        
        if model_path is None:
            print(f"❌ Model file not found in any of these locations:")
            for path in possible_paths:
                print(f"   - {path.absolute()}")
            print(f"\n💡 Possible solutions:")
            print(f"   1. Train the model first: python train_facenet.py")
            print(f"   2. Download model to one of the locations above")
            print(f"   3. Specify path manually: --model-path /path/to/model.pth")
            print(f"   4. Set environment variable: export FACENET_MODEL_PATH=/path/to/model.pth")
            print(f"\n💾 For deployment, you can:")
            print(f"   - Put model.pth in APP/ folder")
            print(f"   - Put model.pth in models/ folder")
            print(f"   - Set FACENET_MODEL_PATH environment variable")
            sys.exit(1)
    
    # Validate model path exists
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path.absolute()}")
        print(f"\n💡 Make sure the model file exists at the specified location")
        sys.exit(1)
    
    print("=" * 70)
    print("🚀 Starting Face Recognition App with FaceNet")
    print("=" * 70)
    print(f"📦 Model Path: {model_path.absolute()}")
    print(f"👥 Num Classes: {args.num_classes}")
    print(f"🌐 Server: {args.server_name}:{args.server_port}")
    print(f"🔗 Share: {args.share}")
    print("=" * 70)
    
    # Create and launch app
    try:
        app = FaceNetApp(
            model_path=str(model_path.absolute()),
            num_classes=args.num_classes
        )
        
        print("\n✅ FaceNet app initialized successfully!")
        print("\n🌐 Launching Gradio interface...")
        print("-" * 70)
        
        # Launch Gradio
        app.launch(
            share=args.share,
            server_name=args.server_name,
            server_port=args.server_port
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
