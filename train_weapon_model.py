import torch
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import gc
import multiprocessing as mp

def train_weapon_model():
    print("=" * 70)
    print("WEAPON MODEL TRAINING - HYBRID CPU+GPU MODE")
    print("=" * 70)
    print("\nUsing BOTH CPU and GPU for maximum performance!")
    print("GPU: Model training (forward/backward pass)")
    print("CPU: Data loading, preprocessing, augmentation\n")
    
    # Use GPU for training, CPU for data processing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_mem_gb:.2f} GB")
        
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        print("CUDA optimizations enabled")
        print("GPU cache cleared")
    
    # Get CPU core count
    cpu_cores = mp.cpu_count()
    print(f"CPU Cores: {cpu_cores}")
    print()
    
    # Load base model
    print("Loading YOLOv8 nano model...")
    model = YOLO('models/yolov8n.pt')
    
    # HYBRID CPU+GPU optimized settings
    BATCH_SIZE = 2          # Very small batch on GPU
    WORKERS = cpu_cores - 2  # Use most CPU cores for data loading
    EPOCHS = 50             # Reasonable epoch count
    IMG_SIZE = 320          # Smallest viable size
    
    # Pin memory for faster CPU->GPU transfer
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    print(f"Training Configuration (Hybrid CPU+GPU):")
    print(f"  Device: {device.upper()}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE} (minimal GPU load)")
    print(f"  Workers: {WORKERS} (CPU cores for parallel data loading)")
    print(f"  Image Size: {IMG_SIZE} (reduced for GPU)")
    print(f"  Pin Memory: {PIN_MEMORY} (fast CPU->GPU transfer)")
    print(f"  Estimated Time: 90-150 minutes")
    print()
    print(f"Strategy: GPU trains while CPU prepares next batches!\n")
    
    try:
        results = model.train(
            data='Weapon-2-2/data.yaml',
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=device,
            
            # Hybrid CPU+GPU optimization
            workers=WORKERS,        # Max CPU cores for data loading
            amp=True,               # FP16 on GPU
            cache='ram',            # Cache on CPU RAM (not GPU)
            
            # Memory settings
            rect=True,              # Rectangular training (less padding)
            close_mosaic=10,        # Disable mosaic after epoch 10
            
            # Training settings
            patience=25,
            save=True,
            save_period=10,
            
            # Light augmentation (faster CPU processing)
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=5.0,
            translate=0.05,
            scale=0.3,
            fliplr=0.5,
            mosaic=0.8,
            mixup=0.0,
            copy_paste=0.0,
            
            # Output
            project='runs/train',
            name=f'weapon_hybrid_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            exist_ok=True,
            pretrained=True,
            val=True,
            plots=True,
            verbose=True
        )
        
        # Success!
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        
        # Export model
        import shutil
        Path('models').mkdir(exist_ok=True)
        best_path = Path('runs/train') / results.save_dir.name / 'weights' / 'best.pt'
        output_path = 'models/weapon_model_upgraded.pt'
        shutil.copy2(best_path, output_path)
        
        print(f"\nUpgraded model: {output_path}")
        print(f"Training logs: runs/train/{results.save_dir.name}")
        print("\nNext steps:")
        print("1. Backup: Rename weapon_model.pt to weapon_model_old.pt")
        print("2. Deploy: Rename weapon_model_upgraded.pt to weapon_model.pt")
        print("3. Test with file_weapon_detector.py")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "=" * 70)
            print("GPU OUT OF MEMORY!")
            print("=" * 70)
            print("\nEven with batch=2, GPU ran out of memory.")
            print("Your GPU may be too busy with other tasks.")
            print("\nSolutions:")
            print("1. Close ALL applications (especially browsers)")
            print("2. Restart computer and run ONLY this script")
            print("3. Check GPU usage with: nvidia-smi")
            raise
        else:
            raise

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    train_weapon_model()
