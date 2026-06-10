"""
Download Weapon Detection Model from Roboflow
"""

from roboflow import Roboflow
import shutil
from pathlib import Path

print("="*80)
print("DOWNLOADING WEAPON DETECTION DATASET")
print("="*80)
print()

try:
    # Your Roboflow code
    rf = Roboflow(api_key="unauthorized")
    project = rf.workspace("joao-assalim-xmovq").project("weapon-2")
    version = project.version(2)
    
    print("📥 Downloading dataset...")
    dataset = version.download("yolov8")
    
    print(f"✅ Downloaded to: {dataset.location}")
    print()
    
    # Find the model file
    dataset_path = Path(dataset.location)
    
    # Common locations for the weights file
    possible_paths = [
        dataset_path / "weights" / "best.pt",
        dataset_path / "best.pt",
        dataset_path / "runs" / "detect" / "train" / "weights" / "best.pt",
    ]
    
    model_file = None
    for path in possible_paths:
        if path.exists():
            model_file = path
            print(f"✅ Found model: {path}")
            break
    
    if not model_file:
        # Search recursively
        print("🔍 Searching for model file...")
        for pt_file in dataset_path.rglob("*.pt"):
            print(f"   Found: {pt_file}")
            if "best" in pt_file.name.lower():
                model_file = pt_file
                break
    
    if model_file:
        # Copy to models directory
        target = Path("models/weapon_model.pt")
        target.parent.mkdir(exist_ok=True)
        
        shutil.copy2(model_file, target)
        
        print()
        print("="*80)
        print("✅ SUCCESS!")
        print("="*80)
        print(f"Model copied to: {target.absolute()}")
        print()
        print("You can now use the weapon detector!")
        print("Run: python file_weapon_detector.py")
        print("="*80)
        
    else:
        print()
        print("⚠️ WARNING: Could not find best.pt file")
        print(f"Dataset downloaded to: {dataset.location}")
        print("Please manually locate the .pt file and copy it to: models/weapon_model.pt")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    print()
    print("The download might have failed. Make sure:")
    print("1. You have internet connection")
    print("2. The API key is valid (even 'unauthorized' should work for public datasets)")
    print("3. Roboflow is installed: pip install roboflow")
