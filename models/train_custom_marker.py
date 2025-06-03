# FILE: train_custom_bullseye.py
# Updated training script for custom bullseye dataset from Roboflow
# Uses modern YOLO11 instead of YOLOv5

from ultralytics import YOLO
import os
from pathlib import Path

def train_bullseye_model():
    """
    Train a custom bullseye detection model using YOLO11
    """

    # Check if dataset exists
    dataset_path = "dataset/data.yaml"  # From Roboflow export
    if not Path(dataset_path).exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please download your dataset from Roboflow and extract it to 'dataset/' folder")
        return

    # Load pretrained YOLO11 nano model (fastest, good for drones)
    print("Loading YOLO11 nano model...")
    model = YOLO("yolo11n.pt")  # You already have this file

    # Alternative models (uncomment to try):
    # model = YOLO("yolo11s.pt")  # Small - better accuracy, slower
    # model = YOLO("yolo11m.pt")  # Medium - even better accuracy, much slower

    print("Starting training...")

    # Train the model with optimized parameters for drone use
    results = model.train(
        data=dataset_path,           # Path to your Roboflow dataset
        name="marker-detection-v1",  # Updated name for your marker dataset
        epochs=100,                  # Full epochs with GPU
        imgsz=320,                   # Image size
        batch=16,                    # Good batch size for RTX 3050 Ti

        # Optimization settings
        optimizer='AdamW',           # Often better than default SGD
        lr0=0.001,                  # Learning rate
        lrf=0.01,                   # Final learning rate factor

        # Regularization
        weight_decay=0.0005,
        dropout=0.1,                # Add dropout for better generalization

        # Data augmentation (good for varying drone conditions)
        degrees=15.0,               # Rotation augmentation
        translate=0.1,              # Translation augmentation
        scale=0.3,                  # Scale augmentation
        shear=5.0,                  # Shear augmentation
        perspective=0.0001,         # Perspective augmentation
        flipud=0.0,                 # No vertical flip (bullseyes have orientation)
        fliplr=0.5,                 # Horizontal flip OK

        # Advanced settings
        mosaic=1.0,                 # Mosaic augmentation
        mixup=0.1,                  # Mixup augmentation (light)
        copy_paste=0.1,             # Copy-paste augmentation

        # Validation settings
        val=True,                   # Run validation
        save_period=10,             # Save checkpoint every 10 epochs

        # Performance settings
        cache=True,                 # Cache images for faster training
        device='cuda:0',            # Use your NVIDIA RTX 3050 Ti
        workers=4,                  # Data loading workers

        # Visualization
        plots=True,                 # Save training plots
        verbose=True,               # Detailed output

        # Early stopping
        patience=15,                # Stop if no improvement for 15 epochs

        # Resume training if interrupted
        resume=False,               # Set to True if continuing training
    )

    print("\nTraining completed!")
    print(f"Best model saved to: runs/detect/marker-detection-v1/weights/best.pt")

    # Automatically export the best model to different formats
    print("\nExporting model to different formats...")
    best_model = YOLO(results.save_dir / "weights" / "best.pt")

    # Export for Jetson Orin Nano
    try:
        best_model.export(format='onnx', imgsz=320)
        print("✓ ONNX export successful")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")

    try:
        best_model.export(format='engine', imgsz=320, device=0)  # TensorRT
        print("✓ TensorRT export successful")
    except Exception as e:
        print(f"✗ TensorRT export failed: {e}")

    try:
        best_model.export(format='torchscript', imgsz=320)
        print("✓ TorchScript export successful")
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")

    return results

def validate_dataset(dataset_path="dataset/data.yaml"):
    """
    Validate the dataset structure and contents
    """
    print("Validating dataset...")

    if not Path(dataset_path).exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return False

    # Load dataset config
    import yaml
    with open(dataset_path, 'r') as f:
        data = yaml.safe_load(f)

    # Check required fields
    required_fields = ['train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in data:
            print(f"❌ Missing field in data.yaml: {field}")
            return False

    # Check paths exist
    dataset_dir = Path(dataset_path).parent
    train_path = dataset_dir / data['train']
    val_path = dataset_dir / data['val']

    if not train_path.exists():
        print(f"❌ Training images not found: {train_path}")
        return False

    if not val_path.exists():
        print(f"❌ Validation images not found: {val_path}")
        return False

    # Count images and labels
    train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
    train_labels = list((train_path.parent / 'labels').glob('*.txt'))
    val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
    val_labels = list((val_path.parent / 'labels').glob('*.txt'))

    print(f"✓ Training: {len(train_images)} images, {len(train_labels)} labels")
    print(f"✓ Validation: {len(val_images)} images, {len(val_labels)} labels")
    print(f"✓ Classes: {data['nc']} ({data['names']})")

    if len(train_images) == 0:
        print("❌ No training images found!")
        return False

    if len(train_images) != len(train_labels):
        print(f"⚠️  Warning: Mismatch between images ({len(train_images)}) and labels ({len(train_labels)})")

    print("✓ Dataset validation passed!")
    return True

if __name__ == "__main__":
    print("Bullseye Detection Model Training")
    print("=" * 40)

    # Validate dataset first
    if validate_dataset():
        # Start training
        results = train_bullseye_model()

        print("\n" + "=" * 40)
        print("Training Summary:")
        print(f"Final mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Final mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print("Check the 'runs/detect/bullseye-drone-v1/' folder for:")
        print("- Training curves and plots")
        print("- Best and last model weights")
        print("- Exported model formats")
    else:
        print("\n📥 To get started:")
        print("1. Go to roboflow.com and create a project")
        print("2. Upload your bullseye video")
        print("3. Annotate frames with bounding boxes")
        print("4. Export as 'YOLO v5 PyTorch' format")
        print("5. Extract the downloaded zip to 'dataset/' folder")
        print("6. Run this script again")
