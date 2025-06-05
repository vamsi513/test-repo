import torch

file_path = 'models/emotion_model.pth'  # update the path if needed

try:
    # Try loading as state_dict
    data = torch.load(file_path, map_location='cpu')
    
    if isinstance(data, dict) and all(isinstance(k, str) for k in data.keys()):
        print("✅ File is a valid state_dict (use model.load_state_dict())")
        print(f"State dict keys example: {list(data.keys())[:5]}")
    else:
        print("⚠ File loaded but does not fully look like a state_dict, check carefully.")
except Exception as e:
    print(f"❌ Failed to load as state_dict: {e}")
    
    try:
        # Try loading as full model
        model = torch.load(file_path, map_location='cpu', weights_only=False)
        print("✅ File is a full model object (use model = torch.load())")
    except Exception as e:
        print(f"❌ Failed to load as full model: {e}")
