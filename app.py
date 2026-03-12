import os
import sys
import json
import datetime
import threading
import time
import subprocess
from pathlib import Path
from flask import (
    Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, Response
)
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# PyTorch imports
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image

# --------------------
# Config / Paths
# --------------------
BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "static" / "uploads"
RESULTS = BASE / "results"
REPORTS = BASE / "reports"
MODEL_FILE = BASE / "fine_tuned_EEG_CIFAR10.pth"
HISTORY_FILE = BASE / "history.json"
CHAT_HISTORY_FILE = BASE / "results" / "chat_history.csv"
PROGRESS_FILE = BASE / "training_progress.json"
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

if not HISTORY_FILE.exists():
    HISTORY_FILE.write_text("[]")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret_please_change")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# --------------------
# Global training state
# --------------------
TRAIN_STATUS = {
    "running": False,
    "progress": 0,
    "message": "Idle",
    "loss": [],
    "acc": [],
    "mode": "dataset"  # dataset or custom
}

# --------------------
# Model Loading
# --------------------
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Use GPU if available for faster predictions
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    if MODEL_FILE.exists():
        try:
            state_dict = torch.load(str(MODEL_FILE), map_location=DEVICE)
            model.load_state_dict(state_dict)
            print("[OK] Model loaded successfully")
        except Exception as e:
            print(f"[WARN] Could not load model state: {e}")
    else:
        print("[WARN] Model not found, using untrained model")
    model = model.to(DEVICE)
    model.eval()
    return model

MODEL = load_model()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# --------------------
# Prediction Logic
# --------------------
def predict_image_from_path(img_path):
    """Predict image class with simple test-time augmentation (scale + center-crop averaging).

    This helps when uploaded photos differ from the small CIFAR-10 training images
    by averaging softmax outputs from a few resized/cropped variants.
    """
    img = Image.open(str(img_path)).convert("RGB")

    # define a small set of transforms to average predictions over
    aug_transforms = [
        transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
        transforms.Compose([transforms.Resize((48, 48)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
        transforms.Compose([transforms.Resize((64, 64)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    ]

    probs_sum = None
    MODEL.eval()
    with torch.no_grad():
        for t in aug_transforms:
            try:
                x = t(img).unsqueeze(0).to(DEVICE)
                out = MODEL(x)
                p = F.softmax(out, dim=1).cpu()
                if probs_sum is None:
                    probs_sum = p
                else:
                    probs_sum += p
            except Exception as e:
                # if a particular augmentation fails, skip it
                print(f"[WARN] augmentation prediction failed: {e}")

    if probs_sum is None:
        # fallback to single-transform prediction
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = MODEL(x)
            probs = F.softmax(out, dim=1).cpu().numpy().squeeze()
    else:
        probs = (probs_sum / len(aug_transforms)).numpy().squeeze()

    top_idx = probs.argsort()[-3:][::-1]
    top3 = [{"class": CLASSES[int(i)], "prob": round(float(probs[int(i)]) * 100, 2)} for i in top_idx]
    primary = top3[0]
    return primary["class"], primary["prob"], top3

# --------------------
# Routes
# --------------------
@app.route("/")
def dashboard():
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    
    summary = (RESULTS / "summary.txt").read_text() if (RESULTS / "summary.txt").exists() else "No training summary available. Run training to generate results."
    loss_png = (RESULTS / "training_curves.png").exists()
    pred_grid = (RESULTS / "predictions_grid.png").exists()
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    model_present = MODEL_FILE.exists()
    
    # Count custom images
    custom_images_count = len(list(UPLOADS.glob("*.*")))
    
    return render_template(
        "dashboard.html",
        history=history,
        summary=summary,
        loss_png=loss_png,
        pred_grid=pred_grid,
        device_name=device_name,
        model_present=model_present,
        custom_images_count=custom_images_count
    )

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

# Enhanced Upload + Prediction API
def predict_and_save_record():
    """Shared prediction logic for both /upload and /predict routes"""
    if "file" not in request.files:
        return {"ok": False, "error": "No file part"}
    file = request.files["file"]
    if file.filename == "":
        return {"ok": False, "error": "No file selected"}
    if not allowed_file(file.filename):
        return {"ok": False, "error": "Unsupported file type"}

    fname = secure_filename(file.filename)
    save_path = UPLOADS / fname
    file.save(str(save_path))

    try:
        label, prob, top3 = predict_image_from_path(save_path)
    except Exception as e:
        return {"ok": False, "error": f"Prediction failed: {e}"}

    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": fname,
        "prediction": label,
        "confidence": f"{prob:.2f}%"
    }

    with open(HISTORY_FILE, "r+") as f:
        try:
            data = json.load(f)
        except:
            data = []
        data.insert(0, record)
        data = data[:200]
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    return {"ok": True, "record": record, "top3": top3}

@app.route("/upload", methods=["POST"])
def upload_file():
    result = predict_and_save_record()
    
    # Check if user wants to train with this image
    train_with_image = request.form.get('train_with_image', 'false').lower() == 'true'
    label = request.form.get('image_label', '').strip()
    
    if train_with_image and label and result.get('ok'):
        # Start background training with this image
        file = request.files.get('file')
        if file:
            # Get the saved file path
            fname = secure_filename(file.filename)
            image_path = UPLOADS / fname
            
            # Start training in background
            def train_with_custom_image():
                try:
                    print(f"\n[TRAINING] Starting training with custom image: {label}")
                    # Import training function
                    from main import train_model
                    from pathlib import Path
                    
                    # Train the model with this custom image
                    accuracy = train_model(
                        train_on_custom=True,
                        custom_image_path=str(image_path),
                        custom_label=label
                    )
                    
                    print(f"[OK] Training completed with custom image. Accuracy: {accuracy:.2f}%")
                except Exception as e:
                    print(f"[ERROR] Training failed: {e}")
            
            thread = threading.Thread(target=train_with_custom_image, daemon=True)
            thread.start()
            result['training_started'] = True
    
    return jsonify(result)

@app.route("/predict", methods=["POST"])
def predict_file():
    return jsonify(predict_and_save_record())

@app.route("/history")
def history_page():
    with open(HISTORY_FILE, "r") as f:
        data = json.load(f)
    return render_template("history.html", history=data)


@app.route("/history_data")
def history_data():
    """Return recent prediction history as JSON for frontend refresh."""
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        data = []
    return jsonify({"history": data})

@app.route("/clear_history", methods=["POST"])
def clear_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)
    return jsonify({"ok": True})

@app.route("/clear_uploads", methods=["POST"])
def clear_uploads():
    """Clear all uploaded custom images"""
    try:
        for file_path in UPLOADS.glob("*.*"):
            file_path.unlink()
        return jsonify({"ok": True, "message": "All uploaded images cleared"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# --------------------
# Training with Real Progress
# --------------------
def run_main_and_wait(mode="dataset"):
    main_py = BASE / "main.py"
    if not main_py.exists():
        return False, "main.py not found"
    try:
        # Pass mode as command line argument
        if mode == "custom":
            p = subprocess.Popen([sys.executable, str(main_py), "--mode", "custom"], cwd=str(BASE))
        else:
            p = subprocess.Popen([sys.executable, str(main_py)], cwd=str(BASE))
        p.wait()
    except Exception as e:
        return False, str(e)

    # Read final results
    summary_path = RESULTS / "summary.txt"
    final_acc = None
    training_mode = "dataset"
    if summary_path.exists():
        try:
            s = summary_path.read_text()
            import re
            m = re.search(r"Final Test Accuracy:\s*([0-9.]+)%", s)
            if m:
                final_acc = float(m.group(1))
            
            # Extract training mode
            mode_match = re.search(r"Training Mode:\s*(\w+)", s)
            if mode_match:
                training_mode = mode_match.group(1)
        except:
            final_acc = None

    csv_path = RESULTS / "training_results.csv"
    losses, accs = [], []
    if csv_path.exists():
        import csv
        with open(csv_path, "r") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                try:
                    if len(row) >= 3:
                        losses.append(float(row[1]))
                        accs.append(float(row[2]))
                    elif len(row) == 2:
                        losses.append(float(row[1]))
                        accs.append(0.0)
                except:
                    continue

    return True, {"final_acc": final_acc, "losses": losses, "accs": accs, "mode": training_mode}

def monitor_training_progress():
    """Monitor training progress from main.py"""
    while TRAIN_STATUS["running"]:
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, "r") as f:
                    progress = json.load(f)
                
                TRAIN_STATUS["progress"] = progress["progress_percent"]
                mode = progress.get("mode", "dataset")
                
                if mode == "custom":
                    TRAIN_STATUS["message"] = f"Custom Training - Epoch {progress['epoch']}/{progress['total_epochs']} - Loss: {progress['loss']:.4f}"
                else:
                    TRAIN_STATUS["message"] = f"Dataset Training - Epoch {progress['epoch']}/{progress['total_epochs']} - Loss: {progress['loss']:.4f}"
                
                if progress.get("accuracy"):
                    TRAIN_STATUS["message"] += f" - Acc: {progress['accuracy']:.2f}%"
                    
            except Exception as e:
                print(f"Progress read error: {e}")
        
        time.sleep(2)

@app.route("/train", methods=["POST"])
def train_route():
    if TRAIN_STATUS["running"]:
        return jsonify({"ok": False, "msg": "Training already running"}), 400

    # Get training mode from request
    data = request.get_json() or {}
    mode = data.get("mode", "dataset")
    
    TRAIN_STATUS["mode"] = mode

    def background_job():
        TRAIN_STATUS["running"] = True
        TRAIN_STATUS["progress"] = 0
        TRAIN_STATUS["message"] = f"Starting {mode} training..."
        TRAIN_STATUS["loss"].clear()
        TRAIN_STATUS["acc"].clear()

        # Start progress monitoring
        progress_thread = threading.Thread(target=monitor_training_progress, daemon=True)
        progress_thread.start()

        # Run training
        ok, info = run_main_and_wait(mode)
        
        if not ok:
            TRAIN_STATUS["message"] = f"Training failed: {info}"
        else:
            if isinstance(info, dict):
                TRAIN_STATUS["loss"] = info.get("losses", [])
                TRAIN_STATUS["acc"] = info.get("accs", [])
                if info.get("final_acc"):
                    TRAIN_STATUS["message"] = f"{info.get('mode', 'Training')} complete — Final Accuracy: {info['final_acc']:.2f}%"
                else:
                    TRAIN_STATUS["message"] = f"{info.get('mode', 'Training')} complete"
            TRAIN_STATUS["progress"] = 100

        TRAIN_STATUS["running"] = False

    threading.Thread(target=background_job, daemon=True).start()
    return jsonify({"ok": True, "msg": f"{mode.capitalize()} training started"})

@app.route("/status")
def status_route():
    return jsonify(TRAIN_STATUS)

@app.route("/chart_data")
def chart_data_route():
    csv_path = RESULTS / "training_results.csv"
    losses, accs = [], []
    
    if csv_path.exists():
        import csv
        try:
            with open(csv_path, "r") as f:
                r = csv.reader(f)
                header = next(r, None)
                for row in r:
                    try:
                        if len(row) >= 3:
                            losses.append(float(row[1]))
                            accs.append(float(row[2]))
                        elif len(row) == 2:
                            losses.append(float(row[1]))
                            accs.append(0.0)
                    except:
                        continue
        except Exception as e:
            print(f"Error reading CSV: {e}")
    
    # If no CSV data, use training status data
    if not losses and TRAIN_STATUS["loss"]:
        losses = TRAIN_STATUS["loss"]
        accs = TRAIN_STATUS["acc"]

    # Compute axis metadata so frontend can match matplotlib PNG axes
    def compute_axis_meta(values, is_loss=False):
        """Compute reasonable axis limits for chart.js"""
        try:
            vals = [float(v) for v in values if v is not None]
            if not vals:
                return {"min": 0.0, "max": 1.0, "step": 0.1}
            
            vmin = min(vals)
            vmax = max(vals)
            
            # Ensure minimum range
            if vmax == vmin:
                if vmax == 0:
                    vmin, vmax = -0.1, 0.1
                else:
                    margin = abs(vmax) * 0.2
                    vmin -= margin
                    vmax += margin
            else:
                # Add margin (20% of range)
                margin = (vmax - vmin) * 0.2
                vmin -= margin
                vmax += margin
            
            # For loss (0-1 range typically)
            if is_loss:
                vmin = max(0, vmin)
                # Ensure at least 0.01 range
                if vmax - vmin < 0.01:
                    vmax = vmin + 0.01
            
            # For accuracy (0-100 range)
            else:
                vmin = max(0, vmin)
                vmax = min(100, vmax)
                # Ensure at least 5% range
                if vmax - vmin < 5:
                    vmax = min(100, vmin + 5)
            
            # Calculate nice step
            import math
            raw_range = vmax - vmin
            raw_step = raw_range / 5.0  # aim for 5 ticks
            
            if raw_step <= 0:
                step = 0.01 if is_loss else 1
            else:
                exp = math.floor(math.log10(raw_step))
                base = 10 ** exp
                frac = raw_step / base
                
                if frac <= 1.5:
                    nice = 1
                elif frac <= 3:
                    nice = 2
                elif frac <= 7:
                    nice = 5
                else:
                    nice = 10
                step = nice * base
            
            # Round to step boundaries
            vmin = float(math.floor(vmin / step) * step)
            vmax = float(math.ceil(vmax / step) * step)
            
            return {"min": vmin, "max": vmax, "step": step}
        except Exception as e:
            print(f"[WARN] axis meta error: {e}")
            return {"min": 0.0, "max": 1.0, "step": 0.1}

    loss_meta = compute_axis_meta(losses, is_loss=True)
    acc_meta = compute_axis_meta(accs, is_loss=False)

    # If matplotlib saved exact chart metadata, include it as well
    chart_meta = None
    try:
        meta_path = RESULTS / 'chart_meta.json'
        if meta_path.exists():
            with open(meta_path, 'r') as mf:
                chart_meta = json.load(mf)
    except Exception:
        chart_meta = None

    payload = {
        "loss": losses,
        "acc": accs,
        "progress": TRAIN_STATUS["progress"],
        "mode": TRAIN_STATUS.get("mode", "dataset"),
        "loss_meta": loss_meta,
        "acc_meta": acc_meta
    }

    if chart_meta:
        payload['chart_meta'] = chart_meta

    return jsonify(payload)

@app.route("/summary")
def summary_route():
    """Return training summary as JSON"""
    summary_path = RESULTS / "summary.txt"
    summary_text = ""
    
    if summary_path.exists():
        try:
            summary_text = summary_path.read_text()
        except:
            summary_text = "Error reading summary"
    
    return jsonify({"summary": summary_text})


# --------------------
# Chat History Helper Functions
# --------------------
def save_chat_message(user_msg, assistant_msg, sample_idx=None):
    """Save a chat message pair to CSV history file."""
    import csv
    try:
        os.makedirs(RESULTS, exist_ok=True)
        timestamp = datetime.datetime.now().isoformat()
        
        # Append to CSV
        file_exists = CHAT_HISTORY_FILE.exists()
        with open(CHAT_HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'User Message', 'Assistant Response', 'Sample Index'])
            writer.writerow([timestamp, user_msg, assistant_msg, sample_idx or ''])
    except Exception as e:
        print(f"[WARN] Could not save chat history: {e}")


def get_chat_history():
    """Load chat history from CSV file."""
    import csv
    try:
        if not CHAT_HISTORY_FILE.exists():
            return []
        
        history = []
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(row)
        return history
    except Exception as e:
        print(f"[WARN] Could not load chat history: {e}")
        return []


@app.route('/ai_chat')
def ai_chat_page():
    """Render clean AI chat page."""
    pred_grid = (RESULTS / "predictions_grid.png").exists()
    return render_template('ai_chat.html', pred_grid=pred_grid, v=int(time.time()))


@app.route('/ai_chat/history', methods=['GET'])
def ai_chat_history():
    """Return chat history as JSON."""
    history = get_chat_history()
    return jsonify({'ok': True, 'history': history})


@app.route('/ai_chat/save_message', methods=['POST'])
def ai_chat_save_message():
    """Save a chat message pair to history."""
    data = request.get_json() or {}
    user_msg = data.get('user_message', '').strip()
    assistant_msg = data.get('assistant_message', '').strip()
    sample_idx = data.get('sample_index')
    
    if user_msg and assistant_msg:
        save_chat_message(user_msg, assistant_msg, sample_idx)
        return jsonify({'ok': True})
    
    return jsonify({'ok': False, 'error': 'Missing messages'}), 400


@app.route('/ai_chat/meta')
def ai_chat_meta():
    """Return predictions grid metadata if available."""
    meta_path = RESULTS / 'predictions_grid_meta.json'
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
            return jsonify({'ok': True, 'meta': data})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    return jsonify({'ok': False, 'error': 'No meta found'})


@app.route('/ai_chat/ask', methods=['POST'])
def ai_chat_ask():
    """A simple rule-based assistant that returns explanations about a selected sample or general project info.

    This is a lightweight on-server helper (not a full LLM). It uses class descriptions and simple heuristics.
    """
    import re
    data = request.get_json() or {}
    msg = (data.get('message') or '').strip().lower()
    sample_idx = data.get('sample_index')

    # Try to extract sample index from message like "describe sample 2" or "sample #3"
    sample_match = re.search(r'sample[\s#]*(\d+)', msg)
    if sample_match and not sample_idx:
        try:
            sample_idx = int(sample_match.group(1))
        except:
            pass

    # small descriptions for CIFAR-10 classes
    class_desc = {
        'airplane': 'A fixed-wing aircraft designed for air transport of passengers or cargo.',
        'automobile': 'A road vehicle, typically with four wheels, powered by an internal combustion engine or electric motor.',
        'bird': 'A warm-blooded egg-laying vertebrate characterized by feathers and wings.',
        'cat': 'A small carnivorous mammal often kept as a pet; agile and with keen senses.',
        'deer': 'A hoofed grazing animal with antlers (in males) found in many parts of the world.',
        'dog': 'A domesticated carnivore, often used as a pet or working animal; loyal and social.',
        'frog': 'A small tailless amphibian with a short body, protruding eyes and strong, webbed hind feet.',
        'horse': 'A large solid-hoofed herbivorous mammal used for riding and to carry loads.',
        'ship': 'A large seafaring vessel used for transporting people or goods across water.',
        'truck': 'A motor vehicle designed to transport cargo, larger than a car.'
    }

    # If a sample index is provided, try to load meta
    sample_info = None
    try:
        meta_path = RESULTS / 'predictions_grid_meta.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            cells = meta.get('cells', [])
            if isinstance(sample_idx, int) and 0 <= sample_idx < len(cells):
                sample_info = cells[sample_idx]
    except Exception:
        sample_info = None

    # Simple heuristics for replies
    if sample_info and ('why' in msg or 'misclass' in msg or 'wrong' in msg or 'why predicted' in msg):
        true = sample_info.get('true')
        pred = sample_info.get('pred')
        if true == pred:
            reply = f"The model predicted '{pred}' and that matches the true label. It likely learned this class well from the training data."
        else:
            # Explain common causes
            reply = (
                f"The model predicted '{pred}' but the true label is '{true}'. Common reasons: the input image is low-resolution or noisy, "
                "CIFAR-10 classes are small (32x32) so high-frequency texture is lost, or the two classes have visual similarity causing confusion. "
                "To improve, consider more training data, stronger augmentations, or fine-tuning on images similar to your deployment photos."
            )
        # add short class descriptions
        reply += f"\n\nClass info — True: {class_desc.get(true, true)}; Predicted: {class_desc.get(pred, pred)}."
        return jsonify({'ok': True, 'reply': reply})

    # General project questions
    if 'what' in msg and 'project' in msg:
        reply = (
            "This project fine-tunes a ResNet18 model on CIFAR-10 and provides a Flask dashboard. "
            "You can upload images for prediction, run training on the CIFAR dataset or custom uploads, "
            "and view live charts. The dashboard also exposes sample predictions and simple explanations."
        )
        return jsonify({'ok': True, 'reply': reply})

    if 'describe' in msg or 'tell me about' in msg or 'explain' in msg:
        # If a sample is selected, describe classes; otherwise give a short project overview
        if sample_info:
            true = sample_info.get('true')
            pred = sample_info.get('pred')
            reply = (
                f"Sample {sample_info.get('index')}: true label '{true}'. Predicted '{pred}'.\n\n"
                f"True class description: {class_desc.get(true, true)}\n"
                f"Predicted class description: {class_desc.get(pred, pred)}\n\n"
                "If you want more detail, ask why it might be confused or how to improve accuracy."
            )
            return jsonify({'ok': True, 'reply': reply})

    # Fallback reply
    fallback = (
        "I can explain samples in the predictions grid if you select one, or provide a short project overview. "
        "Try: 'Describe sample 2' or 'Why was sample 3 misclassified?'"
    )
    return jsonify({'ok': True, 'reply': fallback})


@app.route('/ai_chat/stream', methods=['POST'])
def ai_chat_stream():
    """Stream assistant response from Groq (server-side) to the client.

    Requires environment variable `GROQ_API_KEY`. Handles any question in real-time.
    """
    import re
    data = request.get_json() or {}
    msg = (data.get('message') or '').strip()
    sample_idx = data.get('sample_index')

    # Try to extract sample index from message like "describe sample 2" or "sample #3"
    if not sample_idx:
        sample_match = re.search(r'sample[\s#]*(\d+)', msg, re.IGNORECASE)
        if sample_match:
            try:
                sample_idx = int(sample_match.group(1))
            except:
                pass

    # Get Groq API key
    GROQ_KEY = os.environ.get('GROQ_API_KEY')
    if not GROQ_KEY:
        # Fallback response when no API key
        def generate_fallback():
            yield "Hi! I'm Astra, your AI assistant.\n\n"
            yield "To enable full AI chat with real-time answers to ANY question, set up GROQ_API_KEY.\n\n"
            yield "Quick Setup (FREE):\n"
            yield "1. Visit https://console.groq.com/\n"
            yield "2. Sign up and create an API key\n"
            yield "3. Set environment variable:\n"
            yield "   PowerShell: $env:GROQ_API_KEY='your-key-here'\n"
            yield "   Then restart your app\n\n"
            yield "Once configured, I can answer ANY question - deep learning, coding, math, you name it!\n\n"
            yield "For now, explore the predictions grid above and check the other features!"
        return Response(generate_fallback(), mimetype='text/plain; charset=utf-8')

    # Build system and user messages for the model
    system_prompt = (
        "You are Astra, a highly capable AI assistant. You can:\n"
        "- Explain image classification, computer vision, deep learning concepts\n"
        "- Answer technical questions about PyTorch, CNNs, training, ML/AI\n"
        "- Provide code examples and debugging help\n"
        "- Answer general questions on ANY topic\n"
        "- Discuss model predictions and suggest improvements\n\n"
        "Be conversational, friendly, and informative. You can answer ANY question, not just about this project. "
        "Do NOT use emojis - only use plain text. When discussing specific samples from the predictions grid, provide concrete explanations."
    )

    # Try to include sample info if available
    sample_context = ''
    try:
        meta_path = RESULTS / 'predictions_grid_meta.json'
        if sample_idx is not None and meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            cells = meta.get('cells', [])
            if isinstance(sample_idx, int) and 0 <= sample_idx < len(cells):
                si = cells[sample_idx]
                sample_context = (
                    f"\n\n[CONTEXT: User selected Sample #{sample_idx}. "
                    f"True label: '{si.get('true')}', Model predicted: '{si.get('pred')}'. "
                    f"Use this info if their question relates to this sample.]"
                )
    except Exception as e:
        pass

    user_prompt = msg + sample_context

    # Call Groq's chat completion streaming API
    import requests

    headers = {
        'Authorization': f'Bearer {GROQ_KEY}',
        'Content-Type': 'application/json'
    }

    payload = {
        'model': 'llama-3.3-70b-versatile',
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'temperature': 0.7,
        'stream': True,
        'max_tokens': 2048
    }

    # Stream response and yield as plain chunks
    def generate():
        try:
            with requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=payload, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    yield f"Error: Groq API returned status {r.status_code}.\n\n"
                    yield "Please check your API key at https://console.groq.com/\n"
                    return
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        # Groq sends lines like: data: {json}
                        if line.startswith('data:'):
                            line = line[len('data:'):].strip()
                        if line == '[DONE]':
                            break
                        try:
                            j = json.loads(line)
                            # delta may be in choices[0].delta.content
                            delta = j.get('choices', [])[0].get('delta', {})
                            text = delta.get('content')
                            if text:
                                # Remove non-ASCII characters (keeps letters, numbers, punctuation, newlines)
                                text = ''.join(char for char in text if ord(char) < 128 or char in '\n\r\t')
                                if text.strip():  # Only yield if there's text left
                                    yield text
                        except Exception:
                            pass
        except Exception as e:
            yield f"\n\nError: {str(e)}\n"
            yield "Check your internet connection and API key."

    return Response(generate(), mimetype='text/plain; charset=utf-8')

# --------------------
# PDF Report
# --------------------
@app.route("/download_report")
def download_report():
    pdf_path = REPORTS / "EEGVision_Report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(160, 750, "EEGVision - Project Report")
    c.setFont("Helvetica", 11)
    c.drawString(50, 730, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 712, f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    c.drawString(50, 690, "Recent Predictions:")
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    y = 670
    for rec in history[:8]:
        c.drawString(60, y, f"{rec['timestamp']} - {rec['filename']} -> {rec['prediction']} ({rec['confidence']})")
        y -= 16
        if y < 80:
            c.showPage()
            y = 750
    csv_path = RESULTS / "training_results.csv"
    c.drawString(50, y-10, "Training Metrics:")
    y -= 30
    if csv_path.exists():
        import csv
        with open(csv_path, "r") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                try:
                    if len(row) >= 3:
                        ep, loss, acc = row[0], row[1], row[2]
                        c.drawString(60, y, f"Epoch {ep}: loss={loss} acc={acc}%")
                    elif len(row) == 2:
                        ep, loss = row[0], row[1]
                        c.drawString(60, y, f"Epoch {ep}: loss={loss}")
                    y -= 14
                    if y < 80:
                        c.showPage()
                        y = 750
                except:
                    continue
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Generated by EEGVision — Pragathi Singh & Team")
    c.save()
    return send_file(str(pdf_path), as_attachment=True)

# --------------------
# Static file serving
# --------------------
@app.route("/results/<path:filename>")
def serve_result_file(filename):
    return send_file(str(RESULTS / filename))

@app.route("/uploads/<path:filename>")
def serve_upload_file(filename):
    return send_file(str(UPLOADS / filename))

# --------------------
# Run app
# --------------------
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
import os
