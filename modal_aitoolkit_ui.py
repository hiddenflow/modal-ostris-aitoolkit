import modal

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "cudnn-devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "ffmpeg",
        "libsm6",
        "libxext6"
    )
    .run_commands(
        # Install Node.js 20
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        # Clone AI Toolkit
        "cd /root && git clone https://github.com/ostris/ai-toolkit.git",
        "cd /root/ai-toolkit && git submodule update --init --recursive",
    )
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0+cu128",
        index_url="https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        # Install Python dependencies
        "cd /root/ai-toolkit && pip install -r requirements.txt",
        # Install UI dependencies dan build SEKALI saja saat build image
        "cd /root/ai-toolkit/ui && npm install --legacy-peer-deps",
        # Setup Prisma database
        "cd /root/ai-toolkit/ui && npx prisma generate",
        "cd /root/ai-toolkit/ui && npx prisma db push",
        # Build UI
        "cd /root/ai-toolkit/ui && npm run build",
    )
)

app = modal.App("ai-toolkit-ui")

# Volumes untuk persist data
volumes = {
    "/mnt/output": modal.Volume.from_name(
        "aitoolkit-output", 
        create_if_missing=True
    ),
    "/mnt/cache": modal.Volume.from_name(
        "aitoolkit-cache", 
        create_if_missing=True
    ),
}

@app.function(
    image=image,
    gpu="L40S",  # Minimal 24GB VRAM untuk FLUX training
    timeout=3600,  # 24 jam
    volumes=volumes,
    allow_concurrent_inputs=100,
)
@modal.web_server(8675, startup_timeout=300)
def ui_server():
    """
    Jalankan AI Toolkit Next.js UI (dengan API routes built-in)
    """
    import subprocess
    import os
    
    # Setup symlinks untuk volumes
    subprocess.run(["rm", "-rf", "/root/ai-toolkit/output"], check=False)
    subprocess.run(["rm", "-rf", "/root/.cache"], check=False)
    subprocess.run(["ln", "-sf", "/mnt/output", "/root/ai-toolkit/output"], check=True)
    subprocess.run(["ln", "-sf", "/mnt/cache", "/root/.cache"], check=True)
    
    # Buat directories yang diperlukan
    os.makedirs("/mnt/output/datasets", exist_ok=True)
    os.makedirs("/mnt/output/jobs", exist_ok=True)
    os.makedirs("/mnt/output/models", exist_ok=True)
    os.makedirs("/mnt/cache", exist_ok=True)
    
    # Set permissions
    subprocess.run(["chmod", "-R", "777", "/mnt/output"], check=False)
    subprocess.run(["chmod", "-R", "777", "/mnt/cache"], check=False)
    
    # Setup Prisma database symlink (database juga harus persist)
    db_dir = "/mnt/output/database"
    os.makedirs(db_dir, exist_ok=True)
    subprocess.run(["rm", "-rf", "/root/ai-toolkit/ui/prisma/dev.db"], check=False)
    subprocess.run(["rm", "-rf", "/root/ai-toolkit/ui/prisma/dev.db-journal"], check=False)
    
    # Jika database belum ada, inisialisasi
    if not os.path.exists(f"{db_dir}/dev.db"):
        print("Initializing database...")
        subprocess.run([
            "npx", "prisma", "db", "push"
        ], cwd="/root/ai-toolkit/ui", env=os.environ, check=False)
        # Copy ke volume
        subprocess.run([
            "cp", "/root/ai-toolkit/ui/prisma/dev.db", f"{db_dir}/dev.db"
        ], check=False)
    
    # Link database dari volume
    subprocess.run([
        "ln", "-sf", f"{db_dir}/dev.db", "/root/ai-toolkit/ui/prisma/dev.db"
    ], check=True)
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NODE_ENV"] = "production"
    env["OUTPUT_DIR"] = "/root/ai-toolkit/output"
    env["PYTHONPATH"] = "/root/ai-toolkit"
    env["PORT"] = "8675"
    env["DATABASE_URL"] = "file:./prisma/dev.db"
    
    # Optional: Set auth token untuk keamanan
    # Uncomment baris ini dan ganti dengan password yang kuat
    # env["AI_TOOLKIT_AUTH"] = "your_secure_password_here"
    
    print("=" * 70)
    print("Starting AI Toolkit UI...")
    print("Output directory:", env["OUTPUT_DIR"])
    print("Database:", env["DATABASE_URL"])
    print("UI will be available on port 8675")
    print("=" * 70)
    
    # Change to UI directory dan jalankan 'npm run start'
    # Next.js akan handle API routes secara otomatis
    os.chdir("/root/ai-toolkit/ui")
    
    subprocess.Popen(
        ["npm", "run", "start"],
        env=env,
        cwd="/root/ai-toolkit/ui",
    )

# Helper function untuk download results
@app.function(
    image=image,
    volumes=volumes,
    timeout=3600,
)
def download_files(remote_path: str, local_path: str):
    """
    Download trained models atau output files
    
    Usage:
    modal run modal_aitoolkit_ui.py::download_files \
      --remote-path my_lora/my_lora_000001000.safetensors \
      --local-path ./trained_models/my_lora.safetensors
    """
    import shutil
    from pathlib import Path
    
    src = Path(f"/mnt/output/{remote_path}")
    dst = Path(local_path)
    
    if not src.exists():
        raise FileNotFoundError(f"Remote path tidak ditemukan: {remote_path}")
    
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"✓ Downloaded directory: {src} -> {local_path}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"✓ Downloaded file: {src} -> {local_path}")

# Function untuk check dan setup directories
@app.function(
    image=image,
    volumes=volumes,
)
def setup_directories():
    """
    Setup dan verify directory structure
    
    Usage:
    modal run modal_aitoolkit_ui.py::setup_directories
    """
    import os
    from pathlib import Path
    
    directories = [
        "/mnt/output/datasets",
        "/mnt/output/jobs", 
        "/mnt/output/models",
        "/mnt/cache"
    ]
    
    print("\n🔧 Setting up directories...\n")
    print("-" * 70)
    
    for dir_path in directories:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Set permissions
        os.chmod(dir_path, 0o777)
        
        # Verify
        if path.exists():
            print(f"✓ {dir_path} - OK")
        else:
            print(f"✗ {dir_path} - FAILED")
    
    print("-" * 70)
    print("✓ Directory setup complete!\n")

@app.local_entrypoint()
def main():
    """
    Deploy AI Toolkit UI ke Modal
    """
    print("\n" + "="*70)
    print("🚀 Ostris AI Toolkit UI on Modal.com")
    print("="*70)
    
    print("\n📦 GPU & Resources:")
    print("  - GPU: L40S (48GB VRAM)")
    print("  - Port: 8675 (UI + API)")
    print("  - Timeout: 1 hours")
    print("  - Volumes: aitoolkit-output, aitoolkit-cache")
    
    print("\n🔧 Deployment Commands:")
    print("  Deploy:    modal deploy modal_aitoolkit_ui.py")
    print("  Dev mode:  modal serve modal_aitoolkit_ui.py")
    
    print("\n📥 Download Results:")
    print("  modal run modal_aitoolkit_ui.py::download_files \\")
    print("    --remote-path my_lora/checkpoint.safetensors \\")
    print("    --local-path ./my_lora.safetensors")
    
    print("\n🔧 Setup Directories (jalankan setelah deploy):")
    print("  modal run modal_aitoolkit_ui.py::setup_directories")
    
    print("\n🔐 Security Setup (Recommended):")
    print("  Uncomment AI_TOOLKIT_AUTH di fungsi ui_server() untuk set password")
    print("  env['AI_TOOLKIT_AUTH'] = 'your_secure_password'")
    
    print("\n💡 Important Notes:")
    print("  - UI sudah di-build saat create image (no runtime rebuild)")
    print("  - npm run start = production server tanpa rebuild")
    print("  - FLUX training membutuhkan minimal 24GB VRAM")
    print("  - Training jobs berjalan di background")
    print("  - Semua output tersimpan di Modal Volume")
    
    print("\n🔑 Hugging Face Setup (Optional):")
    print("  Untuk FLUX.1-dev, set HF token sebagai Modal secret:")
    print("  modal secret create huggingface HF_TOKEN=your_token")
    
    print("\n🌐 Access URL:")
    print("  Setelah deploy, Modal akan memberikan URL seperti:")
    print("  https://username--ai-toolkit-ui-ui-server.modal.run")
    
    print("\n" + "="*70 + "\n")
    
    print("\n" + "="*70 + "\n")
