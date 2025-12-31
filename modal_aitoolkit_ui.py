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
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "cd /root && git clone https://github.com/ostris/ai-toolkit.git",
        "cd /root/ai-toolkit && git submodule update --init --recursive",
    )
    .pip_install(
        "torch==2.8.0+cu128",
        "torchvision==0.23.0+cu128",
        index_url="https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        # Install Python dependencies
        "cd /root/ai-toolkit && pip install -r requirements.txt",
        # Build UI (npm install + build di folder ui)
        "cd /root/ai-toolkit/ui && npm install",
        "cd /root/ai-toolkit/ui && npm run build",
    )
)

app = modal.App("ai-toolkit-ui")

# Volumes untuk persist data - mount ke path kosong
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
    timeout=86400,  # 24 jam
    volumes=volumes
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8675, startup_timeout=180)
def ui_server():
    """
    Jalankan AI Toolkit UI di port 8675
    """
    import subprocess
    import os
    
    # Buat symlink dari volume mount ke lokasi yang diharapkan AI Toolkit
    os.makedirs("/root/ai-toolkit/output", exist_ok=True)
    os.makedirs("/root/.cache", exist_ok=True)
    
    # Symlink volume mounts
    subprocess.run(["ln", "-sf", "/mnt/output", "/root/ai-toolkit/output"], check=False)
    subprocess.run(["ln", "-sf", "/mnt/cache", "/root/.cache"], check=False)
    
    # Set working directory ke folder ui
    os.chdir("/root/ai-toolkit/ui")
    
    # Set environment variables
    env = os.environ.copy()
    
    # Optional: Set auth token untuk keamanan
    # Uncomment baris ini dan ganti dengan password yang kuat
    # env["AI_TOOLKIT_AUTH"] = "your_secure_password_here"
    
    # Start UI server
    # Karena sudah di build, gunakan npm run start
    subprocess.Popen(
        ["npm", "run", "start"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# Helper function untuk upload files
@app.function(
    image=image,
    volumes=volumes,
    timeout=3600,
)
def upload_files(local_path: str, remote_path: str):
    """
    Upload dataset atau files ke volume
    
    Usage:
    modal run modal_aitoolkit_ui.py::upload_files \
      --local-path ./my_images \
      --remote-path dataset/my_project
    """
    import shutil
    from pathlib import Path
    
    src = Path(local_path)
    dst = Path(f"/mnt/output/{remote_path}")
    
    if not src.exists():
        raise FileNotFoundError(f"Local path tidak ditemukan: {local_path}")
    
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"✓ Uploaded directory: {local_path} -> {dst}")
        
        # Hitung jumlah files
        file_count = len(list(dst.rglob("*.*")))
        print(f"  Total files: {file_count}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"✓ Uploaded file: {local_path} -> {dst}")


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


# Function untuk list files di volume
@app.function(
    image=image,
    volumes=volumes,
)
def list_files(path: str = ""):
    """
    List files di output volume
    
    Usage:
    modal run modal_aitoolkit_ui.py::list_files --path dataset
    """
    from pathlib import Path
    
    base = Path(f"/mnt/output/{path}")
    
    if not base.exists():
        print(f"Path tidak ditemukan: {path}")
        return
    
    print(f"\n📁 Files di: output/{path}\n")
    print("-" * 70)
    
    if base.is_file():
        size = base.stat().st_size / (1024 * 1024)  # MB
        print(f"  {base.name} ({size:.2f} MB)")
    else:
        for item in sorted(base.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(base)
                size = item.stat().st_size / (1024 * 1024)  # MB
                print(f"  {rel_path} ({size:.2f} MB)")
    
    print("-" * 70)


@app.local_entrypoint()
def main():
    """
    Deploy AI Toolkit UI ke Modal
    """
    print("\n" + "="*70)
    print("🚀 Ostris AI Toolkit UI on Modal.com")
    print("="*70)
    
    print("\n📦 GPU & Resources:")
    print("  - GPU: A10G (24GB VRAM)")
    print("  - UI Port: 8675")
    print("  - Timeout: 24 hours")
    print("  - Volumes: aitoolkit-output, aitoolkit-cache")
    
    print("\n🔧 Deployment Commands:")
    print("  Deploy:    modal deploy modal_aitoolkit_ui.py")
    print("  Dev mode:  modal serve modal_aitoolkit_ui.py")
    
    print("\n📤 Upload Dataset:")
    print("  modal run modal_aitoolkit_ui.py::upload_files \\")
    print("    --local-path ./my_images \\")
    print("    --remote-path dataset/project1")
    
    print("\n📥 Download Results:")
    print("  modal run modal_aitoolkit_ui.py::download_files \\")
    print("    --remote-path my_lora/checkpoint.safetensors \\")
    print("    --local-path ./my_lora.safetensors")
    
    print("\n📋 List Files:")
    print("  modal run modal_aitoolkit_ui.py::list_files --path dataset")
    
    print("\n🔐 Security Setup (Recommended):")
    print("  Uncomment AI_TOOLKIT_AUTH di fungsi ui_server() untuk set password")
    print("  env['AI_TOOLKIT_AUTH'] = 'your_secure_password'")
    
    print("\n💡 Important Notes:")
    print("  - FLUX training membutuhkan minimal 24GB VRAM")
    print("  - Training jobs berjalan di background (UI bisa ditutup)")
    print("  - Semua output tersimpan di Modal Volume")
    print("  - Untuk Hugging Face models, set HF_TOKEN via Modal secrets:")
    print("    modal secret create huggingface HF_TOKEN=your_token")
    
    print("\n🌐 Access URL:")
    print("  Setelah deploy, Modal akan memberikan URL seperti:")
    print("  https://username--ai-toolkit-ui-ui-server.modal.run")
    
    print("\n" + "="*70 + "\n")