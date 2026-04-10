# launch_frontends.py
import subprocess, os, time, webbrowser

NPX = r"C:\Program Files\nodejs\npx.cmd"

experiments = [
    ("exp_1", 5001, 5173, 1),  # ← last number is frontend seed
    ("exp_2", 5002, 5174, 1),
    ("exp_3", 5003, 5175, 1),
    ("exp_4", 5004, 5176, 1),
]

procs = []
for exp_name, backend_port, frontend_port, seed in experiments:
    p = subprocess.Popen(
        [NPX, "vite"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env={
            **os.environ,
            "VITE_PORT":         str(frontend_port),
            "VITE_BACKEND_PORT": str(backend_port),
        }
    )
    procs.append(p)
    print(f"✅ Frontend {exp_name} started on :{frontend_port} → backend :{backend_port}")

time.sleep(3)
for exp_name, _, frontend_port, seed in experiments:
    url = f"http://localhost:{frontend_port}/?seed={seed}"
    webbrowser.open(url)
    print(f"🌐 Opened: {url}")

print("\n▶️  All frontends running. Ctrl+C to stop.")
try:
    for p in procs: p.wait()
except KeyboardInterrupt:
    print("\n🛑 Shutting down frontends...")
    for p in procs: p.terminate()

