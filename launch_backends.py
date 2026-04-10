#launch_backends.py
import subprocess, sys, os, time

experiments = [
    ("exp_1", 5001, 1),  # ← last number is backend seed
    ("exp_2", 5002, 1),
    ("exp_3", 5003, 1),
    ("exp_4", 5004, 1),
]

procs = []
for exp_name, backend_port, seed in experiments:
    p = subprocess.Popen(
        [sys.executable, "server.py"],
        env={
            **os.environ,
            "ACTIVE_EXP": exp_name,
            "PORT":        str(backend_port),
            "SEED":        str(seed),
        }
    )
    procs.append(p)
    print(f"✅ Backend {exp_name} started on :{backend_port}")
    time.sleep(3)  # ← give each backend 3 seconds to fully initialise before starting the next

print("\n▶️  All backends running. Ctrl+C to stop.")
try:
    for p in procs: p.wait()
except KeyboardInterrupt:
    print("\n🛑 Shutting down backends...")
    for p in procs: p.terminate()