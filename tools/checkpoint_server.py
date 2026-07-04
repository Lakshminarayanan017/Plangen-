"""
checkpoint_server.py
====================
Run this on your Mac while training runs on Colab/Kaggle.
It receives checkpoint files sent by the training loop and saves them locally.

Usage:
------
  # Step 1: Run this server on your Mac terminal:
  cd "/Users/badhri.narayanans/Documents/Edits/PlanGen 2"
  python tools/checkpoint_server.py

  # Step 2: In a new terminal, expose port 5001 via ngrok:
  ngrok http 5001

  # Step 3: Copy the ngrok URL (e.g. https://abc123.ngrok-free.app)
  # Step 4: Set it in your Colab/Kaggle notebook before training:
  #   import os; os.environ['CHECKPOINT_SERVER_URL'] = 'https://abc123.ngrok-free.app'

  # Step 5: Run training — checkpoints auto-upload to your Mac after each epoch!

Saved files:
------------
  training_checkpoints/ar_checkpoint_epoch_01.pt
  training_checkpoints/ar_checkpoint_epoch_05.pt
  ...etc.
"""

import http.server
import os
import sys
import json
from datetime import datetime

SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training_checkpoints"
)
PORT = 5001


class CheckpointHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        """Health check endpoint."""
        if self.path == "/ping":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = {
                "status": "ok",
                "save_dir": SAVE_DIR,
                "files": sorted(os.listdir(SAVE_DIR)) if os.path.exists(SAVE_DIR) else [],
                "time": datetime.now().isoformat(),
            }
            self.wfile.write(json.dumps(status, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Receive checkpoint file and save to disk."""
        if self.path != "/checkpoint":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        epoch = self.headers.get("X-Epoch", "unknown")
        val_loss = self.headers.get("X-ValLoss", "")

        if content_length == 0:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"No data received")
            return

        os.makedirs(SAVE_DIR, exist_ok=True)

        # Save as epoch-specific file (keeps history)
        epoch_str = str(epoch).zfill(2)
        fname = f"ar_checkpoint_epoch_{epoch_str}.pt"
        fpath = os.path.join(SAVE_DIR, fname)

        # Also overwrite "latest" for easy resuming
        latest_path = os.path.join(SAVE_DIR, "ar_checkpoint_latest.pt")

        print(f"\n  📥 Receiving epoch {epoch} checkpoint ({content_length/1e6:.1f} MB)...", flush=True)

        received = 0
        try:
            with open(fpath, "wb") as f:
                while received < content_length:
                    chunk_size = min(65536, content_length - received)
                    chunk = self.rfile.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)

            # Copy to latest
            import shutil
            shutil.copy2(fpath, latest_path)

            val_str = f"  val_loss={val_loss}" if val_loss else ""
            print(f"  ✅ Saved: {fpath}{val_str}", flush=True)
            print(f"  ✅ Also saved as: {latest_path}", flush=True)

            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"Saved {fname}".encode())

        except Exception as e:
            print(f"  ❌ Save error: {e}", flush=True)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def log_message(self, format, *args):
        """Suppress noisy request logs."""
        pass


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("🏠 PlanGen Checkpoint Server")
    print("=" * 60)
    print(f"📁 Saving checkpoints to:")
    print(f"   {SAVE_DIR}")
    print()
    print(f"🌐 Server running on: http://0.0.0.0:{PORT}")
    print()
    print("📋 Next steps:")
    print("  1. Open a NEW terminal and run:")
    print("     ngrok http 5001")
    print()
    print("  2. Copy the ngrok URL (e.g. https://abc123.ngrok-free.app)")
    print()
    print("  3. In your Colab/Kaggle notebook, run this BEFORE training:")
    print("     import os")
    print("     os.environ['CHECKPOINT_SERVER_URL'] = 'https://abc123.ngrok-free.app'")
    print()
    print("  4. Start training — checkpoints auto-upload here after each epoch!")
    print()
    print("  Health check: curl http://localhost:5001/ping")
    print("=" * 60)
    print()

    server = http.server.HTTPServer(("0.0.0.0", PORT), CheckpointHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped.")
        files = sorted(os.listdir(SAVE_DIR)) if os.path.exists(SAVE_DIR) else []
        if files:
            print(f"\n📦 Saved checkpoints ({len(files)} files):")
            for f in files:
                fpath = os.path.join(SAVE_DIR, f)
                size_mb = os.path.getsize(fpath) / 1e6
                print(f"   {f}  ({size_mb:.1f} MB)")
        sys.exit(0)


if __name__ == "__main__":
    main()
