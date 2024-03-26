import argparse
from live_stream_processing import capture_and_process_webcam
from python_server import start_server
import threading
import queue

# For development use: send signals to Unity through command-line
def send_signals_from_cli(signal_queue, shutdown_event):
    print("Enter signals to send to Unity (type 'exit' to stop):")
    while not shutdown_event.is_set():
        sample_signal = input()  # Wait for user input
        if sample_signal.lower() == 'exit':
            shutdown_event.set()
        else:
            signal_queue.put(sample_signal)  # Add the entered signal to the queue

def main(use_unity=False, camera_number=0, dev_mode=False):
    # Threading event to signal server thread to stop
    shutdown_event = threading.Event()
    # Thread-safe queue is shared between webcam processing and server communication
    signal_queue = queue.Queue()

    if use_unity:
        # Start the server in a separate thread to handle Unity client
        server_thread = threading.Thread(target=start_server, args=('127.0.0.1', 25001, shutdown_event, signal_queue))
        server_thread.start()
        print("Unity connection established.")

    if dev_mode and use_unity: 
        # Start the CLI input handler in development mode
        cli_thread = threading.Thread(target=send_signals_from_cli, args=(signal_queue, shutdown_event))
        cli_thread.start()
        cli_thread.join() 
    else: 
        capture_and_process_webcam(camera_number, shutdown_event, signal_queue, use_unity)

    if use_unity:
        # Signal the server thread to stop and wait for it to finish
        server_thread.join()
        print("Unity connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process webcam feed and optionally communicate with Unity.")
    parser.add_argument('--use_unity', action='store_true', help='Use Unity connection for sending gestures.')
    parser.add_argument('--camera', type=int, default=0, help='Camera number to use for capturing video.')
    parser.add_argument('--dev_mode', action='store_true', help='Enable development mode for CLI signal input.')

    args = parser.parse_args()

    main(use_unity=args.use_unity, camera_number=args.camera, dev_mode=args.dev_mode)

