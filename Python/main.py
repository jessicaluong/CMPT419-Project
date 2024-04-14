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

def main(use_unity=False, camera_number=0, dev_mode=False, cli=False):
    """
    Main function to initialize the application based on provided arguments.

    This function handles the setup of threading events and queues, initiates server communication if
    the application is set to interface with Unity, and starts webcam processing. 

    Parameters:
        use_unity (bool): If True, establishes a server connection to send data to a Unity application.
        camera_number (int): Specifies the camera index to use for video capture.
        dev_mode (bool): If True, enables development mode which may display additional diagnostic
                         information on the output screen.
        cli (bool): If True, enables CLI-based interaction for sending signals manually, typically used for debugging.

    Behavior:
        - Initiates a server connection for Unity if `use_unity` is set to True.
        - Starts a separate thread for CLI interactions if both `cli` and `use_unity` are True.
        - Handles webcam video capture and processing in the main thread unless CLI mode is specifically requested.
        - Waits for threads to complete before closing, particularly the Unity server connection if established.
    """
    # Threading event to signal server thread to stop
    shutdown_event = threading.Event()
    # Thread-safe queue is shared between webcam processing and server communication
    signal_queue = queue.Queue()

    if use_unity:
        # Start the server in a separate thread to handle Unity client
        server_thread = threading.Thread(target=start_server, args=('127.0.0.1', 25001, shutdown_event, signal_queue))
        server_thread.start()
        print("Unity connection established.")

    if cli and use_unity: 
        # Start the CLI input handler in development mode
        cli_thread = threading.Thread(target=send_signals_from_cli, args=(signal_queue, shutdown_event))
        cli_thread.start()
        cli_thread.join() 
    else: 
        capture_and_process_webcam(camera_number, shutdown_event, signal_queue, use_unity, dev_mode)

    if use_unity:
        server_thread.join()
        print("Unity connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process webcam feed and optionally communicate with Unity.")
    parser.add_argument('--use_unity', action='store_true', help='Use Unity connection for sending gestures.')
    parser.add_argument('--camera', type=int, default=0, help='Camera number to use for capturing video.')
    parser.add_argument('--dev_mode', action='store_true', help='Displays landmarks and predictions to the screen.')
    parser.add_argument('--cli', action='store_true', help='Enable development mode for CLI signal input.')

    args = parser.parse_args()

    main(use_unity=args.use_unity, camera_number=args.camera, dev_mode=args.dev_mode, cli=args.cli)

