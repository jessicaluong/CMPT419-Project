import socket
import threading
import time

def handle_client(connection, address, signal_queue, shutdown_event):
    """
    Manages communication with a connected client.

    This function continuously checks for signals in the `signal_queue` to send to the client,
    and maintains the connection open until the `shutdown_event` is set. If the queue is empty,
    it briefly sleeps to prevent busy waiting.
    """
    print(f"Unity client connected: {address}")
    try:
        while not shutdown_event.is_set():
            if not signal_queue.empty():
                signal = signal_queue.get()
                connection.sendall((signal + "\n").encode())
            else:
                time.sleep(0.01) # Sleep briefly to avoid busy waiting if the queue is empty
    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        print(f"Client disconnected with error: {e}")
    finally:
        connection.close()
        print("Client connection closed.")

def start_server(host, port, shutdown_event, signal_queue):
    """
    Starts a server that listens for incoming client connections on a specific port.

    The server runs in a loop, accepting new client connections and starting a new thread for
    each client using the `handle_client` function. It handles connections until the `shutdown_event`
    is set.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        server_socket.settimeout(1.0)  # Timeout for accepting connections

        print(f"Python server listening on {host}:{port}")

        while not shutdown_event.is_set():
            try: 
                client_sock, client_addr = server_socket.accept()
                client_thread = threading.Thread(target=handle_client, 
                                                 args=(client_sock, client_addr, signal_queue, shutdown_event))
                client_thread.start()
            except socket.timeout:
                continue  # Timeout occurred, loop back and check the shutdown event again
        print("Server shutdown initiated.")