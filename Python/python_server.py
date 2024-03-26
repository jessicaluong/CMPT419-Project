import socket
import threading
import time

def handle_client(connection, address, signal_queue, shutdown_event):
    print(f"Unity client connected: {address}")
    try:
        while not shutdown_event.is_set():
            # print("queue empty")
            if not signal_queue.empty():
                signal = signal_queue.get()
                connection.sendall((signal + "\n").encode())
            else:
                # Sleep briefly to avoid busy waiting if the queue is empty
                time.sleep(0.01)
    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        print(f"Client disconnected with error: {e}")
    finally:
        connection.close()
        print("Client connection closed.")

def start_server(host, port, shutdown_event, signal_queue):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        server_socket.settimeout(1.0)  # Timeout for accepting connections

        print(f"Python server listening on {host}:{port}")

        while not shutdown_event.is_set():
            try: 
                client_sock, client_addr = server_socket.accept()
                # Run handle_client in separate thread, to maintain responsiveness and allow server to handle 'shutdown_event'
                client_thread = threading.Thread(target=handle_client, args=(client_sock, client_addr, signal_queue, shutdown_event))
                client_thread.start()
            except socket.timeout:
                continue  # Timeout occurred, loop back and check the shutdown event again
        print("Server shutdown initiated.")