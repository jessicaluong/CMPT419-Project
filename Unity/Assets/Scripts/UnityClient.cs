using UnityEngine;
using System.Net.Sockets;
using System.IO;
using System.Threading;

/// <summary>
/// Handles network communication by connecting to a server and processing incoming signals.
/// </summary>
public class UnityClient : MonoBehaviour
{
    public string host = "127.0.0.1";
    public int port = 25001;
    private TcpClient client;
    private StreamReader reader;
    private Thread clientThread;
    public RespondToSignals signalResponder;

    /// <summary>
    /// Called when the script instance is being loaded.
    /// Ensures that the UnityMainThreadDispatcher instance is created on the main thread.
    /// </summary>
    private void Awake()
    {
        var dispatcher = UnityMainThreadDispatcher.Instance;
    }

    /// <summary>
    /// Called on the frame when a script is enabled just before any of the Update methods are called the first time.
    /// Starts the connection to the server.
    /// </summary>
    void Start()
    {
        ConnectToServer(); 
    }

    /// <summary>
    /// Initiates the connection to the server by starting a new thread (to avoid blocking main thread) 
    /// that handles the server communication.
    /// </summary>
    void ConnectToServer()
    {
        ThreadStart ts = new ThreadStart(GetSignal); 
        clientThread = new Thread(ts);
        clientThread.Start();
    }

    /// <summary>
    /// Connects to the server and continuously reads incoming data.
    /// If data is available, it processes it using the main thread dispatcher.
    /// </summary>
    private void GetSignal() 
    {
        client = new TcpClient(host, port); 
        Debug.Log("Connected to the server.");

        NetworkStream stream = client.GetStream(); 
        reader = new StreamReader(stream); 

        // Read data continuously from the server while the connection is active
        while (client.Connected)
        {
            if (stream.DataAvailable)
            {
                string data = reader.ReadLine();
                Debug.Log("Received data: " + data);

                // Enqueue the received data for processing on the main thread to avoid conflicts with Unity API calls
                UnityMainThreadDispatcher.Instance.Enqueue(() => signalResponder.ReceiveSignal(data));
            }
        }
    }

    /// <summary>
    /// Called when the application quits. Releases network resources. 
    /// </summary>
    void OnApplicationQuit()
    {
        if (client != null)
        {
            client.Close();
        }
        if (clientThread != null)
        {
            clientThread.Abort();
        }
        Debug.Log("Disconnected from the server.");
    }
}
