using UnityEngine;
using System.Net.Sockets;
using System.IO;
using System.Threading;

public class UnityClient : MonoBehaviour
{
    public string host = "127.0.0.1";
    public int port = 25001;
    private TcpClient client;
    private StreamReader reader;
    private Thread clientThread;
    public RespondToSignals signalResponder;

    private void Awake()
    {
        // Access the Instance property to ensure it's created on the main thread
        var dispatcher = UnityMainThreadDispatcher.Instance;
    }

    void Start()
    {
        ConnectToServer(); 
    }

    void ConnectToServer()
    {
        // Receive data in separate thread to avoid block Unity main thread 
        ThreadStart ts = new ThreadStart(GetSignal); 
        clientThread = new Thread(ts);
        clientThread.Start();
    }

    private void GetSignal() 
    {
        client = new TcpClient(host, port); 
        Debug.Log("Connected to the server.");

        NetworkStream stream = client.GetStream(); 
        reader = new StreamReader(stream); 

        // Read data from the server in a loop
        while (client.Connected)
        {
            if (stream.DataAvailable)
            {
                string data = reader.ReadLine();
                Debug.Log("Received data: " + data);

                // Use Unity's main thread to call functions on GameObjects
                UnityMainThreadDispatcher.Instance.Enqueue(() => signalResponder.ReceiveSignal(data));
            }
        }
    }

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
