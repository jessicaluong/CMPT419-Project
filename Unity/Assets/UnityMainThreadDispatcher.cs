using System.Collections.Generic;
using UnityEngine;
using System;

public class UnityMainThreadDispatcher : MonoBehaviour
{
    private readonly Queue<Action> queue = new Queue<Action>();

    private static UnityMainThreadDispatcher _instance;
    public static UnityMainThreadDispatcher Instance
    {
        get
        {
            if (_instance == null)
            {
                // Try to find an existing instance in the scene
                _instance = FindObjectOfType<UnityMainThreadDispatcher>();
                if (_instance == null)
                {
                    // Create a new GameObject and add this component
                    GameObject dispatcherObject = new GameObject("UnityMainThreadDispatcher");
                    _instance = dispatcherObject.AddComponent<UnityMainThreadDispatcher>();
                    DontDestroyOnLoad(dispatcherObject); 
                }
            }
            return _instance;
        }
    }

    private void Awake()
    {
        if (_instance == null)
        {
            _instance = this;
            DontDestroyOnLoad(gameObject); 
        }
        else if (_instance != this)
        {
            Destroy(gameObject);
        }
    }

    public void Enqueue(Action action)
    {
        lock (queue)
        {
            queue.Enqueue(action);
        }
    }

    void Update()
    {
        while (queue.Count > 0)
        {
            Action action = null;
            lock (queue)
            {
                if (queue.Count > 0)
                {
                    action = queue.Dequeue();
                }
            }

            action?.Invoke();
        }
    }

}
