using System.Collections.Generic;
using UnityEngine;
using System;

/// <summary>
/// Allows actions to be executed on the main thread in Unity. Useful for performing operations that need
/// to interact with Unity's API, which requires them to be run from the main thread (e.g., updating UI elements).
/// Adapted from https://github.com/PimDeWitte/UnityMainThreadDispatcher/blob/master/Runtime/UnityMainThreadDispatcher.cs
/// </summary>
public class UnityMainThreadDispatcher : MonoBehaviour
{
    private readonly Queue<Action> queue = new Queue<Action>();

    private static UnityMainThreadDispatcher _instance;

    /// <summary>
    /// Singleton property that provides a global access point to the instance of the UnityMainThreadDispatcher.
    /// If no instance exists, it attempts to find an existing instance in the scene or creates a new one.
    /// </summary>
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

    /// <summary>
    /// Ensures that this object is a singleton in the scene. Destroys itself if another instance already exists.
    /// </summary>
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

    /// <summary>
    /// Adds an action to the queue to be executed on the main thread.
    /// </summary>
    /// <param name="action">The action to enqueue.</param>
    public void Enqueue(Action action)
    {
        lock (queue)
        {
            queue.Enqueue(action);
        }
    }

    /// <summary>
    /// Executes all enqueued actions on the main thread. 
    /// This method should be called only from the main thread (Unity does this automatically in the Update loop).
    /// </summary>
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
