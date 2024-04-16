using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// This class is responsible for responding to different signals (either through TCP connections or keyboard inputs),
/// changing animations and visual states of the character based on those signals.
/// </summary>
public class RespondToSignals : MonoBehaviour
{
    Animator animator;
    Renderer[] characterMaterials;

    public enum EyePosition { normal, happy, angry, dead }
    public EyePosition eyeState;

    private Vector3 originalPosition; // Original position to reset the character to

    public bool useTcpConnection = true; // Flag to toggle between TCP connection and keyboard input for development purposes

    Coroutine returnToIdleCoroutine;  

    /// <summary>
    /// Start is called before the first frame update. Initialize component references and set initial states.
    /// </summary>
    public void Start()
    {
        animator = GetComponent<Animator>();
        characterMaterials = GetComponentsInChildren<Renderer>();

        originalPosition = transform.position;
    }

    /// <summary>
    /// Update is called once per frame. Handles keyboard inputs to manually trigger animations and state changes.
    /// </summary>
    void Update()
    {
        if (!useTcpConnection)
        {
            HandleKeyboardInputs();
        }
    }

    /// <summary>
    /// Processes incoming signals over TCP to trigger animations and state changes.
    /// </summary>
    /// <param name="signal">The signal received from TCP communication.</param>
    public void ReceiveSignal(string signal)
    {
        if (useTcpConnection)
        {
            // Correctly stop the previous coroutine if it's running
            if (returnToIdleCoroutine != null)
            {
                StopCoroutine(returnToIdleCoroutine);
                returnToIdleCoroutine = null;
            }

            ProcessSignal(signal);  // Process the incoming signal.

            // Restart the coroutine with a new reference
            returnToIdleCoroutine = StartCoroutine(ReturnToIdle());
        }
    }


    /// <summary>
    /// Coroutine to delay before returning to the idle state, resetting position and orientation.
    /// </summary>
    /// <returns>IEnumerator for coroutine handling.</returns>
    IEnumerator ReturnToIdle()
    {
        // Wait for a specified time before returning to idle
        Debug.Log("Starting ReturnToIdle coroutine.");
        yield return new WaitForSeconds(5);
        Debug.Log("Completed 5-second wait.");

        ResetPosition();
        ResetOrientation();

        // Transition back to the idle state
        ChangeEyeOffset(EyePosition.normal);
        ChangeAnimatorIdle("idle");
        Debug.Log("ReturnToIdle completed.");
    }

    /// <summary>
    /// Resets the agent's orientation to a predefined rotation.
    /// </summary>
    void ResetOrientation()
    {
        transform.rotation = Quaternion.Euler(new Vector3(-0.05f, 20.054f, -0.018f));
    }

    /// <summary>
    /// Resets the agent's position to the original location.
    /// </summary>
    void ResetPosition()
    {
        transform.position = originalPosition;
    }

    /// <summary>
    /// Changes the animation state of the character based on the provided trigger.
    /// </summary>
    /// <param name="trigger">The animation trigger to set.</param>
    void ChangeAnimatorIdle(string trigger)
    {
        animator.SetTrigger(trigger);
    }

    /// <summary>
    /// Changes the texture offset of the eyes to simulate different eye positions.
    /// </summary>
    /// <param name="pos">The new eye position.</param>
    void ChangeEyeOffset(EyePosition pos)
    {
        Vector2 offset = Vector2.zero;

        switch (pos)
        {
            case EyePosition.normal:
                offset = new Vector2(0, 0);
                break;
            case EyePosition.happy:
                offset = new Vector2(.33f, 0);
                break;
            case EyePosition.angry:
                offset = new Vector2(.66f, 0);
                break;
            case EyePosition.dead:
                offset = new Vector2(.33f, .66f);
                break;
            default:
                break;
        }

        for (int i = 0; i < characterMaterials.Length; i++)
        {
            if (characterMaterials[i].transform.CompareTag("PlayerEyes"))
                characterMaterials[i].material.SetTextureOffset("_MainTex", offset);
        }
    }


    /// <summary>
    /// Handles keyboard inputs for development purposes, allowing manual control over character state changes.
    /// </summary>
    private void HandleKeyboardInputs()
    {
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            ChangeEyeOffset(EyePosition.normal);
            ChangeAnimatorIdle("idle");
            ResetOrientation();
            ResetPosition();
        }
        if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            ChangeEyeOffset(EyePosition.normal);
            ChangeAnimatorIdle("point");
        }
        if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            ChangeEyeOffset(EyePosition.happy);
            ChangeAnimatorIdle("head_nod");
        }
        if (Input.GetKeyDown(KeyCode.Alpha4))
        {
            ChangeEyeOffset(EyePosition.dead);
            ChangeAnimatorIdle("head_shake");
        }
        if (Input.GetKeyDown(KeyCode.Alpha5))
        {
            ChangeEyeOffset(EyePosition.happy);
            ChangeAnimatorIdle("clap");
        }
        if (Input.GetKeyDown(KeyCode.Alpha6))
        {
            ChangeEyeOffset(EyePosition.normal);
            ChangeAnimatorIdle("shrug");
        }
        if (Input.GetKeyDown(KeyCode.Alpha7))
        {
            ChangeEyeOffset(EyePosition.angry);
            ChangeAnimatorIdle("angry");
        }
        if (Input.GetKeyDown(KeyCode.Alpha8))
        {
            ChangeEyeOffset(EyePosition.happy);
            ChangeAnimatorIdle("wave");
        }
        if (Input.GetKeyDown(KeyCode.Alpha9))
        {
            ChangeEyeOffset(EyePosition.happy);
            ChangeAnimatorIdle("excited");
        }
        if (Input.GetKeyDown(KeyCode.Alpha0))
        {
            ChangeEyeOffset(EyePosition.normal);
            ChangeAnimatorIdle("think");
        }
    }

    /// <summary>
    /// Processes the received signal to change the character's state.
    /// </summary>
    /// <param name="signal">The signal to process.</param>
    private void ProcessSignal(string signal)
    {
        switch (signal)
        {
            case "raise_hand":
                ChangeEyeOffset(EyePosition.normal);
                ChangeAnimatorIdle("point");
                ResetPosition();
                ResetOrientation();
                break;
            case "thumbs_up":
                ChangeEyeOffset(EyePosition.happy);
                ChangeAnimatorIdle("head_nod");
                ResetPosition();
                ResetOrientation();
                break;
            case "thumbs_down":
                ChangeEyeOffset(EyePosition.dead);
                ChangeAnimatorIdle("head_shake");
                ResetPosition();
                ResetOrientation();
                break;
            case "cheer":
                ChangeEyeOffset(EyePosition.happy);
                ChangeAnimatorIdle("excited");
                break;
            case "cross_arms":
                ChangeEyeOffset(EyePosition.normal);
                ChangeAnimatorIdle("shrug");
                break;
            case "clap":
                ChangeEyeOffset(EyePosition.happy);
                ChangeAnimatorIdle("clap");
                break;
            case "neutral":
                ChangeEyeOffset(EyePosition.normal);
                ChangeAnimatorIdle("idle");
                ResetPosition();
                ResetOrientation();
                break;
            default:
                Debug.LogWarning("Unknown signal received: " + signal);
                break;
        }
    }
}

    