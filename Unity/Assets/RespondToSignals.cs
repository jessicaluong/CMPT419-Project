using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RespondToSignals : MonoBehaviour
{
    Animator animator;
    Renderer[] characterMaterials;

    public enum EyePosition { normal, happy, angry, dead }
    public EyePosition eyeState;

    private Vector3 originalPosition;

    // Boolean used for development purposes 
    public bool useTcpConnection = true;

    // Start is called before the first frame update
    public void Start()
    {
        animator = GetComponent<Animator>();
        characterMaterials = GetComponentsInChildren<Renderer>();

        originalPosition = transform.position;

    }

    // Update is called once per frame
    void Update()
    {
        if (!useTcpConnection)
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
    }

    public void ReceiveSignal(string signal)
    {
        if (useTcpConnection)
        {
            // Stop any existing "ReturnToIdle" coroutine in case a new signal is received before the previous one completes
            StopCoroutine("ReturnToIdle");

            switch (signal)
            {
                case "attention":
                    ChangeEyeOffset(EyePosition.normal);
                    ChangeAnimatorIdle("point");
                    break;
                case "approval":
                    ChangeEyeOffset(EyePosition.happy);
                    ChangeAnimatorIdle("head_nod");
                    break;
                case "disapproval":
                    ChangeEyeOffset(EyePosition.dead);
                    ChangeAnimatorIdle("head_shake");
                    break;
                case "celebratory":
                    ChangeEyeOffset(EyePosition.happy);
                    ChangeAnimatorIdle("excited");
                    break;
                case "questioning":
                    ChangeEyeOffset(EyePosition.normal);
                    ChangeAnimatorIdle("shrug");
                    break;
                case "appreciation":
                    ChangeEyeOffset(EyePosition.happy);
                    ChangeAnimatorIdle("clap");
                    break;
                case "disengagement":
                    ChangeEyeOffset(EyePosition.angry);
                    ChangeAnimatorIdle("angry");
                    break;
                case "greeting":
                    ChangeEyeOffset(EyePosition.happy);
                    ChangeAnimatorIdle("wave");
                    break;
                case "unknown":
                    ChangeEyeOffset(EyePosition.normal);
                    ChangeAnimatorIdle("think");
                    break;
                default:
                    Debug.LogWarning("Unknown signal received: " + signal);
                    break;
            }

            // Start the coroutine to return to the idle state after a delay
            StartCoroutine(ReturnToIdle());
        }
    }

    IEnumerator ReturnToIdle()
    {
        // Wait for a specified time before returning to idle
        yield return new WaitForSeconds(5);

        // Reset position and orientation
        ResetOrientation();
        ResetPosition();

        // Transition back to the idle state
        ChangeEyeOffset(EyePosition.normal);
        ChangeAnimatorIdle("idle");  
    }

    void ResetOrientation()
    {
        transform.rotation = Quaternion.Euler(new Vector3(-0.05f, 20.054f, -0.018f));
    }

    void ResetPosition()
    {
        transform.position = originalPosition;
    }


    void ChangeAnimatorIdle(string trigger)
    {
        animator.SetTrigger(trigger);
    }

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
}