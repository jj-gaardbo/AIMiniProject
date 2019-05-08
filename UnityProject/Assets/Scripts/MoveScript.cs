using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveScript : MonoBehaviour
{
    
    List<RaycastHit> rayHits;

    Rigidbody rb;

    public static Transform initialTransform;
    public static Vector3 initialPosition;
    public static Quaternion initialRotation;

    public float rayLength = 5f;

    public static CharacterController characterController;

    public float speed = 100f;
    public float gravity = 10.0f;

    private Vector3 moveDirection = Vector3.zero;

    // Start is called before the first frame update
    void Start()
    {
        initialTransform = transform;
        initialPosition = transform.position;
        initialRotation = transform.rotation;

        Globals.setPlayer(gameObject, initialTransform, initialPosition, initialRotation);

        GameObject goal = GameObject.FindGameObjectWithTag("Finish");
        Globals.setOriginalDistanceToGoal(Vector3.Distance(gameObject.transform.position, goal.transform.position));
    }

    // Update is called once per frame
    void Update()
    {

        Move();
        Globals.setRays(new float[] {
            CastRay(0), 
            CastRay(1), 
            CastRay(2), 
            CastRay(3), 
            CastRay(4), 
            CastRay(5), 
            CastRay(6), 
            CastRay(7),
            CastRay(8), 
            CastRay(9), 
            CastRay(10), 
            CastRay(11), 
            CastRay(12), 
            CastRay(13), 
            CastRay(14),
            CastRay(15),
            CastRay(16)
            });
        

        if(HasReachedGoal()){
            Globals.goalReached();
        }

        if(Globals.done){
            resetProgram();
        }
    }

    public static void resetProgram(){
        Debug.Log("Resetting program");
        Globals.resetPlayer();
        Globals.hasReachedGoal = false;
        Globals.done = false;
    }

    public void Move(){
        string direction = Globals.moveDir;
        if(direction == null){
            direction = "f";
        }
        GameObject player = gameObject;
        
        Vector3 moveDirection = new Vector3();

        characterController = player.GetComponent<CharacterController>();
        switch(direction){
            case "f":
                moveDirection = player.transform.TransformDirection(Vector3.forward);
                break;
            case "b":
                moveDirection = player.transform.TransformDirection(Vector3.back);
                break;
            case "l":
                moveDirection = player.transform.TransformDirection(Vector3.left);
                break;
            case "r":
                moveDirection = player.transform.TransformDirection(Vector3.right);
                break;
            case "fl":
                moveDirection = player.transform.TransformDirection(Vector3.left+Vector3.forward);
                break;
            case "fr":
                moveDirection = player.transform.TransformDirection(Vector3.right+Vector3.forward);
                break;
            case "bl":
                moveDirection = player.transform.TransformDirection(Vector3.left+Vector3.back);
                break;
            case "br":
                moveDirection = player.transform.TransformDirection(Vector3.right+Vector3.back);
                break;
        }
        moveDirection *= speed;
        characterController.Move(moveDirection * Time.deltaTime);
    }

    bool HasReachedGoal()
    {
        float minDistance = 2f;
        GameObject goal = GameObject.FindGameObjectWithTag("Finish");

        float distanceToGoal = Vector3.Distance(gameObject.transform.position, goal.transform.position);
        Globals.setDistanceToGoal(distanceToGoal);
        
        bool PlayerInArea = false;
        if(distanceToGoal < minDistance || distanceToGoal == minDistance){
            PlayerInArea = true;
        }
        return PlayerInArea;
    }



    float CastRay(int direction) {
        float length = 0f;
        Vector3 fwd = transform.TransformDirection(Vector3.forward);
        Vector3 left = transform.TransformDirection(Vector3.left);
        Vector3 right = transform.TransformDirection(Vector3.right);
        Vector3 back = transform.TransformDirection(Vector3.back);
        RaycastHit hit;
        Vector3 currentPosition = transform.position;

        switch (direction) {
            case 0:
                if (Physics.Raycast(currentPosition, fwd, out hit, rayLength)) {
                    Debug.DrawRay(currentPosition, transform.TransformDirection(fwd) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 1:
            if (Physics.Raycast(transform.position, left, out hit, rayLength)) {
                Debug.DrawRay(transform.position, transform.TransformDirection(left) * hit.distance, Color.red);
                length = hit.distance;
            }
            break;
            case 2:
                if (Physics.Raycast(transform.position, right, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(right) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 3:
                if (Physics.Raycast(transform.position, (right + fwd).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection((right + fwd).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 4:
                if (Physics.Raycast(transform.position, (left + fwd).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection((left + fwd).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 5:
                if (Physics.Raycast(transform.position, back.normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection((back).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 6:
                if (Physics.Raycast(transform.position, (right + back).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection((right + back).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 7:
                if (Physics.Raycast(transform.position, ((left+back) + left).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(((left+back) + left).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 8:
                if (Physics.Raycast(transform.position, ((right+back) + right).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(((right+back) + right).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 9:
                if (Physics.Raycast(transform.position, ((left+fwd) + fwd).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(((left+fwd) + fwd).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 10:
                if (Physics.Raycast(transform.position, ((right+fwd) + fwd).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(((right+fwd) + fwd).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 11:
                if (Physics.Raycast(transform.position, ((left+left) + back).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(((left+left) + back).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 12:
                if (Physics.Raycast(transform.position, ((right+right) + back).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(((right+right) + back).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 13:
                if (Physics.Raycast(transform.position, ((left+left) + fwd).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(((left+left) + fwd).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 14:
                if (Physics.Raycast(transform.position, ((right+right) + fwd).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection( ( (right+right) + fwd).normalized ) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 15:
                if (Physics.Raycast(transform.position, ((left+back) + back).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection( ( (left+back) + back).normalized ) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 16:
                if (Physics.Raycast(transform.position, ((right+back) + back).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection( ( (right+back) + back).normalized ) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
        }
        if (length != 0f) {
            return length;
        } else {
            return 5f;
        }
    }

}
