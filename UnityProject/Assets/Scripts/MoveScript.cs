using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveScript : MonoBehaviour
{
    public static GameObject wall1;
    public static GameObject wall2;
    public static GameObject wall3;
    public static GameObject wall4;
    public static GameObject wall5;
    public static GameObject wall6;

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

    public GameObject[] walls;


    // Start is called before the first frame update
    void Start()
    {

        walls = GameObject.FindGameObjectsWithTag("AllWalls");
        /*wall1 = GameObject.FindGameObjectsWithTag("AllWalls")[0];
        wall2 = GameObject.FindGameObjectsWithTag("AllWalls")[1];
        wall3 = GameObject.FindGameObjectsWithTag("AllWalls")[2];
        wall4 = GameObject.FindGameObjectsWithTag("AllWalls")[3];
        wall5 = GameObject.FindGameObjectWithTag("AllWalls")[4];
        wall6 = GameObject.FindGameObjectWithTag("AllWalls")[5]; */

        foreach (var wall in walls)
        {
            wall.SetActive(false);
        }
        
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
            CastRay(15)
            });
        

        if(HasReachedGoal()){
            Globals.goalReached();
        }

        if(Globals.done){
            resetProgram();
        }

        activateWall();
    }

    public void activateWall(){
        switch (Globals.numWalls)
        {
            case 1:
                if(!walls[0].activeInHierarchy)
                    walls[0].SetActive(true);
            break;
            case 2:
                if(!walls[1].activeInHierarchy)
                    walls[1].SetActive(true);
            break;
            case 3:
                if(!walls[2].activeInHierarchy)
                    walls[2].SetActive(true);
            break;
            case 4:
                if(!walls[3].activeInHierarchy)
                    walls[3].SetActive(true);
            break;
            case 5:
                if(!walls[4].activeInHierarchy)
                    walls[4].SetActive(true);
            break;
            case 6:
                if(!walls[5].activeInHierarchy)
                    walls[5].SetActive(true);
            break;
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
            return;
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
                    Debug.DrawRay(currentPosition, transform.TransformDirection(fwd) * hit.distance, Color.green);
                    length = hit.distance;
                }
            break;
            case 1:
                if (Physics.Raycast(transform.position, back.normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection((back).normalized) * hit.distance, Color.green);
                    length = hit.distance;
                }
            break;
            case 2:
            if (Physics.Raycast(transform.position, left, out hit, rayLength)) {
                Debug.DrawRay(transform.position, transform.TransformDirection(left) * hit.distance, Color.green);
                length = hit.distance;
            }
            break;
            case 3:
                if (Physics.Raycast(transform.position, right, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(right) * hit.distance, Color.green);
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
                if (Physics.Raycast(transform.position, ((left+fwd) + fwd).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(((left+fwd) + fwd).normalized) * hit.distance, Color.blue);
                    length = hit.distance;
                }
            break;
            case 6:
                if (Physics.Raycast(transform.position, ((fwd+left) + left).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection( ( (fwd+left) + left).normalized ) * hit.distance, Color.blue);
                    length = hit.distance;
                }
            break;
            case 7:
                if (Physics.Raycast(transform.position, (right + fwd).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection((right + fwd).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 8:
                if (Physics.Raycast(transform.position, ((right+fwd) + fwd).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(((right+fwd) + fwd).normalized) * hit.distance, Color.blue);
                    length = hit.distance;
                }
            break;
            case 9:
                if (Physics.Raycast(transform.position, ((fwd+right) + right).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection( ( (fwd+right) + right).normalized ) * hit.distance, Color.blue);
                    length = hit.distance;
                }
            break;
            case 10:
                if (Physics.Raycast(transform.position, (left + back).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection((left + back).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 11:
                if (Physics.Raycast(transform.position, ((left+back) + left).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(((left+back) + left).normalized) * hit.distance, Color.blue);
                    length = hit.distance;
                }
            break;
            case 12:
                if (Physics.Raycast(transform.position, ((left+back) + back).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection( ( (left+back) + back).normalized ) * hit.distance, Color.blue);
                    length = hit.distance;
                }
            break;
            case 13:
                if (Physics.Raycast(transform.position, (right + back).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection((right + back).normalized) * hit.distance, Color.red);
                    length = hit.distance;
                }
            break;
            case 14:
                if (Physics.Raycast(transform.position, ((right+back) + right).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection(((right+back) + right).normalized) * hit.distance, Color.blue);
                    length = hit.distance;
                }
            break;
            case 15:
                if (Physics.Raycast(transform.position, ((right+back) + back).normalized, out hit, rayLength)) {
                    Debug.DrawRay(transform.position, transform.TransformDirection( ( (right+back) + back).normalized ) * hit.distance, Color.blue);
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
